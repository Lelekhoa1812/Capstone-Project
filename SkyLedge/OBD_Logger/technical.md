## Technical Report: Machine Learning and Mathematical Methods for OBD-II Driver Modeling and Fuel Efficiency

### 1. Time-Series Data Cleaning and Reconstruction

This pipeline addresses missingness, corruption, and extreme outliers in multi-sensor OBD-II streams with heterogeneous sampling. Two complementary estimators are employed: instance-based local geometry via KNN imputation, and temporal structure via linear regression across time. A robust outlier detection stage gates imputations.

- Outlier gating: For each numeric sensor series x(t), compute robust z-score with median m and MAD s = 1.4826·MAD(x): z(t) = |x(t) − m|/max(ε, s). Values with z(t) > τ are flagged as outliers and treated as missing. Temporal spikes are additionally flagged if violating Lipschitz-like bounds on first differences Δx(t) = x(t) − x(t − Δt), relative to rolling quantiles.

- KNNImputer: Given feature vector at timestamp t, v(t) ∈ R^d with missing components, find a neighborhood N_k(t) among valid instances using a masked distance d(v_i, v_j) = sqrt(Σ_{p∈Obs(i)∩Obs(j)} w_p (v_{ip} − v_{jp})^2) where w_p are sensor-specific normalizers (e.g., inverse variance). Imputed component p is the distance-weighted average over neighbors with observed p: 
  \( \hat v_{tp} = \frac{\sum_{j\in N_k(t) \cap Obs(p)} K(d(v_t, v_j))\, v_{jp}}{\sum_{j\in N_k(t) \cap Obs(p)} K(d(v_t, v_j))} \), with kernel K(r) = 1/(r+ε).

- Linear regression on time: For each sensor independently, fit local linear models on a rolling window W around t over valid samples {(τ, x(τ))}: x(τ) ≈ a t + b, via least-squares \( (a, b) = \arg\min_{a,b} \sum_{τ\in W} (x(τ) - a τ - b)^2 \). The imputed value at missing t is \( \hat x(t) = a t + b \). When two valid anchors (t1, x1), (t2, x2) straddle t, this reduces to linear interpolation: \( \hat x(t) = x_1 + (x_2 - x_1) \frac{t - t_1}{t_2 - t_1} \).

- Hybrid imputation selection: For each missing entry, both estimates are computed; the final choice is selected by a confidence rule combining neighborhood density |N_k| and residual diagnostic from the local fit. Tie-breaking favors the regressor when temporal smoothness constraints are stronger, otherwise KNN.

Evaluation against baselines found an average absolute reconstruction error ≈ 2% relative to ground-truth holdouts, outperforming mean/median imputation (≈ 10–20%).


### 2. Semi-Supervised Driver Behavior Modeling (notebook/W5S2.ipynb)

The semi-supervised pipeline integrates limited ground-truth labels with structure learned from large unlabeled OBD sequences, combining rolling-window feature engineering, Bayesian mixture modeling, and confidence-weighted boosting.

- Feature construction: For each sensor s ∈ {SPEED, RPM, ENGINE_LOAD, THROTTLE_POS, MAF} and derived kinematics ACCEL, JERK, define multi-horizon rolling statistics over windows W = {w1, w2, w5, w8} determined by sampling cadence Δt via wΔ = round(Δ/Δt). For any series x_t:
  - Means: \( \mu_{x,\Delta}(t) = \mathrm{mean}(x_{t-w_\Delta+1:t}) \)
  - Standard deviations: \( \sigma_{x,\Delta}(t) = \mathrm{std}(x_{t-w_\Delta+1:t}) \)
  - Extrema: \( x^{\min}_{\Delta}(t), x^{\max}_{\Delta}(t) \) as rolling min/max
  Interaction ratios include AIRFLOW_PER_RPM = MAF/RPM and LOAD_THROTTLE_RATIO = ENGINE_LOAD/THROTTLE_POS (with safe division). Event-rate features are built via quantile gates (e.g., P(|ACCEL| > q_0.85) over w5).

- Supervised anchors: Human-labeled styles y ∈ {Aggressive, Moderate, Passive, Idle} sparsely annotate sessions. These anchors set class semantics and calibrate later steps.

- Bayesian GMM (BGMM): Fit K-component mixtures p(x) = Σ_k π_k N(x | μ_k, Σ_k) with conjugate priors (Dirichlet on π, Normal–Wishart on (μ_k, Σ_k)). Responsibilities are
  \[ \gamma_{nk} = p(z=k|x_n) = \frac{\pi_k \; \mathcal{N}(x_n\,|\,\mu_k, \Sigma_k)}{\sum_j \pi_j \; \mathcal{N}(x_n\,|\,\mu_j, \Sigma_j)} \]
  Parameters maximize the ELBO:
  \[ \mathcal{L} = \mathbb{E}_q[\log p(X, Z, \{\mu,\Sigma,\pi\})] - \mathbb{E}_q[\log q(Z, \{\mu,\Sigma,\pi\})] \]
  yielding soft partitions tolerant to multi-modality and covariance anisotropy.

- Cluster-to-label alignment and pseudo-labeling: Let C be clusters and Y anchor labels. Solve assignment A: C→Y to maximize mutual information I(C;Y) subject to anchors (Hungarian matching over −I). For each n, a pseudo-label \( \tilde y_n = \arg\max_k \gamma_{nk} \) is admitted if confidence \( c_n = \max_k \gamma_{nk} \ge \tau \). Temperature sharpening enforces peaky posteriors: \( \gamma'_{nk} \propto \gamma_{nk}^{1/T} \) (T∈(0,1]). Sample weights combine anchor primacy and confidence: w_n = α·1[n is anchor] + β·c_n·1[n is pseudo], with α > β.

- Classifier training: A multi-class gradient-boosted tree model f(x; θ) minimizes weighted cross-entropy with regularization:
  \[ \min_θ \; \sum_{n} w_n\,\mathrm{CE}(y_n, f(x_n; θ)) + \lambda_1 \|θ\|_1 + \lambda_2 \|θ\|_2^2 \]
  Early stopping monitors a stratified validation split on anchors; trees, depth, learning rate, and subsampling control margin growth and variance.

- Idle handling: A post-hoc idle detector based on joint low SPEED, THROTTLE_POS, ENGINE_LOAD, MAF, and low ACCEL variance (smoothed with median filtering) overwrites predictions to Idle in stationary segments—reducing false positives from micro-fluctuations.

- Outcomes: The pipeline yields stable class boundaries with improved calibration from soft-label integration. In practice, cross-validated accuracy and confusion structure on anchors remain strong while recall on underrepresented styles improves due to BGMM-driven augmentation. Artifacts include the label encoder, a scaler with feature_names_in_, and the trained booster with feature schema locking for reproducible inference.


### 3. Driver Behavior RLHF (Reinforcement Learning from Human Feedback)

Human feedback continually improves the classifier with three fine-tuning strategies that preserve prior knowledge and adapt to shifting behavior distributions.

- Preference-aligned training data: For new sessions with human labels, features X and labels y are prepared using the exact training schema. The existing model f_old supplies predictions and confidences on original sequences, enabling preference signals between y and f_old(x). The RLHF dataset emphasizes human-consistent regions and disagreement pockets.

- Continuation fine-tuning: Create an enhanced representation \( \phi(x) = [x, f_{old}(x)] \) using predicted class-probabilities. Train a small-learning-rate booster f_new on \( \phi(x) \) with limited trees to update margins while preserving past decision structure. This acts as a functional Newton step constrained by the old logits.

- Knowledge distillation: Use soft targets s = f_old(x) as additional guidance with temperature scaling. Optimize \( \mathcal{L} = \sum_n \mathrm{CE}(y_n, f(x_n)) + \mu \cdot \mathrm{KL}(s_n \parallel f(x_n)) \), encouraging the new model to respect prior soft boundaries unless contradicted by human labels.

- Ensemble preservation: Maintain an ensemble g(x) = w·f_old(x) + (1−w)·f_new(x) at the probability level for calibrated stability during transitions. In practice, probabilities are averaged to reduce variance in low-support regions.

- Evaluation and selection: Accuracy and 5-fold cross-validated mean±std are logged on the updated dataset. The new model, scaler, and label encoder are versioned; expected_features guard against schema drift. Compatibility adapters remove deprecated attributes from legacy boosters to ensure reproducibility across library versions.

This RLHF loop aligns predictions with evolving user expectations, improves decision boundaries where disagreement concentrates, and maintains continuity via continuation/distillation penalties rather than wholesale retraining.


### 4. Fuel Efficiency Teacher: Hybrid Analytical–ML Formulation (notebook/W8S2.ipynb)

An analytical “teacher” maps sequence-level operating patterns to a drive-level efficiency target, combining robust thresholds, event segmentation, penalty shaping, and a monotone link; a student regressor is trained against these targets and calibrated for deployment.

- Temporal kinematics and base cadence: Convert speed to m/s: SPEED_ms = SPEED·(1000/3600). Let Δt* = median positive inter-sample interval. Define ACCEL_t = ΔSPEED_ms/Δt*, JERK_t = ΔACCEL/Δt*. Distance integrates SPEED_ms: dist = Σ_t SPEED_ms(t)·Δt_t.

- Fleet thresholds (robust): For a combined dataset, define
  - RPM90 = Q0.90(RPM), MAF90 = Q0.90(MAF)
  - THR_Q85 = Q0.85(THROTTLE_POS), LOAD_Q85 = Q0.85(ENGINE_LOAD)
  - ACCEL_LOW_Q20 = Q0.20(|ACCEL|), ACCEL_HIGH_Q85 = Q0.85(|ACCEL|)
  - JERK_HIGH_Q90 = Q0.90(|JERK|)
  - SPEED_IDLE_MPS ≈ 0.6
  Idle rule: joint predicate of low SPEED_ms, THROTTLE_POS, ENGINE_LOAD, MAF, and |ACCEL| (smoothed by a rolling median) to produce a stable idle mask.

- Sharp-event segmentation: A sharp segment is a maximal contiguous run where |ACCEL| > ACCEL_HIGH_Q85 or |JERK| > JERK_HIGH_Q90. For each drive i with duration T_i and minutes M_i = T_i/60:
  - Frequency: freq_i = (#segments)/M_i
  - Duration fraction: durfrac_i = (Σ_lengths·Δt*)/T_i
  - Peak magnitudes: for each segment j, compute peaks (|ACCEL|_max, |JERK|_max), summarize by mean over segments.

- Idle and smoothness metrics: Idle fraction idle_i = mean(mask_idle); median idle run-length; idle events per minute; rolling speed coefficient of variation CV_i = mean_t(std_w10(SPEED_ms)/mean_w10(SPEED_ms)). High-load fractions: frac_rpm90 = P(RPM ≥ RPM90), frac_maf90 = P(MAF ≥ MAF90).

- Robust penalty shaping: For any metric s with empirical quartiles (q25, q50, q75), define scale \( \hat s = (q_{75}-q_{25})/1.349 \) and penalty
  \[ P(s) = \sigma\Big(\frac{s - q_{50}}{\max(\varepsilon, \hat s)}\Big) = \frac{1}{1 + e^{-(s - q_{50})/\max(\varepsilon, \hat s)}} \]
  Apply to: p_freq, p_dur, p_mag (sharpness magnitude), p_idle (mixture of idle fraction and run-length), p_cv (speed CV), p_rpm, p_maf.

- Proxy link and inefficiency model: Define a bounded proxy combining high-load and idle contributions, proxy ∈ [0, 1). Apply the monotone link
  \[ y^* = -\ln(1 - \mathrm{proxy}) \]
  With penalty design matrix P ∈ R^{n×m} (rows=drives, cols=penalties), solve least-squares weights
  \[ w = \arg\min_w \; \|P w - y^*\|_2^2, \quad \text{solution } w = (P^\top P)^{-1} P^\top y^* \]
  Inefficiency: \( I = 1 - e^{-P w} \). Efficiency score: \( E = 100\,(1 - I) \in [0, 100] \).

- Student training: For each drive, aggregate features: duration, distance, quantiles of SPEED_ms, |ACCEL|, |JERK|, sharp-event frequency, idle metrics, high-load fractions, and interaction proxy fuel_intensity ≈ Q0.90(RPM)·Q0.90(MAF). After scaling numeric subsets, train a student regressor (HistGradientBoosting; RandomForest fallback if raw variance collapses) with GroupKFold splits (by source) and duration-based weights w_i = max(0.5, duration_min_i) to reduce small-trip noise.

- Quantile-mapping calibration: Let r be out-of-fold raw predictions; for quantiles q ∈ {0.05,…,0.95}, compute (r_q, y_q) between r and teacher E. Enforce strictly increasing r_q and define calibrated prediction
  \[ \hat E = \mathrm{clip}\big(\mathrm{interp}(r; \{r_q\}, \{y_q\}),\; 0, 100\big) \]
  ensuring monotonic alignment and bounded outputs.

- Outcomes: Training reports out-of-fold mean absolute error (after quantile-mapping) and rank/linear correlation between student and teacher targets, with healthy variance captured across groups. Calibration compresses tail bias and stabilizes deployment predictions. Artifacts persist the scaler, feature schema, thresholds, calibration knots, model kind, and OOF statistics for reproducibility and audits.


### 5. Fuel Efficiency Prediction: Student Enhancement from the Teacher

The deployed efficiency predictor is a calibrated student model distilled from the analytical teacher and improved via repeated retraining as new drives accumulate.

- Inference pipeline: For a drive, compute kinematics and features, align to the stored feature schema (adding zeros for absent features), transform numeric columns with the saved scaler, predict a raw score with the trained model, and apply quantile-mapping calibration to obtain \( \hat E \in [0,100] \).

- Continual improvement: Periodically, thresholds are recomputed over the enlarged fleet, the teacher re-derives targets E_teacher from updated penalties, and the student is re-trained with GroupKFold and duration-based weights. This teacher–student reinforcement scheme incrementally refines the student’s mapping as operating distributions drift and as the teacher’s robust statistics stabilize on larger samples.

- Robustness measures: Zero-variance feature pruning, schema alignment via feature_names_in_, and strictly increasing calibration quantiles prevent degeneracies. When gradient boosting underfits with near-constant raw outputs, a ridge-rescue is used within folds and a RandomForest fallback is used for the final model.


### 6. Summary of Algorithms and Objectives

- Imputation: KNN with masked distances and kernel weighting; local linear temporal regression; robust outlier gating; hybrid selector. Error ≈ 2% versus 10–20% for mean/median imputation.

- Semi-supervised behavior: BGMM for latent structure with soft assignments; high-confidence label propagation; multi-class gradient boosting with weighted anchors and pseudo-labels; L1/L2 regularization and early stopping.

- RLHF classifier updates: Continuation fine-tuning on concatenated [x, f_old(x)]; knowledge distillation via CE + μ·KL; ensemble probability blending for stability; cross-validated selection with versioned artifacts.

- Efficiency teacher–student: Analytical penalties with robust logistic shaping, link y* = −ln(1−proxy), least-squares weights for penalties, inefficiency I = 1 − e^{−Pw}, calibrated student regressor with GroupKFold and duration weights, quantile-mapping calibration to [0,100].


