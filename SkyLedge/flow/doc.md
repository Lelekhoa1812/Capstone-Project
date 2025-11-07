title OBD-II Driver Modeling & Fuel Efficiency — End-to-End Data Flow

// ---------- S1: On-Vehicle Sensing → Upload ----------
On-Vehicle Sensing | Upload [color: orange, icon: car] {
OBD-II PIDs [icon: gauge, color: yellow]
Pi Logger [icon: cpu, color: yellow]
}

// ---------- S2: Ingestion & Cleaning ----------
Ingestion | Cleaning [color: teal, icon: filter] {
Ingestion [icon: download, color: teal]
Cadence Inference [icon: clock, color: teal]
Robust Gating [icon: alert-triangle, color: teal]
Imputation [icon: cpu, color: teal, label:"KNN Imputation + local linear"]
Kinematics [icon: activity, color: teal]
Validation [icon: check-circle, color: teal]
}

// ---------- S3: Feature Engineering ----------
Feature Engineering [color: blue, icon: cog] {
Length-Invariant Stats [icon: bar-chart, color: blue, label:"freq/min | duration fraction | p90"]
High-Load Exposure [icon: zap, color: blue]
Idle Masks [icon: moon, color: blue]
Feature Schema [icon: list, color: blue]
}

// ---------- S4: Behaviour Inference ----------
Behaviour Inference [color: green, icon: brain] {
Rule Idle Baseline [icon: toggle-left, color: green, label:"Threshold rules"]
BGMM Structure [icon: layers, color: green, label:"BGMM (soft)"]
Supervised Classifier [icon: target, color: green, label:"XGBoost (encoder + scaler)"]
Temporal Smoothing [icon: shuffle, color: green, label:"k-of-n hysteresis"]
}

// ---------- S5: Fuel-Efficiency Scoring (Teacher → Student) ----------
Efficiency Scoring [color: green, icon: gauge] {
Teacher (Analytical) [icon: function-square, color: green]
Student (Monotone GBM) [icon: trending-up, color: green]
Segment Colorizer [icon: droplet, color: green]
Export Kit [icon: package, color: green, label:"thresholds | weights | sc_ml | gbm"]
}

// ---------- S6: Human-in-the-Loop (RLHF) ----------
RLHF Loop [color: red, icon: brain] {
Data Loader [icon: download, color: red]
Batch Builder [icon: layers, color: red, label:"Batch Builder disagree↑ | rare↑"]
Continuation + Distill [icon: refresh-cw, color: red]
Versioning & Promote [icon: git-branch, color: red]
}

// ---------- S7: Frontend Reporting & Coaching ----------
Frontend Dashboard [color: lightblue, icon: monitor] {
Upload UI [icon: upload-cloud, color: lightblue, label:"Upload"]
Streaming Labeler [icon: edit-3, color: lightblue, label:"Behavior Labeler"]
Model Controls [icon: sliders, color: lightblue]
Visuals & Tips [icon: image, color: lightblue]
}

// ---------- Storage & Governance ----------
Storage | Governance [color: purple, icon: database] {
Firebase Buckets [icon: cloud, color: purple, label:"Data Bucket"]
MongoDB Atlas [icon: database, color: purple, label:"runs | metadata | provenance | metrics"]
Artifacts [icon: package, color: purple]
Parity Tests [icon: check-square, color: purple]
}

// ---------- Observability ----------
Observability [color: gray, icon: activity] {
Metrics & Logs [icon: bar-chart, color: gray]
Audit Trail [icon: file-text, color: gray]
}

// ---------- Flows ----------
OBD-II PIDs > Ingestion: stream | batch upload
Pi Logger > Ingestion: session files | auth token

Ingestion > Cadence Inference: normalise | infer Δt
Cadence Inference > Robust Gating: gaps | placeholders
Robust Gating > Imputation: mark missing | extremes preserved
Imputation > Kinematics: reconstruct | derive ACCEL/JERK
Kinematics > Validation: schema | coherence

Validation > Feature Engineering: clean tables
Feature Engineering > Behaviour Inference: features+schema
Feature Engineering > Efficiency Scoring: features+idle masks

Behaviour Inference > Frontend Dashboard: labels+conf | flicker-reduced
Efficiency Scoring > Frontend Dashboard: segment colours | trip score

Frontend Dashboard > Firebase Buckets: raw/ | processed | labeled
Ingestion > Firebase Buckets: write raw/
Validation > Firebase Buckets: write processed/

Frontend Dashboard > RLHF Loop: labeled spans (provenance)
Firebase Buckets > RLHF Loop: labeled | trained

RLHF Loop > Artifacts: new model tag | metadata
Artifacts > Behaviour Inference: load pinned/latest
Artifacts > Efficiency Scoring: sc_ml | gbm | iso_cal | thresholds

MongoDB Atlas > Frontend Dashboard: session state | metrics
Behaviour Inference > Observability: per-version CV | confusion
Efficiency Scoring > Observability: parity | R²/MAE
Frontend Dashboard > Observability: UX events | labeling stats

Artifacts > Parity Tests: export kit → fixtures
Parity Tests > Versioning & Promote: pass → promote; fail → rollback

Versioning & Promote > Continuation + Distill 
Continuation + Distill <> Data Loader
Continuation + Distill <> Batch Builder