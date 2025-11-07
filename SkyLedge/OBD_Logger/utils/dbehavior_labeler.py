# dbehavior_labeler.py
# Load UL models and predict driving style
import os, logging, pickle
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy.signal import medfilt

# Import download functionality
import sys
sys.path.append(os.path.dirname(__file__))
from dbehavior_download import download_latest_models

log = logging.getLogger("dbehavior-labeler")
log.setLevel(logging.INFO)

# Suppress version compatibility warnings in production
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.base")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost.core")

MODEL_DIR = os.getenv("MODEL_DIR", "/app/models/ul")
LE_PATH   = os.path.join(MODEL_DIR, "label_encoder_ul.pkl")
SC_PATH   = os.path.join(MODEL_DIR, "scaler_ul.pkl")
XGB_PATH  = os.path.join(MODEL_DIR, "xgb_drivestyle_ul.pkl")

SAFE_DROP = {
    "timestamp","driving_style","ul_drivestyle","gt_drivestyle",
    "session_id","imported_at","record_index",
    "Fuel Efficiciency", "Fuel Efficiency (L/100KM)", "Distance", "Fuel Used", "Route",
    "Fuel consumed", "Fuel consumed (total)", "RoadType", "Road Style", "Road Type"
}

def infer_base_interval_seconds(ts: pd.Series) -> float:
    """Infer sampling cadence from timestamp diffs (robust)."""
    if ts.size < 2:
        return 1.0
    diffs = ts.sort_values().diff().dropna().dt.total_seconds()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return 1.0
    q05, q95 = diffs.quantile([0.05, 0.95])
    core = diffs[(diffs >= q05) & (diffs <= q95)]
    rounded = (core / 0.01).round() * 0.01
    mode = rounded.mode()
    est = float(mode.iloc[0]) if not mode.empty else float(core.median())
    return max(est, 1e-3)

def rows_for(seconds, med_dt):
    return max(3, int(round(seconds / max(med_dt, 1e-3))))

def safe_numeric(df, skip=set()):
    out = df.copy()
    for c in out.columns:
        if c in skip: continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def engineer_features(df):
    """
    Recreate the exact feature engineering pipeline used during training
    """
    log.info("🔧 Starting feature engineering...")
    fe = df.copy()
    
    # Clean sentinel values
    log.info("🧹 Cleaning sentinel values...")
    SENTINELS = {-22, -40, 255}
    fe.replace(list(SENTINELS), np.nan, inplace=True)
    
    # Ensure timestamp is proper datetime
    if "timestamp" in fe.columns:
        fe["timestamp"] = pd.to_datetime(fe["timestamp"], errors="coerce", utc=True)
    
    # Define non-sensor columns that should be excluded from feature engineering
    # These are metadata/calculated fields that shouldn't be used as features
    NON_SENSOR_COLS = {
        "timestamp", "driving_style", "ul_drivestyle", "gt_drivestyle",
        "session_id", "imported_at", "record_index",
        "Fuel Efficiciency", "Fuel Efficiency (L/100KM)", "Distance", "Fuel Used", "Route",
        "Fuel consumed", "Fuel consumed (total)", "RoadType", "Road Style", "Road Type",
        "Fuel consumed (total)", "Road Type", "Road Style"
    }
    
    # Convert all numeric columns safely, excluding non-sensor columns
    fe = safe_numeric(fe, skip=NON_SENSOR_COLS)
    
    # Fill NaN values with median for sensor columns only
    sensor_cols = [c for c in fe.select_dtypes(include=[np.number]).columns 
                   if c not in NON_SENSOR_COLS]
    fe[sensor_cols] = fe[sensor_cols].fillna(fe[sensor_cols].median())
    
    # Estimate sampling period
    if "timestamp" in fe.columns:
        base_sec = infer_base_interval_seconds(fe["timestamp"])
    else:
        base_sec = 1.0
    med_dt = base_sec

    # Define window sizes (must match training)
    W1 = rows_for(1.0, med_dt)
    W2 = rows_for(2.0, med_dt)
    W5 = rows_for(5.0, med_dt)
    W8 = rows_for(8.0, med_dt)

    log.info(f"Using window sizes: W1={W1}, W2={W2}, W5={W5}, W8={W8}")

    # Base signals - use available columns
    available_base = [c for c in ["SPEED", "RPM", "ENGINE_LOAD", "ABSOLUTE_LOAD", "THROTTLE_POS", "MAF"] 
                     if c in fe.columns]
    log.info(f"Available base signals: {available_base}")

    # Kinematics
    if "SPEED" in fe.columns:
        fe["ACCEL"] = fe["SPEED"].diff() / max(med_dt, 1e-3)
        fe["JERK"] = fe["ACCEL"].diff() / max(med_dt, 1e-3)
    else:
        fe["ACCEL"] = 0.0
        fe["JERK"] = 0.0

    # Throttle change rate
    if "THROTTLE_POS" in fe.columns:
        fe["THROTTLE_D"] = fe["THROTTLE_POS"].diff() / max(med_dt, 1e-3)
    else:
        fe["THROTTLE_D"] = 0.0

    # Rolling stats at multiple horizons (CRITICAL - this creates most features)
    def add_roll(col):
        if col not in fe.columns: 
            return
        # Safety check to prevent infinite recursion - only skip if it already has rolling suffix
        if any(col.endswith(suffix) for suffix in ['_mean_w1', '_std_w1', '_mean_w2', '_std_w2', '_mean_w5', '_std_w5', '_mean_w8', '_std_w8', '_min_w1', '_max_w1']):
            return  # Skip if already a rolling feature
        for w, tag in [(W1, "w1"), (W2, "w2"), (W5, "w5"), (W8, "w8")]:
            fe[f"{col}_mean_{tag}"] = fe[col].rolling(w, min_periods=1, center=True).mean()
            fe[f"{col}_std_{tag}"] = fe[col].rolling(w, min_periods=1, center=True).std()
            fe[f"{col}_min_{tag}"] = fe[col].rolling(w, min_periods=1, center=True).min()
            fe[f"{col}_max_{tag}"] = fe[col].rolling(w, min_periods=1, center=True).max()

    # Apply rolling features to all base signals and derived signals
    all_signals = available_base + ["ACCEL", "JERK", "THROTTLE_D"]
    for col in all_signals:
        add_roll(col)

    # Additional derived features that might have been used in training
    if {"MAF", "RPM"}.issubset(fe.columns):
        fe["AIRFLOW_PER_RPM"] = fe["MAF"] / fe["RPM"].replace(0, np.nan)
        add_roll("AIRFLOW_PER_RPM")

    if {"ENGINE_LOAD", "THROTTLE_POS"}.issubset(fe.columns):
        fe["LOAD_THROTTLE_RATIO"] = fe["ENGINE_LOAD"] / fe["THROTTLE_POS"].replace(0, np.nan)
        add_roll("LOAD_THROTTLE_RATIO")

    # Event rates based on quantiles
    def q(x, p, default):
        return fe[x].abs().quantile(p) if x in fe.columns else default

    if "ACCEL" in fe.columns:
        a_pos = q("ACCEL", 0.85, 0.5)
        a_neg = q("ACCEL", 0.15, -0.5)
        evt = pd.DataFrame(index=fe.index)
        evt["pos_accel_rate_w5"] = (fe["ACCEL"] > a_pos).rolling(W5, min_periods=1, center=True).mean()
        evt["neg_accel_rate_w5"] = (fe["ACCEL"] < a_neg).rolling(W5, min_periods=1, center=True).mean()
        fe = pd.concat([fe, evt], axis=1)

    if "THROTTLE_D" in fe.columns:
        thr_q = q("THROTTLE_D", 0.85, 1.0)
        fe["thr_change_rate_w5"] = (fe["THROTTLE_D"].abs() > thr_q).rolling(W5, min_periods=1, center=True).mean()

    # Magnitude features
    if "ACCEL" in fe.columns:
        fe["ACCEL_MAG"] = fe["ACCEL"].abs()
        add_roll("ACCEL_MAG")

    # Fill any remaining NaN values
    fe = fe.bfill().ffill().fillna(0)
    
    log.info(f"Engineered features shape: {fe.shape}")
    log.info(f"Total features created: {len(fe.columns)}")
    
    return fe

def detect_idle_episodes(fe):
    """
    Detect idle episodes using the same logic as during training
    """
    def pick(*names, default=None):
        for n in names:
            if n in fe.columns: 
                return fe[n]
        return pd.Series(default, index=fe.index, dtype=float)

    # Use rolling means for stability
    speed_mean = pick("SPEED_mean_w5", "SPEED", default=0.0)
    thr_mean = pick("THROTTLE_POS_mean_w5", "THROTTLE_POS", default=0.0)
    acc_std = pick("ACCEL_std_w5", "ACCEL_std_w2", "ACCEL_std_w1", default=0.0)
    rpm_std = pick("RPM_std_w5", "RPM", default=0.0)
    if rpm_std.sum() > 0:  # Only apply rolling std if there's actual data
        rpm_std = rpm_std.rolling(5, min_periods=1).std()
    maf_mean = pick("MAF_mean_w5", "MAF", default=0.0)

    # Quantile-based gating
    s_gate = speed_mean <= speed_mean.quantile(0.15)
    t_gate = thr_mean.fillna(0) <= thr_mean.quantile(0.20)
    a_gate = acc_std.fillna(0) <= acc_std.quantile(0.25)
    r_gate = rpm_std.fillna(0) <= rpm_std.quantile(0.25)
    m_gate = maf_mean.fillna(0) <= maf_mean.quantile(0.20)
    
    idle_mask = (s_gate & t_gate & a_gate & r_gate & m_gate)
    
    # Smooth the mask
    idle_mask = medfilt(idle_mask.astype(int), kernel_size=5).astype(bool)
    
    return idle_mask

def _load_any(path):
    # Suppress version compatibility warnings for production
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
        warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
        warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")
        try:
            model = joblib.load(path)
        except Exception:
            with open(path, "rb") as f:
                model = pickle.load(f)
    
    # Fix XGBoost compatibility issues for older trained models
    if hasattr(model, 'get_booster'):  # This is an XGBoost model
        # Remove deprecated use_label_encoder attribute that causes issues in newer XGBoost versions
        if hasattr(model, '__dict__'):
            # Remove all deprecated attributes that cause issues
            deprecated_attrs = [
                'use_label_encoder', '_le', '_label_encoder', 
                'use_label_encoder_', '_le_', '_label_encoder_'
            ]
            for attr in deprecated_attrs:
                model.__dict__.pop(attr, None)
            
            # Set use_label_encoder to False for newer XGBoost versions
            if hasattr(model, 'set_params'):
                try:
                    model.set_params(use_label_encoder=False)
                except Exception:
                    pass
    
    return model

class ULLabeler:
    _instance = None

    def __init__(self, auto_download: bool = True):
        # Auto-download latest models if enabled
        if auto_download:
            log.info("🔄 Checking for latest model version...")
            try:
                download_latest_models()
            except Exception as e:
                log.warning(f"⚠️ Failed to download latest models: {e}")
        
        if not (os.path.exists(LE_PATH) and os.path.exists(SC_PATH) and os.path.exists(XGB_PATH)):
            raise FileNotFoundError("Model files not found. Ensure download.py ran successfully.")
        self.le   = _load_any(LE_PATH)
        self.scal = _load_any(SC_PATH)
        self.clf  = _load_any(XGB_PATH)

        # Additional XGBoost compatibility fixes
        self._fix_xgb_compatibility()

        # Try to discover expected feature names from scaler or model
        self.expected = None
        if hasattr(self.scal, "feature_names_in_"):
            self.expected = list(self.scal.feature_names_in_)
        elif hasattr(self.clf, "feature_names_in_"):
            self.expected = list(self.clf.feature_names_in_)

        log.info(f"ULLabeler ready | expected_features={len(self.expected) if self.expected else 'unknown'}")

    def _fix_xgb_compatibility(self):
        """Fix XGBoost compatibility issues with older trained models."""
        try:
            # Check if this is an XGBoost classifier
            if hasattr(self.clf, 'get_booster'):
                # Remove deprecated attributes that cause issues in newer XGBoost versions
                deprecated_attrs = [
                    'use_label_encoder', '_le', '_label_encoder',
                    'use_label_encoder_', '_le_', '_label_encoder_'
                ]
                for attr in deprecated_attrs:
                    if hasattr(self.clf, attr):
                        try:
                            delattr(self.clf, attr)
                        except (AttributeError, TypeError):
                            pass
                
                # Set use_label_encoder to False for newer XGBoost versions
                if hasattr(self.clf, 'set_params'):
                    try:
                        self.clf.set_params(use_label_encoder=False)
                    except Exception:
                        pass
                
                # Ensure the model is properly configured for prediction
                if hasattr(self.clf, 'n_classes_') and self.clf.n_classes_ is None:
                    # Try to infer number of classes from the label encoder
                    if hasattr(self.le, 'classes_'):
                        self.clf.n_classes_ = len(self.le.classes_)
                
                # For newer XGBoost versions, ensure the model is properly initialized
                if hasattr(self.clf, '_le') and self.clf._le is None:
                    self.clf._le = None
                
                log.info("XGBoost compatibility fixes applied successfully")
        except Exception as e:
            log.warning(f"XGBoost compatibility fix failed: {e}")

    @classmethod
    def get(cls, auto_download: bool = True):
        if cls._instance is None:
            cls._instance = ULLabeler(auto_download=auto_download)
        return cls._instance

    def _prepare(self, df: pd.DataFrame):
        """
        Prepare features using the exact same pipeline as training
        """
        log.info("🔧 Starting feature engineering pipeline...")
        
        # Step 1: Engineer features to match training set
        engineered_df = engineer_features(df)
        
        # Step 2: Get the feature names the model expects
        try:
            # Try to get feature names from model
            training_columns = self.clf.get_booster().feature_names
            if training_columns is None:
                # Try to get from scaler
                if hasattr(self.scal, 'feature_names_in_'):
                    training_columns = self.scal.feature_names_in_.tolist()
                else:
                    raise ValueError("Cannot determine feature names")
        except:
            # Final fallback - use all numeric columns from engineered data
            training_columns = engineered_df.select_dtypes(include=[np.number]).columns.tolist()

        log.info(f"Model expects {len(training_columns)} features")
        
        # Step 3: Align features with what model expects
        missing_in_data = set(training_columns) - set(engineered_df.columns)
        extra_in_data = set(engineered_df.columns) - set(training_columns)
        
        log.info(f"Missing features in data: {len(missing_in_data)}")
        log.info(f"Extra features in data: {len(extra_in_data)}")
        
        # Define non-sensor features that should be excluded
        NON_SENSOR_FEATURES = {
            "Fuel consumed", "Fuel Efficiency (L/100KM)", "RoadType", "Road Style", "Road Type",
            "Fuel consumed (total)", "Fuel Efficiciency", "Distance", "Fuel Used", "Route"
        }
        
        # Create final feature matrix
        X_final = pd.DataFrame(index=engineered_df.index)
        
        # Add expected features, handling missing features appropriately
        for col in training_columns:
            if col in engineered_df.columns:
                X_final[col] = engineered_df[col]
            elif col in NON_SENSOR_FEATURES:
                # These are metadata/calculated features that shouldn't be used
                log.info(f"ℹ️  Excluding non-sensor feature: {col}")
                X_final[col] = 0.0  # Fill with 0 but don't warn
            else:
                # This is a missing sensor-derived feature
                X_final[col] = 0.0
                log.warning(f"⚠️  Added missing sensor feature: {col} (filled with 0)")
        
        # Ensure correct order
        X_final = X_final[training_columns]
        
        log.info(f"Final feature matrix shape: {X_final.shape}")
        
        # Step 4: Scale features
        try:
            Xs = self.scal.transform(X_final)
        except Exception as e:
            log.warning(f"Scaler transform failed ({e}); using raw features.")
            Xs = X_final.values
            
        return Xs, engineered_df

    def predict_df(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict driving styles with proper feature engineering and idle detection
        """
        try:
            log.info("🔧 Starting UL prediction pipeline...")
            
            # Step 1: Prepare features using training pipeline
            log.info("📊 Step 1: Feature engineering...")
            Xs, engineered_df = self._prepare(df)
            log.info("✅ Feature engineering completed")
            
            # Step 2: Make predictions
            log.info("🎯 Step 2: Making predictions...")
            try:
                predictions_encoded = self.clf.predict(Xs)
                log.info("✅ Model predictions completed")
            except Exception as e:
                log.error(f"❌ Model prediction failed: {e}")
                raise
            
            # Step 3: Convert encoded predictions to labels
            log.info("🏷️ Step 3: Converting predictions to labels...")
            try:
                predictions_labels = self.le.inverse_transform(predictions_encoded)
                log.info("✅ Label conversion completed")
            except Exception:
                predictions_labels = predictions_encoded
                log.info("✅ Using raw predictions as labels")
            
            # Step 4: Detect idle episodes and override predictions
            log.info("🔍 Step 4: Detecting idle episodes...")
            try:
                idle_mask = detect_idle_episodes(engineered_df)
                idle_count = idle_mask.sum()
                total_count = len(idle_mask)
                log.info(f"✅ Idle detection completed: {idle_count} ({idle_count/total_count*100:.1f}%)")
            except Exception as e:
                log.error(f"❌ Idle detection failed: {e}")
                # Fallback: no idle detection
                idle_mask = np.zeros(len(predictions_labels), dtype=bool)
                log.warning("⚠️ Using fallback: no idle detection")
            
            # Step 5: Override predictions for idle samples
            log.info("🔄 Step 5: Applying idle overrides...")
            final_predictions = predictions_labels.copy()
            final_predictions[idle_mask] = "Idle"
            log.info("✅ Idle overrides applied")
            
            # Step 6: Log prediction distribution
            log.info("📊 Step 6: Logging results...")
            style_proportions = pd.Series(final_predictions).value_counts(normalize=True).sort_index()
            log.info("📊 FINAL PREDICTION RESULTS:")
            for style, prop in style_proportions.items():
                count = (final_predictions == style).sum()
                log.info(f"  {style:15}: {prop:.3f} ({prop*100:.1f}%) [{count} samples]")
            
            log.info("✅ UL prediction pipeline completed successfully")
            return final_predictions
            
        except Exception as e:
            log.error(f"❌ UL prediction pipeline failed: {e}")
            import traceback
            log.error(f"❌ Traceback: {traceback.format_exc()}")
            raise

    def predict_csv(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        y = self.predict_df(df)
        out = df.copy()
        out["driving_style"] = y
        return out
