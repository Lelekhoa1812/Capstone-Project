"""
Fuel Efficiency Labeler
Provides fuel efficiency scoring for OBD data using the trained model
Similar to utils/dbehavior_labeler.py but for fuel efficiency scoring
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

logger = logging.getLogger("efficiency-labeler")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s"))
    logger.addHandler(handler)

# Constants
KMH_TO_MS = 1000.0/3600.0
SEED = 42

class EfficiencyLabeler:
    """
    Fuel efficiency scorer for OBD data using machine learning model.
    Provides drive-level efficiency scores (0-100%) for entire drives.
    """
    
    _instance = None
    _model_artifacts = None
    _metadata = None
    _initialized = False
    
    def __init__(self):
        if not EfficiencyLabeler._initialized:
            self._load_model()
            EfficiencyLabeler._initialized = True
        EfficiencyLabeler._instance = self
    
    @classmethod
    def get(cls):
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _load_model(self):
        """Load the efficiency model and metadata"""
        try:
            from utils.efficiency_download import load_efficiency_model, check_efficiency_model_exists
            
            # Check if model exists locally
            if not check_efficiency_model_exists():
                logger.warning("⚠️ Efficiency model not found locally, attempting download...")
                from utils.efficiency_download import download_latest_efficiency_models
                success = download_latest_efficiency_models()
                if not success:
                    raise RuntimeError("Failed to download efficiency model")
            
            # Load model
            model_artifacts, metadata = load_efficiency_model()
            if model_artifacts is None:
                raise RuntimeError("Failed to load efficiency model")
            
            EfficiencyLabeler._model_artifacts = model_artifacts
            EfficiencyLabeler._metadata = metadata
            
            logger.info(f"✅ Efficiency model loaded | kind: {model_artifacts.get('model_kind', 'unknown')}")
            logger.info(f"📊 Model features: {len(model_artifacts.get('feature_names', []))}")
            
        except Exception as e:
            logger.error(f"❌ Error loading efficiency model: {e}")
            raise
    
    def _ensure_dt(self, s):
        """Ensure datetime conversion"""
        return pd.to_datetime(s, errors="coerce")
    
    def _infer_base_interval_seconds(self, ts, fallback=1.0):
        """Infer base interval from timestamps"""
        ts = pd.to_datetime(ts, errors="coerce")
        dt = ts.diff().dt.total_seconds().dropna()
        med = float(np.nanmedian(dt)) if len(dt) else fallback
        return fallback if (not np.isfinite(med) or med <= 0) else med
    
    def _rows_for(self, seconds, base_sec):
        """Calculate number of rows for given time window"""
        return max(3, int(round(seconds / max(1e-3, base_sec))))
    
    def _add_basic_derivatives(self, d):
        """Add basic derivatives (acceleration, jerk, distance)"""
        d = d.copy()
        d["timestamp"] = self._ensure_dt(d["timestamp"])
        d = d.dropna(subset=["timestamp"]).sort_values("timestamp")
        base = self._infer_base_interval_seconds(d["timestamp"], 1.0)
        
        # Convert numeric columns
        for c in ["SPEED","RPM","MAF","ENGINE_LOAD","THROTTLE_POS"]:
            if c in d.columns: 
                d[c] = pd.to_numeric(d[c], errors="coerce")
        
        # Convert speed to m/s
        if "SPEED_ms" not in d.columns:
            d["SPEED_ms"] = (d["SPEED"] * KMH_TO_MS) if "SPEED" in d.columns else np.nan
        
        # Calculate derivatives
        d["ACCEL"] = d["SPEED_ms"].diff()/max(base,1e-3)
        d["JERK"] = d["ACCEL"].diff()/max(base,1e-3)
        
        # Calculate distance
        dt = d["timestamp"].diff().dt.total_seconds().fillna(0).clip(lower=0, upper=10*base)
        d["dist_m"] = d["SPEED_ms"] * dt
        
        return d
    
    def _idle_rule(self, d, thr):
        """Apply idle detection rule"""
        speed_low = (d["SPEED_ms"].abs() <= thr.get("SPEED_IDLE_MPS", 0.6))
        thr_low = (d["THROTTLE_POS"] <= thr.get("THR_LOW_Q10", 0.0)) if "THROTTLE_POS" in d else True
        load_low = (d["ENGINE_LOAD"] <= thr.get("LOAD_LOW_Q15", 0.0)) if "ENGINE_LOAD" in d else True
        maf_low = (d["MAF"] <= thr.get("MAF_LOW_Q10", 0.0)) if "MAF" in d else True
        accel_low = (d["ACCEL"].abs() <= thr.get("ACCEL_LOW_Q20", 0.0))
        
        mask = (speed_low & thr_low & load_low & maf_low & accel_low).astype(int)
        k = 5
        return (mask.rolling(k, center=True, min_periods=1).median().round().astype(bool)
                if len(mask) >= k else mask.astype(bool))
    
    def _sharp_mask_from_thresholds(self, d, thr):
        """Detect sharp acceleration/deceleration events"""
        thr_a = thr.get("ACCEL_HIGH_Q85",
                       np.nanquantile(d["ACCEL"].abs().dropna(), 0.85) if d["ACCEL"].notna().any() else 0.3)
        thr_j = thr.get("JERK_HIGH_Q90",
                       np.nanquantile(d["JERK"].abs().dropna(), 0.90) if d["JERK"].notna().any() else 0.5)
        return (d["ACCEL"].abs() > thr_a) | (d["JERK"].abs() > thr_j)
    
    def _agg_for_ml_drive(self, g, thr):
        """Aggregate drive-level features for ML model"""
        g = self._add_basic_derivatives(g.copy())
        base = self._infer_base_interval_seconds(g["timestamp"], 1.0)
        g["IDLE_RULE"] = self._idle_rule(g, thr)
        
        dt = g["timestamp"].diff().dt.total_seconds().fillna(0).clip(lower=0, upper=10*base)
        T = float(dt.sum())
        mins = max(1e-6, T/60)
        
        sharp = self._sharp_mask_from_thresholds(g, thr).values
        edges = np.flatnonzero(np.diff(np.r_[False, sharp, False]))
        sharp_freq_pm = (len(edges)//2)/mins
        
        def q(s, p): 
            s = pd.to_numeric(s, errors="coerce")
            return float(np.nanquantile(s, p)) if s.notna().any() else 0.0
        
        rpm90, maf90 = thr.get("RPM90", np.nan), thr.get("MAF90", np.nan)
        frac_rpm90 = float((g["RPM"] >= rpm90).mean()) if ("RPM" in g and np.isfinite(rpm90)) else 0.0
        frac_maf90 = float((g["MAF"] >= maf90).mean()) if ("MAF" in g and np.isfinite(maf90)) else 0.0
        
        W10 = self._rows_for(10, base)
        speed_cv = float((g["SPEED_ms"].rolling(W10,1).std()/(g["SPEED_ms"].rolling(W10,1).mean()+1e-6)).mean())
        
        return {
            "duration_min": max(1e-6, T/60),
            "distance_km": g["dist_m"].sum()/1000.0,
            "speed_mean": float(g["SPEED_ms"].mean()),
            "speed_q90": q(g["SPEED_ms"], 0.90),
            "speed_cv": speed_cv,
            "accel_q90": q(g["ACCEL"].abs(), 0.90),
            "jerk_q90": q(g["JERK"].abs(), 0.90),
            "sharp_freq_pm": sharp_freq_pm,
            "idle_frac": float(g["IDLE_RULE"].mean()),
            "idle_epm": (len(np.flatnonzero(np.diff(np.r_[False, g['IDLE_RULE'].values, False])))//2)/mins,
            "rpm_q90": q(g["RPM"], 0.90) if "RPM" in g else 0.0,
            "maf_q90": q(g["MAF"], 0.90) if "MAF" in g else 0.0,
            "load_q85": q(g["ENGINE_LOAD"], 0.85) if "ENGINE_LOAD" in g else 0.0,
            "thr_q85": q(g["THROTTLE_POS"], 0.85) if "THROTTLE_POS" in g else 0.0,
            "frac_rpm90": frac_rpm90,
            "frac_maf90": frac_maf90,
            "fuel_intensity": (q(g["RPM"], 0.90)*q(g["MAF"], 0.90)) if (("RPM" in g) and ("MAF" in g)) else 0.0
        }
    
    def _align_to_schema(self, feats, art):
        """Align features to model schema"""
        x = pd.DataFrame([feats])
        for c in art["feature_names"]:
            if c not in x.columns: 
                x[c] = 0.0
        x = x[art["feature_names"]]
        if len(art["num_cols"]): 
            x.loc[:, art["num_cols"]] = art["scaler"].transform(x[art["num_cols"]])
        return x
    
    def _predict_drive(self, df_drive):
        """Predict efficiency for a single drive"""
        art = EfficiencyLabeler._model_artifacts
        thr = art["thr"]
        
        feats = self._agg_for_ml_drive(df_drive, thr)
        x = self._align_to_schema(feats, art)
        
        # Get model
        mdl = art["rf"] if art.get("model_kind") == "rf" else art["gbm"]
        raw = float(mdl.predict(x)[0])
        
        # Apply quantile-mapping calibration
        if art.get("calib", {}).get("type") == "qmap":
            rq = np.array(art["calib"]["rq"])
            yq = np.array(art["calib"]["yq"])
            
            # Ensure strictly increasing rq for stable interpolation
            for i in range(1, len(rq)):
                if rq[i] <= rq[i-1]: 
                    rq[i] = rq[i-1] + 1e-6
            
            pred = float(np.clip(np.interp(raw, rq, yq), 0, 100))
        else:
            pred = float(np.clip(raw, 0, 100))
        
        return pred, raw
    
    def predict_df(self, df: pd.DataFrame) -> List[float]:
        """
        Predict fuel efficiency for a DataFrame containing OBD data.
        Returns a single efficiency score (0-100%) for the entire drive.
        
        Args:
            df: DataFrame with OBD data including timestamp, SPEED, RPM, MAF, etc.
            
        Returns:
            List containing single efficiency score for the drive
        """
        try:
            if EfficiencyLabeler._model_artifacts is None:
                raise RuntimeError("Efficiency model not loaded")
            
            if len(df) < 5:
                logger.warning("⚠️ Drive too short for efficiency prediction")
                return [0.0]  # Return minimum efficiency for very short drives
            
            # Ensure timestamp column exists
            if "timestamp" not in df.columns:
                logger.error("❌ No timestamp column found")
                return [0.0]
            
            # Predict efficiency for the entire drive
            efficiency_score, raw_score = self._predict_drive(df)
            
            logger.info(f"📊 Drive efficiency: {efficiency_score:.1f}% (raw: {raw_score:.3f})")
            return [efficiency_score]
            
        except Exception as e:
            logger.error(f"❌ Error predicting efficiency: {e}")
            return [0.0]  # Return minimum efficiency on error
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if EfficiencyLabeler._model_artifacts is None:
            return {"error": "Model not loaded"}
        
        art = EfficiencyLabeler._model_artifacts
        return {
            "model_kind": art.get("model_kind", "unknown"),
            "feature_count": len(art.get("feature_names", [])),
            "features": art.get("feature_names", []),
            "calibration_type": art.get("calib", {}).get("type", "none"),
            "oof_stats": art.get("oof_stats", {}),
            "metadata": EfficiencyLabeler._metadata
        }

# Convenience function for backward compatibility
def predict_efficiency(df: pd.DataFrame) -> List[float]:
    """Convenience function to predict efficiency"""
    labeler = EfficiencyLabeler.get()
    return labeler.predict_df(df)

if __name__ == "__main__":
    # Test the efficiency labeler
    try:
        labeler = EfficiencyLabeler.get()
        print("✅ Efficiency labeler initialized successfully")
        
        # Print model info
        info = labeler.get_model_info()
        print(f"📊 Model info: {info}")
        
    except Exception as e:
        print(f"❌ Error initializing efficiency labeler: {e}")
