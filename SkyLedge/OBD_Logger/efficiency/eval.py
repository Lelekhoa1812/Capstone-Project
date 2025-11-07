"""
Fuel Efficiency Model Evaluation Script
Integration-ready evaluation script for fuel efficiency scoring in the main pipeline
Based on the original eval.py but reformatted for system integration
"""

import os
import glob
import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger("efficiency-eval")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s"))
    logger.addHandler(handler)

# Constants
KMH_TO_MS = 1000.0/3600.0

class EfficiencyEvaluator:
    """
    Fuel efficiency evaluator for OBD data using trained model.
    Provides drive-level efficiency scoring for integration into main pipeline.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the trained model. If None, will try to load from default location.
        """
        self.model_path = model_path or self._find_model_path()
        self.model_artifacts = None
        self.metadata = None
        self._load_model()
    
    def _find_model_path(self) -> str:
        """Find the model path from various possible locations"""
        possible_paths = [
            "./efficiency_export/efficiency_model.joblib",
            "/app/models/efficiency/efficiency_model.joblib",
            "./efficiency_model.joblib"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"📁 Found model at: {path}")
                return path
        
        # Try to download from Hugging Face
        logger.warning("⚠️ Model not found locally, attempting download...")
        try:
            from utils.efficiency_download import download_latest_efficiency_models
            success = download_latest_efficiency_models()
            if success:
                return "/app/models/efficiency/efficiency_model.joblib"
        except Exception as e:
            logger.error(f"❌ Failed to download model: {e}")
        
        raise FileNotFoundError("Could not find or download efficiency model")
    
    def _load_model(self):
        """Load the efficiency model and metadata"""
        try:
            logger.info(f"📥 Loading efficiency model from: {self.model_path}")
            
            # Load model artifacts
            self.model_artifacts = joblib.load(self.model_path)
            
            # Load metadata if available
            meta_path = self.model_path.replace("efficiency_model.joblib", "efficiency_meta.json")
            if os.path.exists(meta_path):
                import json
                with open(meta_path, 'r') as f:
                    self.metadata = json.load(f)
            
            logger.info(f"✅ Model loaded | kind: {self.model_artifacts.get('model_kind', 'unknown')}")
            logger.info(f"📊 Features: {len(self.model_artifacts.get('feature_names', []))}")
            
            if self.metadata:
                logger.info(f"📅 Training date: {self.metadata.get('training_date', 'unknown')}")
                logger.info(f"📈 OOF MAE: {self.metadata.get('oof_stats', {}).get('oof_mae_qmap', 'unknown')}")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
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
    
    def _q(self, s, p):
        """Quantile helper function"""
        s = pd.to_numeric(s, errors="coerce")
        return float(np.nanquantile(s, p)) if s.notna().any() else 0.0
    
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
        
        rpm90, maf90 = thr.get("RPM90", np.nan), thr.get("MAF90", np.nan)
        frac_rpm90 = float((g["RPM"] >= rpm90).mean()) if ("RPM" in g and np.isfinite(rpm90)) else 0.0
        frac_maf90 = float((g["MAF"] >= maf90).mean()) if ("MAF" in g and np.isfinite(maf90)) else 0.0
        
        W10 = self._rows_for(10, base)
        speed_cv = float((g["SPEED_ms"].rolling(W10,1).std()/(g["SPEED_ms"].rolling(W10,1).mean()+1e-6)).mean())
        
        return {
            "duration_min": max(1e-6, T/60),
            "distance_km": g["dist_m"].sum()/1000.0,
            "speed_mean": float(g["SPEED_ms"].mean()),
            "speed_q90": self._q(g["SPEED_ms"], 0.90),
            "speed_cv": speed_cv,
            "accel_q90": self._q(g["ACCEL"].abs(), 0.90),
            "jerk_q90": self._q(g["JERK"].abs(), 0.90),
            "sharp_freq_pm": sharp_freq_pm,
            "idle_frac": float(g["IDLE_RULE"].mean()),
            "idle_epm": (len(np.flatnonzero(np.diff(np.r_[False, g['IDLE_RULE'].values, False])))//2)/mins,
            "rpm_q90": self._q(g["RPM"], 0.90) if "RPM" in g else 0.0,
            "maf_q90": self._q(g["MAF"], 0.90) if "MAF" in g else 0.0,
            "load_q85": self._q(g["ENGINE_LOAD"], 0.85) if "ENGINE_LOAD" in g else 0.0,
            "thr_q85": self._q(g["THROTTLE_POS"], 0.85) if "THROTTLE_POS" in g else 0.0,
            "frac_rpm90": frac_rpm90,
            "frac_maf90": frac_maf90,
            "fuel_intensity": (self._q(g["RPM"], 0.90)*self._q(g["MAF"], 0.90)) if (("RPM" in g) and ("MAF" in g)) else 0.0
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
        art = self.model_artifacts
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
        
        return pred, raw, feats
    
    def predict_single_drive(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict fuel efficiency for a single drive.
        
        Args:
            df: DataFrame with OBD data including timestamp, SPEED, RPM, MAF, etc.
            
        Returns:
            Dictionary containing efficiency prediction and metadata
        """
        try:
            if self.model_artifacts is None:
                raise RuntimeError("Efficiency model not loaded")
            
            if len(df) < 5:
                logger.warning("⚠️ Drive too short for efficiency prediction")
                return {
                    "efficiency_score": 0.0,
                    "raw_score": 0.0,
                    "duration_min": 0.0,
                    "distance_km": 0.0,
                    "note": "too short",
                    "features": {}
                }
            
            # Calculate basic drive metrics
            g2 = self._add_basic_derivatives(df[["timestamp","SPEED"]].assign(
                RPM=df.get("RPM"), MAF=df.get("MAF"), 
                ENGINE_LOAD=df.get("ENGINE_LOAD"), THROTTLE_POS=df.get("THROTTLE_POS")))
            
            dt = g2["timestamp"].diff().dt.total_seconds().fillna(0)
            mins = float(dt.sum())/60.0
            dist_km = float(pd.to_numeric(g2["dist_m"], errors="coerce").fillna(0).sum())/1000.0
            
            # Predict efficiency
            efficiency_score, raw_score, features = self._predict_drive(df)
            
            logger.info(f"📊 Drive efficiency: {efficiency_score:.1f}% (raw: {raw_score:.3f})")
            
            return {
                "efficiency_score": round(efficiency_score, 1),
                "raw_score": round(raw_score, 3),
                "duration_min": round(mins, 2),
                "distance_km": round(dist_km, 3),
                "features": features,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error predicting efficiency: {e}")
            return {
                "efficiency_score": 0.0,
                "raw_score": 0.0,
                "duration_min": 0.0,
                "distance_km": 0.0,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def predict_batch(self, csv_files: List[str]) -> pd.DataFrame:
        """
        Predict efficiency for multiple CSV files (batch processing).
        
        Args:
            csv_files: List of CSV file paths
            
        Returns:
            DataFrame with predictions for each file
        """
        logger.info(f"📊 Processing {len(csv_files)} CSV files...")
        
        rows = []
        for i, csv_path in enumerate(csv_files, start=1):
            try:
                # Load CSV
                df = pd.read_csv(csv_path)
                df["source_file"] = os.path.basename(csv_path)
                df["drive_id"] = i
                df["timestamp"] = self._ensure_dt(df["timestamp"])
                df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
                
                if len(df) < 5:
                    rows.append({
                        "source_file": os.path.basename(csv_path),
                        "drive_id": i,
                        "duration_min": np.nan,
                        "distance_km": np.nan,
                        "pred_efficiency_ml": np.nan,
                        "raw": np.nan,
                        "note": "too short"
                    })
                    continue
                
                # Predict efficiency
                result = self.predict_single_drive(df)
                
                rows.append({
                    "source_file": os.path.basename(csv_path),
                    "drive_id": i,
                    "duration_min": result["duration_min"],
                    "distance_km": result["distance_km"],
                    "pred_efficiency_ml": result["efficiency_score"],
                    "raw": result["raw_score"]
                })
                
            except Exception as e:
                logger.error(f"❌ Error processing {csv_path}: {e}")
                rows.append({
                    "source_file": os.path.basename(csv_path),
                    "drive_id": i,
                    "duration_min": np.nan,
                    "distance_km": np.nan,
                    "pred_efficiency_ml": np.nan,
                    "raw": np.nan,
                    "error": str(e)
                })
        
        pred_df = pd.DataFrame(rows).sort_values("drive_id").reset_index(drop=True)
        
        # Calculate statistics
        valid_preds = pred_df["pred_efficiency_ml"].dropna()
        if len(valid_preds) > 0:
            logger.info(f"📊 Batch results: {len(valid_preds)} valid predictions")
            logger.info(f"📈 Efficiency range: {valid_preds.min():.1f}% - {valid_preds.max():.1f}%")
            logger.info(f"📊 Mean efficiency: {valid_preds.mean():.1f}%")
            logger.info(f"📊 Std efficiency: {valid_preds.std():.1f}%")
        
        return pred_df
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self.model_artifacts is None:
            return {"error": "Model not loaded"}
        
        return {
            "model_kind": self.model_artifacts.get("model_kind", "unknown"),
            "feature_count": len(self.model_artifacts.get("feature_names", [])),
            "features": self.model_artifacts.get("feature_names", []),
            "calibration_type": self.model_artifacts.get("calib", {}).get("type", "none"),
            "oof_stats": self.model_artifacts.get("oof_stats", {}),
            "metadata": self.metadata,
            "model_path": self.model_path
        }

def evaluate_csv_files(csv_directory: str = "./") -> pd.DataFrame:
    """
    Convenience function to evaluate all CSV files in a directory.
    
    Args:
        csv_directory: Directory containing CSV files
        
    Returns:
        DataFrame with efficiency predictions
    """
    # Find CSV files
    csv_patterns = [
        os.path.join(csv_directory, "*.csv"),
        os.path.join("/content", "*.csv")  # For Colab compatibility
    ]
    
    csv_files = []
    for pattern in csv_patterns:
        csv_files.extend(glob.glob(pattern))
    
    csv_files = sorted([p for p in csv_files if os.path.isfile(p)])
    
    if not csv_files:
        logger.warning("⚠️ No CSV files found")
        return pd.DataFrame()
    
    # Initialize evaluator and process files
    evaluator = EfficiencyEvaluator()
    return evaluator.predict_batch(csv_files)

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate fuel efficiency model")
    parser.add_argument("--csv-dir", default="./", help="Directory containing CSV files")
    parser.add_argument("--model-path", help="Path to efficiency model file")
    parser.add_argument("--output", help="Output CSV file path")
    
    args = parser.parse_args()
    
    try:
        # Initialize evaluator
        evaluator = EfficiencyEvaluator(model_path=args.model_path)
        
        # Print model info
        info = evaluator.get_model_info()
        print(f"📊 Model info: {info}")
        
        # Evaluate CSV files
        results_df = evaluate_csv_files(args.csv_dir)
        
        if len(results_df) > 0:
            print("\n=== Batch Efficiency Scores (per CSV / drive) ===")
            print(results_df.to_string(index=False))
            
            # Save results if output path specified
            if args.output:
                results_df.to_csv(args.output, index=False)
                print(f"\n💾 Results saved to: {args.output}")
        else:
            print("❌ No valid CSV files found for evaluation")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())