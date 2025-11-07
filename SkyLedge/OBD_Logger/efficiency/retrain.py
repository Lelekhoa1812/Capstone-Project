"""
Fuel Efficiency Model Retraining Script
Reproducible training script for fuel efficiency model with Hugging Face integration
Based on the original retrain.py but reformatted for system integration
"""

import os
import glob
import json
import math
import joblib
import warnings
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# ML imports
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge

# Hugging Face integration
from huggingface_hub import HfApi, Repository

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Setup logging
logger = logging.getLogger("efficiency-retrain")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s"))
    logger.addHandler(handler)

# Constants
SEED = 42
KMH_TO_MS = 1000.0/3600.0
np.random.seed(SEED)

class EfficiencyModelTrainer:
    """
    Fuel efficiency model trainer with Hugging Face integration.
    Handles data loading, feature engineering, model training, and model upload.
    """
    
    def __init__(self, 
                 csv_directory: str = "./",
                 export_directory: str = "./efficiency_export",
                 repo_id: str = "BinKhoaLe1812/Fuel_Efficiency_OBD"):
        """
        Initialize the trainer.
        
        Args:
            csv_directory: Directory containing CSV files for training
            export_directory: Directory to save trained model artifacts
            repo_id: Hugging Face repository ID for model upload
        """
        self.csv_directory = csv_directory
        self.export_directory = Path(export_directory)
        self.repo_id = repo_id
        self.hf_token = os.getenv("HF_TOKEN")
        
        # Create export directory
        self.export_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize HF API if token available
        self.hf_api = None
        if self.hf_token:
            self.hf_api = HfApi(token=self.hf_token)
            logger.info(f"✅ Hugging Face API initialized for {repo_id}")
        else:
            logger.warning("⚠️ HF_TOKEN not set - model will not be uploaded to Hugging Face")
    
    def load_training_data(self) -> pd.DataFrame:
        """Load and preprocess training data from CSV files"""
        logger.info("📊 Loading training data...")
        
        # Find CSV files
        csv_patterns = [
            os.path.join(self.csv_directory, "*.csv"),
            os.path.join("/content", "*.csv")  # For Colab compatibility
        ]
        
        csvs = []
        for pattern in csv_patterns:
            csvs.extend(glob.glob(pattern))
        
        csvs = sorted([p for p in csvs if os.path.isfile(p)])
        
        if not csvs:
            raise RuntimeError("No CSV logs found for training")
        
        logger.info(f"📁 Found {len(csvs)} CSV files")
        
        # Load and combine CSV files
        frames = []
        for i, p in enumerate(csvs, start=1):
            try:
                d = pd.read_csv(p)
                d["source_file"] = os.path.basename(p)
                d["drive_id"] = i
                frames.append(d)
                logger.info(f"✅ Loaded {os.path.basename(p)} ({len(d)} rows)")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {p}: {e}")
        
        if not frames:
            raise RuntimeError("No valid CSV files could be loaded")
        
        # Combine all data
        df = pd.concat(frames, ignore_index=True)
        df["timestamp"] = self._ensure_dt(df["timestamp"])
        df = df.dropna(subset=["timestamp"]).sort_values(["drive_id", "timestamp"]).reset_index(drop=True)
        df = self._add_basic_derivatives(df)
        
        logger.info(f"📊 Combined dataset: {len(df)} rows, {df['drive_id'].nunique()} drives")
        return df
    
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
    
    def _run_lengths(self, mask):
        """Calculate run lengths from boolean mask"""
        m = np.asarray(mask, dtype=bool)
        if m.size == 0: 
            return np.array([], int), np.array([], int)
        dm = np.diff(np.r_[False, m, False].astype(int))
        starts = np.where(dm == 1)[0]
        ends = np.where(dm == -1)[0]
        return starts, (ends - starts)
    
    def _penalty(self, series):
        """Calculate penalty function for efficiency scoring"""
        arr = pd.to_numeric(series, errors="coerce").fillna(0).values
        if arr.size == 0: 
            return pd.Series([], dtype=float, index=series.index)
        q25, q50, q75 = np.quantile(arr, [0.25, 0.50, 0.75])
        s = (q75-q25)/1.349 if (q75 > q25) else (np.std(arr) if np.std(arr) > 0 else 1.0)
        return pd.Series(1/(1+np.exp(-(arr - q50)/max(1e-6, s))), index=series.index)
    
    def compute_fleet_thresholds(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute fleet-wide thresholds for feature engineering"""
        logger.info("🔧 Computing fleet thresholds...")
        
        thr = {}
        
        # RPM threshold
        if "RPM" in df and df["RPM"].notna().any():
            thr["RPM90"] = float(np.nanquantile(df["RPM"], 0.90))
        
        # MAF threshold
        if "MAF" in df and df["MAF"].notna().any():
            thr["MAF90"] = float(np.nanquantile(df["MAF"], 0.90))
        
        # Throttle position thresholds
        if "THROTTLE_POS" in df and df["THROTTLE_POS"].notna().any():
            thr["THR_LOW_Q10"] = float(np.nanquantile(df["THROTTLE_POS"], 0.10))
            thr["THR_Q85"] = float(np.nanquantile(df["THROTTLE_POS"], 0.85))
        
        # Engine load thresholds
        if "ENGINE_LOAD" in df and df["ENGINE_LOAD"].notna().any():
            thr["LOAD_LOW_Q15"] = float(np.nanquantile(df["ENGINE_LOAD"], 0.15))
            thr["LOAD_Q85"] = float(np.nanquantile(df["ENGINE_LOAD"], 0.85))
        
        # Acceleration and jerk thresholds
        tmpd = self._add_basic_derivatives(df[["timestamp","SPEED"]].assign(
            RPM=df.get("RPM"), MAF=df.get("MAF"), 
            THROTTLE_POS=df.get("THROTTLE_POS"), ENGINE_LOAD=df.get("ENGINE_LOAD")))
        
        thr["ACCEL_LOW_Q20"] = float(np.nanquantile(tmpd["ACCEL"].abs().dropna(), 0.20)) if tmpd["ACCEL"].notna().any() else 0.05
        thr["ACCEL_HIGH_Q85"] = float(np.nanquantile(tmpd["ACCEL"].abs().dropna(), 0.85)) if tmpd["ACCEL"].notna().any() else 0.3
        thr["JERK_HIGH_Q90"] = float(np.nanquantile(tmpd["JERK"].abs().dropna(), 0.90)) if tmpd["JERK"].notna().any() else 0.5
        thr["SPEED_IDLE_MPS"] = 0.6
        
        logger.info(f"✅ Computed {len(thr)} fleet thresholds")
        return thr
    
    def create_algorithmic_teacher(self, df: pd.DataFrame, thr: Dict[str, float]) -> pd.DataFrame:
        """Create algorithmic teacher labels for training"""
        logger.info("🎯 Creating algorithmic teacher labels...")
        
        # Apply idle rule to all drives
        df["IDLE_RULE"] = False
        for gid, g in df.groupby("drive_id", sort=True):
            df.loc[g.index, "IDLE_RULE"] = self._idle_rule(g, thr)
        
        # Extract thresholds
        thr_accel, thr_jerk = thr["ACCEL_HIGH_Q85"], thr["JERK_HIGH_Q90"]
        thr_rpm90, thr_maf90 = thr.get("RPM90", np.nan), thr.get("MAF90", np.nan)
        
        # Process each drive
        drv = []
        for gid, g in df.groupby("drive_id", sort=True):
            if len(g) < 5: 
                continue
            
            base = self._infer_base_interval_seconds(g["timestamp"], 1.0)
            dt_s = g["timestamp"].diff().dt.total_seconds().fillna(0).clip(lower=0, upper=10*base)
            T = float(dt_s.sum())
            mins = max(1e-6, T/60)
            
            # Sharp acceleration analysis
            sharp = self._sharp_mask_from_thresholds(g, thr).values
            st, ln = self._run_lengths(sharp)
            freq_pm = len(ln)/mins
            dur_frac = (ln.sum()*base)/max(1e-6, T)
            
            # Peak analysis
            peaks = []
            for a, b in zip(st, ln):
                seg = g.iloc[a:a+b]
                pa = float(np.nanmax(np.abs(seg["ACCEL"])))
                pj = float(np.nanmax(np.abs(seg["JERK"])))
                over_a = max(0.0, (pa-thr_accel)/max(1e-6, thr_accel))
                over_j = max(0.0, (pj-thr_jerk)/max(1e-6, thr_jerk))
                peaks.append(min(1.5, 0.7*over_a + 0.3*over_j))
            
            sharp_mag = float(np.mean(peaks)) if peaks else 0.0
            
            # Idle analysis
            idle_frac = float(g["IDLE_RULE"].mean())
            sti, lni = self._run_lengths(g["IDLE_RULE"].values)
            idle_med_s = float(np.median(lni)*base if len(lni) else 0.0)
            idle_epm = len(lni)/mins
            
            # Speed variability
            W10 = self._rows_for(10, base)
            speed_cv = float((g["SPEED_ms"].rolling(W10,1).std()/(g["SPEED_ms"].rolling(W10,1).mean()+1e-6)).mean())
            
            # High-load fractions
            frac_rpm90 = float((g["RPM"] >= thr_rpm90).mean()) if ("RPM" in g and np.isfinite(thr_rpm90)) else 0.0
            frac_maf90 = float((g["MAF"] >= thr_maf90).mean()) if ("MAF" in g and np.isfinite(thr_maf90)) else 0.0
            frac_load85 = float((g["ENGINE_LOAD"] >= thr.get("LOAD_Q85", np.inf)).mean()) if "ENGINE_LOAD" in g else 0.0
            frac_thr85 = float((g["THROTTLE_POS"] >= thr.get("THR_Q85", np.inf)).mean()) if "THROTTLE_POS" in g else 0.0
            
            # Efficiency proxy
            proxy = (0.80*frac_rpm90 + 0.60*frac_maf90 + 0.15*frac_load85 + 0.10*frac_thr85 + 0.10*idle_frac)
            
            drv.append(dict(
                drive_id=gid, duration_min=mins, distance_km=g["dist_m"].sum()/1000.0,
                freq_pm=freq_pm, dur_frac=dur_frac, sharp_mag=sharp_mag,
                idle_frac=idle_frac, idle_med_s=idle_med_s, idle_epm=idle_epm,
                speed_cv=speed_cv, frac_rpm90=frac_rpm90, frac_maf90=frac_maf90, proxy=proxy
            ))
        
        dfeat = pd.DataFrame(drv).set_index("drive_id")
        
        # Calculate penalty-based features
        P = pd.DataFrame({
            "p_freq": self._penalty(dfeat["freq_pm"]),
            "p_dur": self._penalty(dfeat["dur_frac"]),
            "p_mag": self._penalty(dfeat["sharp_mag"]),
            "p_idle": 0.7*self._penalty(dfeat["idle_frac"]) + 0.3*self._penalty(dfeat["idle_med_s"]),
            "p_cv": self._penalty(dfeat["speed_cv"]),
            "p_rpm": self._penalty(dfeat["frac_rpm90"]),
            "p_maf": self._penalty(dfeat["frac_maf90"]),
        }, index=dfeat.index)
        
        # Calculate efficiency scores
        proxy = dfeat["proxy"].clip(0, 1-1e-6)
        target_lin = -np.log(1 - proxy)
        w = np.linalg.lstsq(P.values, target_lin.values, rcond=None)[0]
        dfeat["ineff_model"] = 1 - np.exp(-P.values @ w)
        dfeat["efficiency_algo"] = 100*(1 - dfeat["ineff_model"])
        
        logger.info(f"✅ Teacher range: {dfeat['efficiency_algo'].min():.1f} → {dfeat['efficiency_algo'].max():.1f}")
        return dfeat
    
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
    
    def prepare_ml_data(self, df: pd.DataFrame, dfeat: pd.DataFrame, thr: Dict[str, float]) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Prepare data for machine learning training"""
        logger.info("🔧 Preparing ML training data...")
        
        rows, y, groups = [], [], []
        for gid, g in df.groupby("drive_id", sort=True):
            if len(g) < 5: 
                continue
            rows.append(self._agg_for_ml_drive(g, thr))
            y.append(float(dfeat.loc[gid, "efficiency_algo"]))
            groups.append(g["source_file"].iloc[0] if "source_file" in g.columns else gid)
        
        X = pd.DataFrame(rows)
        y = np.asarray(y, float)
        groups = np.asarray(groups)
        
        # Remove zero-variance features
        zv = X.std(numeric_only=True).fillna(0.0)
        drop_cols = list(zv[zv <= 1e-10].index)
        if drop_cols:
            X = X.drop(columns=drop_cols)
            logger.info(f"🗑️ Dropped zero-variance features: {drop_cols}")
        
        # Scale features
        holdout_cols = ["duration_min", "distance_km"]
        num_cols = [c for c in X.columns if c not in holdout_cols]
        sc = StandardScaler().fit(X[num_cols])
        X[num_cols] = sc.transform(X[num_cols])
        
        logger.info(f"✅ Prepared ML data: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y, groups, sc, num_cols, holdout_cols
    
    def train_model(self, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> Tuple[Any, str, Dict[str, Any]]:
        """Train the efficiency model with cross-validation"""
        logger.info("🤖 Training efficiency model...")
        
        # Out-of-fold predictions for calibration
        gkf = GroupKFold(n_splits=min(5, max(2, len(np.unique(groups)))))
        oof_raw = np.zeros_like(y)
        
        for tr, va in gkf.split(X, y, groups):
            gbm_fold = HistGradientBoostingRegressor(
                loss="squared_error", max_depth=6, learning_rate=0.08, max_bins=255,
                early_stopping=True, random_state=SEED
            )
            wtr = np.clip(X.iloc[tr]["duration_min"].values, 0.5, None)
            gbm_fold.fit(X.iloc[tr], y[tr], sample_weight=wtr)
            pred = gbm_fold.predict(X.iloc[va])
            
            if np.std(pred) < 1e-6:
                # Ridge rescue to enforce variability
                ridge = Ridge(alpha=1.0, random_state=SEED).fit(X.iloc[tr][X.columns[2:]], y[tr])
                pred = ridge.predict(X.iloc[va][X.columns[2:]])
            
            oof_raw[va] = pred
        
        # Calculate OOF statistics
        raw_std = float(np.std(oof_raw))
        y_std = float(np.std(y))
        corr = float(np.corrcoef(oof_raw, y)[0,1]) if len(y) > 1 else 1.0
        
        logger.info(f"📊 OOF: corr={corr:.3f} | raw_std={raw_std:.3f} | y_std={y_std:.3f}")
        
        # Quantile-mapping calibration
        qs = np.linspace(0.05, 0.95, 19)
        rq = np.quantile(oof_raw, qs)
        yq = np.quantile(y, qs)
        
        # Ensure strictly increasing rq for stable interpolation
        for i in range(1, len(rq)):
            if rq[i] <= rq[i-1]: 
                rq[i] = rq[i-1] + 1e-6
        
        calib = {"type": "qmap", "rq": rq.tolist(), "yq": yq.tolist()}
        
        def apply_calib_qmap(raw):
            return float(np.clip(np.interp(raw, rq, yq), 0, 100))
        
        oof_cal = np.array([apply_calib_qmap(r) for r in oof_raw], float)
        oof_mae = float(mean_absolute_error(y, oof_cal))
        
        logger.info(f"📊 OOF MAE (qmap): {oof_mae:.2f}")
        
        # Final model training
        gbm = HistGradientBoostingRegressor(
            loss="squared_error", max_depth=6, learning_rate=0.08, max_bins=255,
            early_stopping=False, max_iter=400, random_state=SEED
        )
        w_all = np.clip(X["duration_min"].values, 0.5, None)
        gbm.fit(X, y, sample_weight=w_all)
        raw_all = gbm.predict(X)
        
        if np.std(raw_all) < 1e-6:
            logger.warning("⚠️ Final GBM raw constant — switching to RandomForest")
            rf = RandomForestRegressor(n_estimators=600, min_samples_leaf=2, random_state=SEED, n_jobs=-1)
            rf.fit(X, y)
            model_kind, model = "rf", rf
        else:
            model_kind, model = "gbm", gbm
        
        oof_stats = {
            "oof_mae_qmap": oof_mae,
            "oof_corr": corr,
            "raw_std": raw_std,
            "y_std": y_std
        }
        
        logger.info(f"✅ Model training complete | kind: {model_kind}")
        return model, model_kind, calib, oof_stats
    
    def save_model(self, model, model_kind: str, scaler, feature_names: List[str], 
                   num_cols: List[str], holdout_cols: List[str], thr: Dict[str, float],
                   calib: Dict[str, Any], oof_stats: Dict[str, Any]) -> str:
        """Save the trained model and artifacts"""
        logger.info("💾 Saving model artifacts...")
        
        # Prepare artifacts
        artifacts = {
            "scaler": scaler,
            "model_kind": model_kind,
            "gbm": model if model_kind == "gbm" else None,
            "rf": model if model_kind == "rf" else None,
            "feature_names": feature_names,
            "num_cols": num_cols,
            "holdout_cols": holdout_cols,
            "windowing": {"size_s": 120, "step_s": 60},  # For future use
            "thr": thr,
            "seed": SEED,
            "calib": calib,
            "oof_stats": oof_stats,
            "training_timestamp": datetime.now().isoformat(),
            "version": "1.0"  # Will be updated based on HF versioning
        }
        
        # Save model
        model_path = self.export_directory / "efficiency_model.joblib"
        joblib.dump(artifacts, model_path)
        
        # Save metadata
        metadata = {
            "model_type": "fuel_efficiency",
            "version": "1.0",
            "training_date": datetime.now().isoformat(),
            "model_kind": model_kind,
            "feature_count": len(feature_names),
            "oof_stats": oof_stats,
            "calibration_type": calib.get("type", "none")
        }
        
        meta_path = self.export_directory / "efficiency_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Model saved to {model_path}")
        logger.info(f"✅ Metadata saved to {meta_path}")
        
        return str(model_path)
    
    def upload_to_huggingface(self, version: str = None) -> bool:
        """Upload the trained model to Hugging Face Hub"""
        if not self.hf_api:
            logger.warning("⚠️ Hugging Face API not available - skipping upload")
            return False
        
        try:
            if version is None:
                version = self._get_next_version()
            
            logger.info(f"📤 Uploading model version {version} to Hugging Face...")
            
            # Upload model file
            model_path = self.export_directory / "efficiency_model.joblib"
            meta_path = self.export_directory / "efficiency_meta.json"
            
            if not model_path.exists():
                logger.error(f"❌ Model file not found: {model_path}")
                return False
            
            # Upload files
            self.hf_api.upload_file(
                path_or_fileobj=str(model_path),
                path_in_repo=f"{version}/efficiency_model.joblib",
                repo_id=self.repo_id,
                repo_type="model"
            )
            
            if meta_path.exists():
                self.hf_api.upload_file(
                    path_or_fileobj=str(meta_path),
                    path_in_repo=f"{version}/efficiency_meta.json",
                    repo_id=self.repo_id,
                    repo_type="model"
                )
            
            logger.info(f"✅ Model {version} uploaded successfully to {self.repo_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error uploading to Hugging Face: {e}")
            return False
    
    def _get_next_version(self) -> str:
        """Get the next version number (1.0, 1.1, 1.2, ..., 1.9, 2.0, etc.)"""
        try:
            repo_files = self.hf_api.list_repo_files(
                repo_id=self.repo_id,
                repo_type="model"
            )
            
            # Find existing versions
            versions = []
            for f in repo_files:
                if f.startswith('v') and '/' not in f:
                    try:
                        version_str = f[1:]  # Remove 'v' prefix
                        major, minor = map(int, version_str.split('.'))
                        versions.append((major, minor))
                    except ValueError:
                        continue
            
            if not versions:
                return "v1.0"
            
            # Sort and get next version
            versions.sort(key=lambda x: (x[0], x[1]))
            latest_major, latest_minor = versions[-1]
            
            if latest_minor < 9:
                return f"v{latest_major}.{latest_minor + 1}"
            else:
                return f"v{latest_major + 1}.0"
                
        except Exception as e:
            logger.warning(f"⚠️ Could not determine next version: {e}")
            return "v1.0"
    
    def train_and_upload(self, upload_to_hf: bool = True) -> Dict[str, Any]:
        """Complete training pipeline"""
        try:
            logger.info("🚀 Starting fuel efficiency model training pipeline...")
            
            # Load data
            df = self.load_training_data()
            
            # Compute thresholds
            thr = self.compute_fleet_thresholds(df)
            
            # Create teacher labels
            dfeat = self.create_algorithmic_teacher(df, thr)
            
            # Prepare ML data
            X, y, groups, scaler, num_cols, holdout_cols = self.prepare_ml_data(df, dfeat, thr)
            
            # Train model
            model, model_kind, calib, oof_stats = self.train_model(X, y, groups)
            
            # Save model
            model_path = self.save_model(
                model, model_kind, scaler, list(X.columns), 
                num_cols, holdout_cols, thr, calib, oof_stats
            )
            
            # Upload to Hugging Face
            upload_success = False
            if upload_to_hf:
                upload_success = self.upload_to_huggingface()
            
            result = {
                "success": True,
                "model_path": model_path,
                "model_kind": model_kind,
                "oof_stats": oof_stats,
                "upload_success": upload_success,
                "training_samples": len(X),
                "feature_count": len(X.columns)
            }
            
            logger.info("✅ Training pipeline completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ Training pipeline failed: {e}")
            return {"success": False, "error": str(e)}

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train fuel efficiency model")
    parser.add_argument("--csv-dir", default="./", help="Directory containing CSV files")
    parser.add_argument("--export-dir", default="./efficiency_export", help="Export directory")
    parser.add_argument("--repo-id", default="BinKhoaLe1812/Fuel_Efficiency_OBD", help="Hugging Face repo ID")
    parser.add_argument("--no-upload", action="store_true", help="Skip Hugging Face upload")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = EfficiencyModelTrainer(
        csv_directory=args.csv_dir,
        export_directory=args.export_dir,
        repo_id=args.repo_id
    )
    
    # Train and upload
    result = trainer.train_and_upload(upload_to_hf=not args.no_upload)
    
    if result["success"]:
        print("✅ Training completed successfully!")
        print(f"📊 Model: {result['model_kind']}")
        print(f"📈 OOF MAE: {result['oof_stats']['oof_mae_qmap']:.2f}")
        print(f"📤 Upload: {'✅' if result['upload_success'] else '❌'}")
    else:
        print(f"❌ Training failed: {result['error']}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())