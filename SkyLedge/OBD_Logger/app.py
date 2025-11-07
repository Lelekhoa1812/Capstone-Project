# Access: https://binkhoale1812-obd-logger.hf.space/ui


# ───────────── Installation ─────────────
# Router
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel
# ML/DL
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# Utils
from sklearn.impute import KNNImputer
import os, datetime, json, logging, re
from datetime import timedelta
import pathlib

# Drive
from data.drive_saver import DriveSaver, get_drive_service, upload_to_folder

# Database
from data.mongo_saver import MongoSaver, save_csv_to_mongo, save_dataframe_to_mongo, MONGODB_AVAILABLE
from data.firebase_saver import FirebaseSaver, save_csv_increment, save_dataframe_increment, save_efficiency_data, get_efficiency_by_filename as firebase_get_efficiency_by_filename

# UL Model
from utils.dbehavior_labeler import ULLabeler

# Fuel Efficiency Model
from utils.efficiency_labeler import EfficiencyLabeler

# RLHF Training
from train import RLHFTrainer

# ───────────── Logging Setup ─────────────
logger = logging.getLogger("obd-logger")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(fmt)
logger.addHandler(handler)


# ───────────── FastAPI Init ─────────────
app = FastAPI(title="OBD-II Logging & Processing API")


# ───────────── Directory Paths ─────────────
APP_ROOT = pathlib.Path(__file__).parent.resolve()  # Absolute base dir
BASE_DIR = os.path.join(APP_ROOT, './cache/obd_data')
CLEANED_DIR = os.path.join(BASE_DIR, "cleaned")
PLOT_DIR = os.path.join(BASE_DIR, "plots")
RAW_CSV = os.path.join(BASE_DIR, "raw_logs.csv")
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(CLEANED_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

DRIVE_STYLE = []  # latest UL predictions (string labels) — overwritten each run
FUEL_EFFICIENCY = []  # latest fuel efficiency predictions (0-100%) — overwritten each run

# Init temp empty file
if not os.path.exists(RAW_CSV):
    pd.DataFrame(columns=["timestamp", "driving_style"]).to_csv(RAW_CSV, index=False)

PIPELINE_EVENTS = {}


# ───────────── Drive & Database Services ─────────────
# Initialize services
drive_saver = DriveSaver()
mongo_saver = MongoSaver()
firebase_saver = FirebaseSaver()

# ───────────── Model Download on Startup ─────────────
@app.on_event("startup")
async def startup_event():
    """Download models on app startup"""
    try:
        logger.info("🚀 Starting model download...")
        from utils.dbehavior_download import download_latest_models
        
        # Load .env file if it exists
        env_path = pathlib.Path(".env")
        if env_path.exists():
            logger.info("📄 Loading .env file...")
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value
        
        # Download models
        success_ul = download_latest_models()
        if success_ul:
            logger.info("✅ Driver behavior models downloaded successfully on startup")
        else:
            logger.warning("⚠️ Driver behavior model download failed - some features may not work")
        
        # Download fuel efficiency models
        from utils.efficiency_download import download_latest_efficiency_models
        success_efficiency = download_latest_efficiency_models()
        if success_efficiency:
            logger.info("✅ Fuel efficiency models downloaded successfully")
        else:
            logger.warning("⚠️ Fuel efficiency model download failed - some features may not work")
            
        if success_ul or success_efficiency:
            logger.info("✅ At least one model type downloaded successfully")
        else:
            logger.warning("⚠️ All model downloads failed - some features may not work")
            
    except Exception as e:
        logger.error(f"❌ Startup model download failed: {e}")
        logger.warning("⚠️ Continuing without models - some features may not work")

# ───────────── Render Dashboard UI ──────────────
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/plots", StaticFiles(directory=str(PLOT_DIR)), name="plots")
templates = Jinja2Templates(directory="static")
# Endpoint
@app.get("/ui", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ───────────── Streamed Entry Ingest ─────────────
class OBDEntry(BaseModel):
    timestamp: str
    driving_style: str
    data: dict
    status: str = None  # Optional for control signal (start/end streaming)

# Direct centralized timestamp format
def normalize_timestamp(ts):
    return ts.replace(":", "-").replace(".", "-").replace(" ", "T").replace("/", "-")

# Real time endpoint
@app.post("/ingest")
def ingest(entry: OBDEntry, background_tasks: BackgroundTasks):
    norm_ts = normalize_timestamp(entry.timestamp)
    logger.info(f"Ingest received: {norm_ts} | Status: {entry.status}")
    # Start logging
    if entry.status == "start":
        PIPELINE_EVENTS[norm_ts] = {"status": "started", "time": norm_ts}
        return {"status": "started"}
    # End logging, start processing
    if entry.status == "end":
        background_tasks.add_task(process_data, norm_ts)
        return {"status": "processed"}
    # Moderate row append
    try:
        df = pd.read_csv(RAW_CSV)
        row = {"timestamp": norm_ts, "driving_style": entry.driving_style}
        row.update(entry.data)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(RAW_CSV, index=False)
        return {"status": "row appended"}
    except Exception as e:
        logger.error(f"Streaming ingest failed: {e}")
        raise HTTPException(status_code=500, detail="Ingest error")


# ───────────── Bulk CSV Upload ───────────────────
@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    ts = datetime.datetime.now().isoformat()
    norm_ts = normalize_timestamp(ts)
    path = os.path.join(BASE_DIR, file.filename)
    PIPELINE_EVENTS[norm_ts] = {"status": "started", "time": norm_ts}
    with open(path, "wb") as f:
        f.write(await file.read())
    logger.info(f"CSV uploaded: {path}")
    background_tasks.add_task(process_uploaded_csv, path, norm_ts)
    return {"status": "processing started", "file": file.filename}


# ───────────── Data Processing ──────────────────
# Bulk CSV
def process_uploaded_csv(path, norm_ts):
    try:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        PIPELINE_EVENTS[norm_ts] = {
            "status": "processed",
            "time": norm_ts
        }
        _process_and_save(df, norm_ts)
    except Exception as e:
        logger.error(f"CSV processing failed: {e}")

# Process streaming
def process_data(norm_ts):
    try:
        df = pd.read_csv(RAW_CSV, parse_dates=["timestamp"])
        PIPELINE_EVENTS[norm_ts] = {
            "status": "processed",
            "time": norm_ts
        }
        _process_and_save(df, norm_ts)
    except Exception as e:
        logger.error(f"Streamed data processing failed: {e}")


# All processing pipeline
def _process_and_save(df, norm_ts):
    """
    Gap-aware, multi-sensor backfill for OBD-II streams with unknown cadence.
    - Infers sampling interval from data (robust).
    - Inserts placeholder rows for gaps using the inferred interval.
    - Flags only corrupted values (NaN/inf/sentinels); does NOT trim 'extreme but plausible' outliers.
    - Backfills ALL numeric sensors with KNNImputer (+ time as a feature).
    - Keeps your plotting, Drive upload, and PIPELINE_EVENTS wiring intact.
    """
    global DRIVE_STYLE, FUEL_EFFICIENCY
    logger.info("🔧 Cleaning started (auto-interval, KNN for all sensors)")

    # ----------------------- helpers (scoped locally) -----------------------
    protected_cols = {"timestamp", "driving_style"}
    SENTINELS = {-22, -40, 255}

    def _to_dt(_df: pd.DataFrame) -> pd.DataFrame:
        _df = _df.copy()
        _df["timestamp"] = pd.to_datetime(_df["timestamp"], errors="coerce", utc=True)
        _df = _df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        # drop exact duplicate timestamps (keep first)
        _df = _df[~_df["timestamp"].duplicated(keep="first")].reset_index(drop=True)
        return _df

    def _drop_dead_weight(_df: pd.DataFrame) -> pd.DataFrame:
        _df = _df.copy()
        # drop all-NaN or constant columns (except protected)
        drop_cols = [c for c in _df.columns
                     if c not in protected_cols and (_df[c].nunique(dropna=True) <= 1 or _df[c].isna().all())]
        if drop_cols:
            _df.drop(columns=drop_cols, inplace=True, errors="ignore")
        # drop duplicate columns
        _df = _df.loc[:, ~_df.T.duplicated()]
        # drop duplicate rows
        _df.drop_duplicates(inplace=True)
        return _df

    def _normalize_corruption(_df: pd.DataFrame) -> pd.DataFrame:
        _df = _df.copy()
        # normalize obvious corruptions: NaN/inf/sentinels → NaN
        _df.replace(list(SENTINELS), np.nan, inplace=True)
        num_cols = _df.select_dtypes(include=[np.number]).columns
        for c in num_cols:
            s = _df[c]
            s = s.astype(float)
            s[~np.isfinite(s)] = np.nan
            _df[c] = s
        return _df

    def _light_row_col_filters(_df: pd.DataFrame) -> pd.DataFrame:
        _df = _df.copy()
        # keep rows with <=80% NaN (excluding timestamp)
        if "timestamp" in _df.columns and _df.shape[1] > 1:
            keep = _df.drop(columns=["timestamp"]).isna().mean(axis=1) <= 0.8
            _df = _df[keep]
        # prune columns with >80% NaN (except protected)
        na_frac = _df.isna().mean(numeric_only=False)
        high_na = [c for c in na_frac.index if na_frac[c] > 0.8 and c not in protected_cols]
        if high_na:
            _df.drop(columns=high_na, inplace=True, errors="ignore")
        # keep rows that have >1 observed value across non-timestamp columns
        if "timestamp" in _df.columns and _df.shape[1] > 1:
            valid = _df.drop(columns=["timestamp"]).notna().sum(axis=1) > 1
            _df = _df[valid]
        return _df

    def _infer_base_interval_seconds(ts: pd.Series) -> float:
        """
        Robustly infer base cadence from timestamp diffs.
        Strategy:
          - take positive diffs
          - winsorize to 5–95% to reduce impact of long gaps
          - compute a 'rounded mode' on 10ms grid; fall back to median if needed
        """
        if ts.size < 2:
            return 1.0  # fallback
        diffs = ts.sort_values().diff().dropna().dt.total_seconds()
        diffs = diffs[diffs > 0]
        if diffs.empty:
            return 1.0
        q05, q95 = diffs.quantile([0.05, 0.95])
        core = diffs[(diffs >= q05) & (diffs <= q95)]
        if core.empty:
            core = diffs
        # round to 10ms and take the most frequent bin
        rounded = (core / 0.01).round() * 0.01
        mode = rounded.mode()
        if not mode.empty:
            est = float(mode.iloc[0])
        else:
            est = float(core.median())
        # guardrails
        if est <= 0:
            est = float(core.median())
        logger.info(f"⏱️ Inferred base interval ≈ {est:.3f}s")
        return est

    def _insert_time_gaps(_df: pd.DataFrame, base_sec: float) -> pd.DataFrame:
        """
        Insert placeholder rows at multiples of inferred base_sec when gaps exceed ~1.5× base.
        All numeric columns are NaN in inserted rows; non-numeric are forward-filled (except protected).
        """
        if _df.empty:
            return _df
        _df = _df.copy()
        _df = _to_dt(_df)
        expected = timedelta(seconds=base_sec)
        # tolerance ~ half interval to avoid jittery inserts
        tol = timedelta(seconds=0.5 * base_sec)
        # Normalize data
        num_cols = _df.select_dtypes(include=[np.number]).columns.tolist()
        non_num_cols = [c for c in _df.columns if c not in num_cols]
        # Missing detection on interval expectation
        rows = [_df.iloc[0].copy()]
        for i in range(1, len(_df)):
            prev = _df.iloc[i - 1]
            curr = _df.iloc[i]
            dt = curr["timestamp"] - prev["timestamp"]
            if dt > expected * 1.5 + tol:
                n_missing = int(round(dt / expected)) - 1
                if n_missing > 0:
                    for j in range(1, n_missing + 1):
                        gap = prev.copy()
                        gap["timestamp"] = prev["timestamp"] + j * expected
                        # numeric sensors left as NaN to be imputed
                        for c in num_cols:
                            if c not in protected_cols:
                                gap[c] = np.nan
                        # for non-numeric, keep last known (except protected)
                        for c in non_num_cols:
                            if c not in protected_cols:
                                gap[c] = prev[c]
                        rows.append(gap)
            rows.append(curr.copy())
        # Sorting
        out = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        return out

    def _knn_impute_all(_df: pd.DataFrame) -> pd.DataFrame:
        """
        Backfill ALL numeric sensors jointly with KNN, using time (ts_sec) as an additional feature.
        """
        _df = _df.copy()
        _df["ts_sec"] = (_df["timestamp"] - _df["timestamp"].min()).dt.total_seconds()
        # Normalize data
        num_cols = _df.select_dtypes(include=[np.number]).columns.tolist()
        # ensure ts_sec included
        if "ts_sec" not in num_cols:
            num_cols = num_cols + ["ts_sec"]
        # Build imputation frame and remember order
        X = _df[num_cols].copy()
        non_missing_rows = X.dropna().shape[0]
        k = min(5, max(1, non_missing_rows))
        logger.info(f"🤝 KNNImputer n_neighbors={k} on {len(num_cols)} features")
        # Impute and backfill data using KNN 
        imputer = KNNImputer(n_neighbors=k)
        X_imp = imputer.fit_transform(X)
        X_imp = pd.DataFrame(X_imp, columns=num_cols, index=_df.index)
        # Write back (excluding ts_sec)
        for c in num_cols:
            if c == "ts_sec":
                continue
            _df[c] = X_imp[c]

        _df.drop(columns=["ts_sec"], inplace=True)
        return _df

    # Copy data from selective sensor types for Feature Engineering
    def _feature_engineering(_df: pd.DataFrame) -> pd.DataFrame:
        _df = _df.copy()
        if {"ENGINE_LOAD", "ABSOLUTE_LOAD"}.issubset(_df.columns):
            _df["AVG_ENGINE_LOAD"] = _df[["ENGINE_LOAD", "ABSOLUTE_LOAD"]].mean(axis=1)
        if {"INTAKE_TEMP", "OIL_TEMP", "COOLANT_TEMP"}.issubset(_df.columns):
            _df["TEMP_MEAN"] = _df[["INTAKE_TEMP", "OIL_TEMP", "COOLANT_TEMP"]].mean(axis=1)
        if {"MAF", "RPM"}.issubset(_df.columns):
            _df["AIRFLOW_PER_RPM"] = _df["MAF"] / _df["RPM"].replace(0, np.nan)
        return _df

    # Apply StandardScaler to match training preprocessing
    def _scale_numeric(_df: pd.DataFrame) -> pd.DataFrame:
        _df = _df.copy()
        num_cols = _df.select_dtypes(include=[np.number]).columns.tolist()
        for c in list(protected_cols):
            if c in num_cols:
                num_cols.remove(c)
        if num_cols:
            scaler = StandardScaler()
            _df[num_cols] = scaler.fit_transform(_df[num_cols])
        return _df

    # Correlation heatmap plotter
    def _plot_corr(_df: pd.DataFrame, _id: str):
        try:
            num = _df.select_dtypes(include=[np.number])
            if num.shape[1] < 2:
                return
            plt.figure(figsize=(12, 10))
            sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm")
            plt.title("Correlation Between Numeric OBD-II Variables")
            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, f"heatmap_{_id}.png"))
            plt.close()
        except Exception as e:
            logger.error(f"Heatmap generation failed: {e}")

    # Sensor trend plotter
    def _plot_trend(_df: pd.DataFrame, _id: str):
        try:
            plt.figure(figsize=(15, 6))
            for col in ['RPM', 'ENGINE_LOAD', 'ABSOLUTE_LOAD', 'COOLANT_TEMP',
                        'INTAKE_TEMP', 'OIL_TEMP', 'INTAKE_PRESSURE', 'BAROMETRIC_PRESSURE',
                        'CONTROL_MODULE_VOLTAGE']:
                if col in _df.columns:
                    plt.plot(_df.index, _df[col], label=col)
            plt.title("Sensor Trends (Index-Based, No Time Gaps)")
            plt.xlabel("Sample Index")
            plt.ylabel("Sensor Value")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, f"trend_{_id}.png"))
            plt.close()
        except Exception as e:
            logger.error(f"Trend plot failed: {e}")

    # ----------------------- pipeline -----------------------
    df = df.copy()
    # 0) Basic tidy
    df = _drop_dead_weight(df)
    df = _to_dt(df)
    # 1) Corruption-only normalization (no outlier trimming)
    df = _normalize_corruption(df)
    # 2) Light row/column filtering for extreme sparsity
    df = _light_row_col_filters(df)
    # 3) Auto infer base interval & insert gap rows
    base_sec = _infer_base_interval_seconds(df["timestamp"])
    df = _insert_time_gaps(df, base_sec)
    # 4) KNN backfill all numeric sensors (time-aware)
    df = _knn_impute_all(df)
    # 5) Feature engineering AFTER imputation
    df = _feature_engineering(df)
    # 6) Final sort / index
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    # 7) Note: Scaling is now handled by UL labeler to match training pipeline
    # 8) Save
    out_path = os.path.join(CLEANED_DIR, f"cleaned_{norm_ts}.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"✅ Cleaned saved: {out_path}")
    # 9) UL drivestyle predictions
    df_for_persist = df
    labeled_path = None
    try:
        ul = ULLabeler.get()
        preds = ul.predict_df(df)
        # update global DRIVE_STYLE (overwrite if already exists)
        DRIVE_STYLE = [str(p) for p in preds]
        # write labeled CSV (driving_style column)
        df_labeled = df.copy()
        df_labeled["driving_style"] = DRIVE_STYLE
        labeled_path = os.path.join(CLEANED_DIR, f"cleaned_{norm_ts}_labeled.csv")
        df_labeled.to_csv(labeled_path, index=False)
        df_for_persist = df_labeled
        # Update the global DRIVE_STYLE list
        logger.info(f"✅ UL labels generated ({len(DRIVE_STYLE)}) → {labeled_path}")
    except Exception as e:
        logger.error(f"❌ UL labeling failed: {e}")
        import traceback
        logger.error(f"❌ UL labeling traceback: {traceback.format_exc()}")
        # Fallback: provide default labels
        logger.warning("⚠️ Using fallback: default 'Moderate' labels")
        DRIVE_STYLE = ["Moderate"] * len(df)
        df_labeled = df.copy()
        df_labeled["driving_style"] = DRIVE_STYLE
        labeled_path = os.path.join(CLEANED_DIR, f"cleaned_{norm_ts}_labeled.csv")
        df_labeled.to_csv(labeled_path, index=False)
        df_for_persist = df_labeled
        logger.info(f"✅ Fallback labels generated ({len(DRIVE_STYLE)}) → {labeled_path}")
    
    # 9.5) Fuel efficiency predictions
    efficiency_path = None
    try:
        efficiency_labeler = EfficiencyLabeler.get()
        efficiency_preds = efficiency_labeler.predict_df(df)
        # update global FUEL_EFFICIENCY (overwrite if already exists)
        # Efficiency is predicted per drive, so replicate the single score for all rows
        efficiency_score = float(efficiency_preds[0]) if efficiency_preds else 0.0
        FUEL_EFFICIENCY = [efficiency_score] * len(df_for_persist)
        # write efficiency CSV (fuel_efficiency column)
        df_efficiency = df_for_persist.copy()
        df_efficiency["fuel_efficiency"] = FUEL_EFFICIENCY
        efficiency_path = os.path.join(CLEANED_DIR, f"cleaned_{norm_ts}_efficiency.csv")
        df_efficiency.to_csv(efficiency_path, index=False)
        df_for_persist = df_efficiency
        # Update the global FUEL_EFFICIENCY list
        logger.info(f"✅ Fuel efficiency scores generated ({len(FUEL_EFFICIENCY)}) → {efficiency_path}")
        logger.info(f"📊 Drive efficiency: {FUEL_EFFICIENCY[0]:.1f}%" if FUEL_EFFICIENCY else "No efficiency score")
    except Exception as e:
        logger.error(f"❌ Fuel efficiency scoring failed: {e}")
    # 10) Plots
    _plot_corr(df, norm_ts)
    _plot_trend(df, norm_ts)
    # 11) Update event
    try:
        PIPELINE_EVENTS[norm_ts]["status"] = "done"
    except Exception:
        pass
    # 12) Upload to Drive
    try:
        if drive_saver.is_service_available():
            if labeled_path and os.path.exists(labeled_path):
                drive_saver.upload_csv_to_drive(labeled_path)
                logger.info("✅ Uploaded labeled to Google Drive")
            else:
                drive_saver.upload_csv_to_drive(out_path)
                logger.info("✅ Uploaded default to Google Drive")
        else:
            logger.warning("⚠️  Google Drive service not available")
    except Exception as e:
        logger.error(f"❌ Drive upload error: {e}")
    # 13) Save to MongoDB
    try:
        if mongo_saver.is_connected():
            # Save the cleaned DataFrame directly to MongoDB
            session_id = f"session_{norm_ts}"
            if mongo_saver.save_dataframe_to_mongo(df_for_persist, session_id):
                logger.info("✅ Saved to MongoDB")
            else:
                logger.warning("⚠️  MongoDB save failed")
        else:
            logger.warning("⚠️  MongoDB not connected")
    except Exception as e:
            logger.error(f"❌ MongoDB save error: {e}")
    # 14) Save to Firebase Storage (incremented NNN_YYYY-MM-DD_processed.csv at fixed path)
    try:
        if firebase_saver and firebase_saver.is_available():
            # Choose the final artifact to persist
            if labeled_path and os.path.exists(labeled_path):
                target_path = labeled_path
            else:
                target_path = out_path
            # Optional: use the acquisition date if norm_ts starts with YYYY-MM-DD, else let saver use AUS/Melbourne "today"
            date_str = None
            try:
                date_str = str(norm_ts)[:10] if norm_ts and len(str(norm_ts)) >= 10 else None
            except Exception:
                date_str = None
            # Upload with auto-incremented name: NNN_YYYY-MM-DD_processed.csv under skyledge/processed
            gs_url = firebase_saver.upload_file_with_increment(target_path, date_str=date_str)
            # Save to Firebase Storage (incremented NNN_YYYY-MM-DD_processed.csv at fixed path)
            if gs_url:
                logger.info(f"✅ Saved to Firebase Storage: {gs_url}")
                
                # Extract filename from gs_url for efficiency data storage
                try:
                    filename = gs_url.split('/')[-1]  # Get filename from gs://bucket/path/filename.csv
                    if FUEL_EFFICIENCY and len(FUEL_EFFICIENCY) > 0:
                        efficiency_score = FUEL_EFFICIENCY[0]  # Get the first (and only) efficiency score
                        success = save_efficiency_data(filename, efficiency_score)
                        if success:
                            logger.info(f"✅ Efficiency data saved for {filename}: {efficiency_score}%")
                        else:
                            logger.warning(f"⚠️ Failed to save efficiency data for {filename}")
                    else:
                        logger.warning("⚠️ No fuel efficiency data available to save")
                except Exception as e:
                    logger.error(f"❌ Error saving efficiency data: {e}")
            else:
                logger.warning("⚠️ Firebase Storage upload returned empty URL")
        else:
            logger.warning("⚠️ Firebase Storage not available")
    except Exception as e:
        logger.error(f"❌ Firebase Storage save error: {e}")



# ───────────── Health Check ──────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/models/status")
def models_status():
    """Check if models are loaded and available"""
    try:
        # Driver behavior model status
        ul_model_dir = pathlib.Path(os.getenv("MODEL_DIR", "/app/models/ul"))
        ul_required_files = ["label_encoder_ul.pkl", "scaler_ul.pkl", "xgb_drivestyle_ul.pkl"]
        
        ul_available_files = []
        ul_missing_files = []
        
        for file in ul_required_files:
            file_path = ul_model_dir / file
            if file_path.exists():
                ul_available_files.append(file)
            else:
                ul_missing_files.append(file)
        
        ul_status = "ready" if len(ul_available_files) == len(ul_required_files) else "loading"
        
        # Fuel efficiency model status
        efficiency_model_dir = pathlib.Path(os.getenv("EFFICIENCY_MODEL_DIR", "/app/models/efficiency"))
        efficiency_required_files = ["efficiency_model.joblib"]
        
        efficiency_available_files = []
        efficiency_missing_files = []
        
        for file in efficiency_required_files:
            file_path = efficiency_model_dir / file
            if file_path.exists():
                efficiency_available_files.append(file)
            else:
                efficiency_missing_files.append(file)
        
        efficiency_status = "ready" if len(efficiency_available_files) == len(efficiency_required_files) else "loading"
        
        return {
            "driver_behavior": {
                "status": ul_status,
                "model_directory": str(ul_model_dir),
                "available_files": ul_available_files,
                "missing_files": ul_missing_files,
                "total_files": len(ul_required_files),
                "loaded_files": len(ul_available_files)
            },
            "fuel_efficiency": {
                "status": efficiency_status,
                "model_directory": str(efficiency_model_dir),
                "available_files": efficiency_available_files,
                "missing_files": efficiency_missing_files,
                "total_files": len(efficiency_required_files),
                "loaded_files": len(efficiency_available_files)
            },
            "overall_status": "ready" if (ul_status == "ready" and efficiency_status == "ready") else "loading"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }


# ─────── Send status to frontend ─────────────────
@app.get("/events")
def get_events():
    return PIPELINE_EVENTS

@app.get("/predictions/latest")
def get_latest_predictions():
    """Get the latest driver behavior and fuel efficiency predictions"""
    return {
        "driver_behavior": DRIVE_STYLE,
        "fuel_efficiency": FUEL_EFFICIENCY,
        "timestamp": datetime.datetime.now().isoformat(),
        "driver_behavior_count": len(DRIVE_STYLE),
        "fuel_efficiency_count": len(FUEL_EFFICIENCY)
    }

@app.get("/efficiency/{filename}")
def get_efficiency_by_filename(filename: str):
    """
    Get fuel efficiency prediction for a specific processed file from Firebase.
    
    Args:
        filename: The processed filename (e.g., "001_2024-12-01_processed.csv")
        
    Returns:
        dict: Efficiency data or error message
    """
    try:
        if not firebase_saver.is_available():
            raise HTTPException(status_code=503, detail="Firebase Storage not available")
        
        efficiency_data = firebase_get_efficiency_by_filename(filename)
        
        if efficiency_data is None:
            raise HTTPException(status_code=404, detail=f"Efficiency data not found for {filename}")
        
        return {
            "filename": filename,
            "efficiency_score": efficiency_data["efficiency_score"],
            "timestamp": efficiency_data["timestamp"],
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error retrieving efficiency data for {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve efficiency data: {str(e)}")


# ────── Delete event from dashboard ──────────────
@app.delete("/events/remove/{timestamp}")
def remove_event(timestamp: str):
    if timestamp in PIPELINE_EVENTS:
        del PIPELINE_EVENTS[timestamp]
    return {"status": "deleted"}


# ───────────── Download Cleaned ──────────────────
@app.get("/download/{filename}")
def download_file(filename: str):
    path = os.path.join(CLEANED_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path, media_type='text/csv', filename=filename)


# ───────────── MongoDB Operations ──────────────────
@app.get("/mongo/status")
def mongo_status():
    """Check MongoDB connection status"""
    return {
        "connected": mongo_saver.is_connected(),
        "available": MONGODB_AVAILABLE if 'MONGODB_AVAILABLE' in globals() else False
    }


@app.get("/mongo/sessions")
def get_mongo_sessions():
    """Get summary of all MongoDB sessions"""
    if not mongo_saver.is_connected():
        raise HTTPException(status_code=503, detail="MongoDB not connected")
    
    sessions = mongo_saver.get_session_summary()
    return {"sessions": sessions}


@app.get("/mongo/query")
def query_mongo_data(
    session_id: str = None,
    driving_style: str = None,
    start_time: str = None,
    end_time: str = None,
    limit: int = 1000
):
    """Query data from MongoDB with filters"""
    if not mongo_saver.is_connected():
        raise HTTPException(status_code=503, detail="MongoDB not connected")
    
    # Parse datetime strings if provided
    start_dt = None
    end_dt = None
    
    if start_time:
        try:
            start_dt = pd.to_datetime(start_time)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid start_time format")
    
    if end_time:
        try:
            end_dt = pd.to_datetime(end_time)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid end_time format")
    
    results = mongo_saver.query_data(
        session_id=session_id,
        driving_style=driving_style,
        start_time=start_dt,
        end_time=end_dt,
        limit=limit
    )
    
    return {"results": results, "count": len(results)}


@app.post("/mongo/save-csv")
async def save_csv_to_mongo_endpoint(
    file: UploadFile = File(...),
    session_id: str = None
):
    """Save uploaded CSV directly to MongoDB"""
    if not mongo_saver.is_connected():
        raise HTTPException(status_code=503, detail="MongoDB not connected")
    
    try:
        # Save uploaded file temporarily
        temp_path = os.path.join(BASE_DIR, f"temp_{file.filename}")
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        # Save to MongoDB
        success = mongo_saver.save_csv_to_mongo(temp_path, session_id)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if success:
            return {"status": "success", "message": "CSV saved to MongoDB"}
        else:
            raise HTTPException(status_code=500, detail="Failed to save to MongoDB")
            
    except Exception as e:
        logger.error(f"CSV to MongoDB save failed: {e}")
        raise HTTPException(status_code=500, detail=f"Save failed: {str(e)}")


# ───────────── RLHF Training Endpoints ─────────────

class RLHFTrainingRequest(BaseModel):
    max_datasets: int = 10
    force_retrain: bool = False

class RLHFTrainingResponse(BaseModel):
    status: str
    model_version: str = None
    datasets_processed: int = 0
    samples_processed: int = 0
    performance_metrics: dict = None
    error: str = None
    timestamp: str = None

@app.post("/rlhf/train", response_model=RLHFTrainingResponse)
async def trigger_rlhf_training(
    request: RLHFTrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    Trigger RLHF (Reinforcement Learning from Human Feedback) training session.
    
    This endpoint:
    1. Loads human-labeled data from Firebase storage (skyledge/labeled)
    2. Combines it with existing model predictions for RLHF
    3. Retrains the XGBoost model with the combined dataset
    4. Saves the new model to Hugging Face Hub
    """
    try:
        logger.info(f"🚀 RLHF training requested with max_datasets={request.max_datasets}")
        
        # Initialize trainer
        trainer = RLHFTrainer()
        
        # Run training
        result = trainer.train(max_datasets=request.max_datasets)
        
        if result["status"] == "success":
            logger.info(f"✅ RLHF training completed: v{result['model_version']}")
            return RLHFTrainingResponse(
                status="success",
                model_version=result["model_version"],
                datasets_processed=result["datasets_processed"],
                samples_processed=result["samples_processed"],
                performance_metrics=result["performance_metrics"],
                timestamp=datetime.datetime.now().isoformat()
            )
        elif result["status"] == "no_data":
            logger.info("ℹ️ No new data available for RLHF training")
            return RLHFTrainingResponse(
                status="no_data",
                timestamp=datetime.datetime.now().isoformat()
            )
        else:
            logger.error(f"❌ RLHF training failed: {result.get('error', 'Unknown error')}")
            return RLHFTrainingResponse(
                status="error",
                error=result.get("error", "Unknown error"),
                timestamp=datetime.datetime.now().isoformat()
            )
            
    except Exception as e:
        logger.error(f"❌ RLHF training endpoint failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"RLHF training failed: {str(e)}"
        )

@app.get("/rlhf/status")
async def get_rlhf_status():
    """
    Get status of RLHF training system and available labeled data.
    """
    try:
        from train import LabeledDataLoader
        
        loader = LabeledDataLoader()
        datasets = loader.list_labeled_datasets()
        
        return {
            "status": "available",
            "labeled_datasets_count": len(datasets),
            "datasets": [
                {
                    "name": d["name"],
                    "size": d["size"],
                    "created": d["created"]
                } for d in datasets[:10]  # Limit to first 10 for response size
            ],
            "firebase_bucket": "skyledge-36b56.firebasestorage.app",
            "labeled_path": "skyledge/labeled",
            "timestamp": datetime.datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ RLHF status check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Status check failed: {str(e)}"
        )

@app.get("/rlhf/trained-datasets")
async def get_trained_datasets():
    """
    Get list of datasets that have already been used for training.
    """
    try:
        from train import LabeledDataLoader
        
        loader = LabeledDataLoader()
        trained_datasets = loader._get_trained_datasets()
        
        return {
            "trained_datasets_count": len(trained_datasets),
            "trained_datasets": trained_datasets,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get trained datasets: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get trained datasets: {str(e)}"
        )

@app.get("/rlhf/pending-datasets")
async def get_pending_datasets():
    """
    Get list of datasets that are available for training but haven't been trained yet.
    """
    try:
        from train import LabeledDataLoader
        
        loader = LabeledDataLoader()
        
        # Get all labeled datasets
        all_datasets = loader.list_labeled_datasets()
        
        # Get trained datasets
        trained_datasets = loader._get_trained_datasets()
        
        # Filter out trained datasets to get pending ones
        pending_datasets = []
        for dataset in all_datasets:
            dataset_name = dataset['name']
            # Check if this dataset has been trained
            is_trained = any(dataset_name in entry for entry in trained_datasets)
            if not is_trained:
                pending_datasets.append(dataset)
        
        return {
            "pending_datasets_count": len(pending_datasets),
            "pending_datasets": pending_datasets,
            "total_available": len(all_datasets),
            "already_trained": len(trained_datasets),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get pending datasets: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get pending datasets: {str(e)}"
        )

@app.get("/rlhf/latest-model")
async def get_latest_model_version():
    """
    Get the latest model version information for the UI.
    """
    try:
        from utils.dbehavior_download import get_latest_version
        
        # Get the latest version from Hugging Face
        latest_version = get_latest_version()
        
        if latest_version:
            return {
                "status": "available",
                "latest_version": latest_version,
                "model_repository": "BinKhoaLe1812/Driver_Behavior_OBD",
                "version_format": "semantic (v1.0, v1.1, v2.0, etc.)",
                "timestamp": datetime.datetime.now().isoformat()
            }
        else:
            return {
                "status": "no_models",
                "latest_version": None,
                "model_repository": "BinKhoaLe1812/Driver_Behavior_OBD",
                "message": "No trained models found in repository",
                "timestamp": datetime.datetime.now().isoformat()
            }
        
    except Exception as e:
        logger.error(f"❌ Failed to get latest model version: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get latest model version: {str(e)}"
        )