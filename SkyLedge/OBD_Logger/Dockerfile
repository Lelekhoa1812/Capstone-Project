FROM python:3.11-slim

# ── Create and switch to non-root user ──
RUN useradd -m -u 1000 user
USER user

# ── Set environment and working directory ──
ENV HOME=/home/user
WORKDIR $HOME/app

# ── Upgrade pip and install dependencies ──
COPY --chown=user requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install latest versions for UL model inference
RUN pip install --no-cache-dir huggingface_hub xgboost joblib scikit-learn

# ── Pre-mount GDrive (no-op if creds not found) ──
COPY --chown=user utils/mount_drive.py .
RUN python mount_drive.py || true

# ── Copy application source ──
COPY --chown=user . .

# ── Create required folders ──
RUN mkdir -p $HOME/app/logs \
    $HOME/app/cache \
    $HOME/app/cache/obd_data \
    $HOME/app/cache/obd_data/plots \
    $HOME/app/models/ul \
    $HOME/app/models/efficiency

# ── Environment variables for HuggingFace models ──
ENV MODEL_DIR=$HOME/app/models/ul
ENV HF_MODEL_REPO=BinKhoaLe1812/Driver_Behavior_OBD
ENV EFFICIENCY_MODEL_DIR=$HOME/app/models/efficiency
ENV HF_EFFICIENCY_MODEL_REPO=BinKhoaLe1812/Fuel_Efficiency_OBD

# ── Models will be downloaded at runtime when app starts ──

# ── Default port ──
EXPOSE 7860

# ── Start app ──
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
