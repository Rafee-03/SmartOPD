"""
Reproducible training pipeline (section 32):

  raw dataset -> validation -> preprocessing -> feature engineering ->
  time-aware train/test split -> train several real regressors ->
  evaluate (MAE/RMSE/R2) -> pick the best on validation results ->
  version + save -> (caller registers it in model_versions table)

This is a genuine scikit-learn pipeline. Nothing here is a hard-coded
formula, a manual weight, or an if/else rule pretending to be ML.

Minimum viable training size: if fewer than MIN_TRAINING_ROWS valid rows
are available, training is refused and the caller must show the
"insufficient historical data" state (section 13) instead of faking a
model.
"""
import json
import os
from datetime import datetime
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .features import FEATURE_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

MIN_TRAINING_ROWS = 50  # below this, cold-start mode applies (section 13)
MODELS_DIR = os.getenv("MODELS_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "..", "models"))


class InsufficientDataError(Exception):
    pass


def _compute_doctor_stats(train_df: pd.DataFrame) -> dict:
    """
    Per-doctor average consultation duration, computed ONLY from the
    training split. This value is later stored and reused identically at
    live-prediction time (never recomputed from test/future data) to
    avoid leakage.
    """
    if "consultation_duration_minutes" not in train_df.columns:
        return {}
    stats = (
        train_df.groupby("doctor")["consultation_duration_minutes"]
        .mean()
        .dropna()
        .to_dict()
    )
    return {k: float(v) for k, v in stats.items()}


def _build_feature_matrix(df: pd.DataFrame, doctor_stats: dict, global_avg: float) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        reg_time = r["registration_time_parsed"] if "registration_time_parsed" in df.columns else r["registration_time"]
        doctor = r.get("doctor")
        rows.append({
            "hour_of_day": reg_time.hour,
            "day_of_week": reg_time.weekday(),
            "department": r.get("department", "UNKNOWN"),
            "doctor": doctor or "UNASSIGNED",
            "priority": int(bool(r.get("priority", False))),
            "emergency": int(bool(r.get("emergency", False))),
            "patients_ahead": int(r.get("patients_ahead_at_registration", 0) or 0),
            "priority_patients_ahead": int(r.get("priority_patients_ahead_at_registration", 0) or 0),
            "queue_size_at_registration": int(r.get("patients_ahead_at_registration", 0) or 0),
            "doctor_avg_consultation_minutes": doctor_stats.get(doctor, global_avg),
            "minutes_since_opd_start": float(r.get("minutes_since_opd_start", 0.0) or 0.0),
        })
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def _make_pipeline(model) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ],
        remainder="passthrough",  # numeric columns pass through unchanged
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def train(clean_df: pd.DataFrame) -> Tuple[Pipeline, dict, dict]:
    """
    clean_df must already be validated (see data_validation.validate_dataframe)
    and contain: registration_time_parsed, department, doctor, priority,
    emergency, patients_ahead_at_registration, priority_patients_ahead_at_registration,
    actual_wait_minutes, and optionally consultation_duration_minutes.

    Returns (fitted_pipeline, metrics_dict, metadata_dict).
    """
    df = clean_df.dropna(subset=["actual_wait_minutes"]).copy()
    if len(df) < MIN_TRAINING_ROWS:
        raise InsufficientDataError(
            f"Only {len(df)} valid records available; need at least {MIN_TRAINING_ROWS}. "
            "Insufficient historical data for reliable ML training."
        )

    # --- time-aware split: sort chronologically, last 20% is the test set.
    # This mirrors how the model will actually be used (predicting the
    # future from the past) rather than a random shuffle, per section 6.
    df = df.sort_values("registration_time_parsed").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    if "minutes_since_opd_start" not in train_df.columns:
        for part in (train_df, test_df):
            day_start = part.groupby(part["registration_time_parsed"].dt.date)["registration_time_parsed"].transform("min")
            part["minutes_since_opd_start"] = (part["registration_time_parsed"] - day_start).dt.total_seconds() / 60.0

    doctor_stats = _compute_doctor_stats(train_df)
    global_avg = float(train_df["consultation_duration_minutes"].mean()) if "consultation_duration_minutes" in train_df.columns else 10.0

    X_train = _build_feature_matrix(train_df, doctor_stats, global_avg)
    y_train = train_df["actual_wait_minutes"].values
    X_test = _build_feature_matrix(test_df, doctor_stats, global_avg)
    y_test = test_df["actual_wait_minutes"].values

    candidates = {
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=3, random_state=42, n_jobs=-1
        ),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
            max_depth=8, learning_rate=0.08, random_state=42
        ),
    }

    results = {}
    fitted = {}
    for name, model in candidates.items():
        pipe = _make_pipeline(model)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = r2_score(y_test, preds) if len(np.unique(y_test)) > 1 else None
        results[name] = {"mae": float(mae), "rmse": rmse, "r2": (float(r2) if r2 is not None else None)}
        fitted[name] = pipe

    # Pick the best model on actual validation MAE — not because a name
    # "sounds more advanced" (section 5).
    best_name = min(results, key=lambda n: results[n]["mae"])
    best_pipeline = fitted[best_name]
    best_metrics = results[best_name]

    metadata = {
        "model_type": best_name,
        "all_candidate_results": results,
        "features_used": FEATURE_COLUMNS,
        "doctor_stats": doctor_stats,
        "global_avg_consultation_minutes": global_avg,
        "training_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "trained_at": datetime.utcnow().isoformat(),
    }
    return best_pipeline, best_metrics, metadata


def save_model(pipeline: Pipeline, metadata: dict, version_name: str) -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)
    file_path = os.path.join(MODELS_DIR, f"{version_name}.joblib")
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, file_path)
    with open(os.path.join(MODELS_DIR, f"{version_name}.meta.json"), "w") as f:
        json.dump({k: v for k, v in metadata.items() if k != "doctor_stats"} | {"doctor_stats": metadata["doctor_stats"]}, f, indent=2, default=str)
    return file_path


def load_model(file_path: str) -> dict:
    return joblib.load(file_path)


def next_version_name(existing_versions: list) -> str:
    nums = []
    for v in existing_versions:
        try:
            nums.append(int(v.replace("model_v", "")))
        except ValueError:
            continue
    return f"model_v{(max(nums) + 1) if nums else 1}"


if __name__ == "__main__":
    # Standalone CLI usage, e.g. from Google Colab or Termux:
    #   python -m app.ml.train path/to/data.csv
    import sys
    from .data_validation import validate_dataframe

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "synthetic_opd_data.csv"
    raw = pd.read_csv(csv_path)
    report = validate_dataframe(raw)
    print("Data quality report:", json.dumps({k: v for k, v in report.items() if k != "clean_df"}, indent=2))
    if report["clean_df"] is None or len(report["clean_df"]) < MIN_TRAINING_ROWS:
        print("Insufficient valid data to train.")
        sys.exit(1)

    pipeline, metrics, metadata = train(report["clean_df"])
    version = next_version_name([])
    path = save_model(pipeline, metadata, version)
    print(f"Saved {version} to {path}")
    print("Metrics:", metrics)
