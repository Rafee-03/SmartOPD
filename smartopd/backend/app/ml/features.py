"""
Feature engineering shared by:
  - ml/train.py          (building the training matrix from historical data)
  - prediction_service.py (building the live feature row at prediction time)

Using the SAME function in both places is what prevents train/serve skew.

LEAKAGE RULE (see project section 6): every feature here must be something
that is actually knowable at the moment a prediction is requested — i.e.
at patient registration time, or at the moment of a re-forecast trigger.
We never use consultation_end, future arrivals, or any patient not yet
registered.
"""
from datetime import datetime
from typing import Optional

FEATURE_COLUMNS = [
    "hour_of_day",
    "day_of_week",
    "department",
    "doctor",
    "priority",
    "emergency",
    "patients_ahead",
    "priority_patients_ahead",
    "queue_size_at_registration",
    "doctor_avg_consultation_minutes",
    "minutes_since_opd_start",
]

CATEGORICAL_COLUMNS = ["department", "doctor"]
NUMERIC_COLUMNS = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]


def build_feature_row(
    *,
    at_time: datetime,
    opd_start_time: datetime,
    department: str,
    doctor: Optional[str],
    priority: bool,
    emergency: bool,
    patients_ahead: int,
    priority_patients_ahead: int,
    queue_size_at_registration: int,
    doctor_avg_consultation_minutes: float,
) -> dict:
    """Build one feature row (dict) for either training or live prediction."""
    minutes_since_start = max((at_time - opd_start_time).total_seconds() / 60.0, 0.0)
    return {
        "hour_of_day": at_time.hour,
        "day_of_week": at_time.weekday(),  # 0=Monday
        "department": department or "UNKNOWN",
        "doctor": doctor or "UNASSIGNED",
        "priority": int(bool(priority)),
        "emergency": int(bool(emergency)),
        "patients_ahead": int(patients_ahead),
        "priority_patients_ahead": int(priority_patients_ahead),
        "queue_size_at_registration": int(queue_size_at_registration),
        "doctor_avg_consultation_minutes": float(doctor_avg_consultation_minutes),
        "minutes_since_opd_start": float(minutes_since_start),
    }
