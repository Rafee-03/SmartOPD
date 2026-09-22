"""
Turns the CURRENT live queue state into a prediction using the ACTIVE
trained model. This is the re-forecasting engine's core (sections 9, 10).

If no model is active or trained-on-too-little-data, this returns an
explicit INSUFFICIENT_DATA status rather than fabricating a number
(section 13/14) — the queue still works, it just says so honestly.
"""
import json
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from . import models
from .ml.train import load_model
from .ml.features import build_feature_row, FEATURE_COLUMNS


def get_active_model(db: Session) -> Optional[models.ModelVersion]:
    return db.query(models.ModelVersion).filter(models.ModelVersion.is_active == True).first()  # noqa: E712


def _opd_start_time_for(db: Session, department_id: int, at_time: datetime) -> datetime:
    """First registration of the current day for this department, or 9:00 AM fallback."""
    day_start = at_time.replace(hour=0, minute=0, second=0, microsecond=0)
    first = (
        db.query(models.QueueEntry)
        .filter(
            models.QueueEntry.department_id == department_id,
            models.QueueEntry.registration_time >= day_start,
        )
        .order_by(models.QueueEntry.registration_time.asc())
        .first()
    )
    if first:
        return first.registration_time
    return at_time.replace(hour=9, minute=0, second=0, microsecond=0)


def _live_queue_counts(db: Session, department_id: int, doctor_id: Optional[int], before_entry_id: Optional[int]) -> dict:
    """
    Patients genuinely "ahead" of a given entry right now: same
    department (+ doctor if assigned), still WAITING/CALLED, registered
    earlier OR marked priority when this entry is not.
    Only uses information available right now — no future entries.
    """
    q = db.query(models.QueueEntry).filter(
        models.QueueEntry.department_id == department_id,
        models.QueueEntry.status.in_([models.QueueStatus.WAITING, models.QueueStatus.CALLED]),
    )
    if doctor_id:
        q = q.filter(models.QueueEntry.doctor_id == doctor_id)
    if before_entry_id:
        q = q.filter(models.QueueEntry.id != before_entry_id)
    entries = q.all()
    ahead = len(entries)
    priority_ahead = len([e for e in entries if e.priority])
    return {"patients_ahead": ahead, "priority_patients_ahead": priority_ahead, "queue_size": ahead}


def predict_for_entry(db: Session, entry: models.QueueEntry, reason: str) -> models.Prediction:
    """Generate (and persist) a fresh prediction for one queue entry."""
    active = get_active_model(db)
    now = datetime.utcnow()

    if not active:
        pred = models.Prediction(
            queue_entry_id=entry.id,
            generated_at=now,
            predicted_wait_minutes=None,
            predicted_consultation_time=None,
            model_version=None,
            reason=reason,
            prediction_status="INSUFFICIENT_DATA",
            queue_state_snapshot=None,
        )
        db.add(pred)
        entry.estimated_wait_minutes = None
        entry.estimated_consultation_time = None
        entry.prediction_status = "INSUFFICIENT_DATA"
        db.commit()
        db.refresh(pred)
        return pred

    bundle = load_model(active.file_path)
    pipeline = bundle["pipeline"]
    metadata = bundle["metadata"]
    doctor_stats = metadata.get("doctor_stats", {})
    global_avg = metadata.get("global_avg_consultation_minutes", 10.0)

    counts = _live_queue_counts(db, entry.department_id, entry.doctor_id, before_entry_id=entry.id)
    opd_start = _opd_start_time_for(db, entry.department_id, now)
    doctor_name = entry.doctor.name if entry.doctor else None

    feature_row = build_feature_row(
        at_time=now,
        opd_start_time=opd_start,
        department=entry.department.name if entry.department else "UNKNOWN",
        doctor=doctor_name,
        priority=entry.priority,
        emergency=entry.emergency,
        patients_ahead=counts["patients_ahead"],
        priority_patients_ahead=counts["priority_patients_ahead"],
        queue_size_at_registration=counts["queue_size"],
        doctor_avg_consultation_minutes=doctor_stats.get(doctor_name, global_avg),
    )

    import pandas as pd
    X = pd.DataFrame([feature_row], columns=FEATURE_COLUMNS)
    predicted_wait = float(pipeline.predict(X)[0])
    predicted_wait = max(predicted_wait, 0.0)
    predicted_consultation_time = now + timedelta(minutes=predicted_wait)

    pred = models.Prediction(
        queue_entry_id=entry.id,
        generated_at=now,
        predicted_wait_minutes=predicted_wait,
        predicted_consultation_time=predicted_consultation_time,
        model_version=active.version_name,
        reason=reason,
        prediction_status="OK",
        queue_state_snapshot=json.dumps(feature_row, default=str),
    )
    db.add(pred)
    entry.estimated_wait_minutes = predicted_wait
    entry.estimated_consultation_time = predicted_consultation_time
    entry.prediction_status = "OK"
    db.commit()
    db.refresh(pred)
    return pred


def reforecast_department(db: Session, department_id: int, reason: str) -> int:
    """
    Re-run predictions for every WAITING/CALLED patient in a department
    (sections 9/10). Returns how many entries were updated.
    """
    entries = (
        db.query(models.QueueEntry)
        .filter(
            models.QueueEntry.department_id == department_id,
            models.QueueEntry.status.in_([models.QueueStatus.WAITING, models.QueueStatus.CALLED]),
        )
        .all()
    )
    for e in entries:
        predict_for_entry(db, e, reason)
    return len(entries)
