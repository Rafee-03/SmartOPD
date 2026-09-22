from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from . import models
from .prediction_service import reforecast_department, predict_for_entry


def _log_event(db: Session, event_type: str, queue_entry_id: Optional[int] = None, detail: str = ""):
    db.add(models.EventLog(event_type=event_type, queue_entry_id=queue_entry_id, detail=detail))
    db.commit()


def _next_token_number(db: Session, department: models.Department) -> str:
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    count_today = (
        db.query(models.QueueEntry)
        .filter(models.QueueEntry.department_id == department.id, models.QueueEntry.registration_time >= today_start)
        .count()
    )
    return f"{department.token_prefix}-{count_today + 1}"


def register_patient(db: Session, department_id: int, doctor_id: Optional[int], priority: bool, emergency: bool, reason: Optional[str] = None) -> models.QueueEntry:
    department = db.query(models.Department).get(department_id)
    if not department:
        raise ValueError("Unknown department")

    entry = models.QueueEntry(
        token_number=_next_token_number(db, department),
        department_id=department_id,
        doctor_id=doctor_id,
        priority=priority,
        emergency=emergency,
        status=models.QueueStatus.WAITING,
        registration_time=datetime.utcnow(),
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)

    _log_event(db, "PATIENT_REGISTERED", entry.id, detail=f"token={entry.token_number}")
    if emergency or priority:
        db.add(models.PriorityEvent(queue_entry_id=entry.id, reason=reason or ("emergency" if emergency else "priority")))
        db.commit()
        _log_event(db, "PRIORITY_PATIENT_ADDED", entry.id, detail=reason or "")
        # Core feature (section 9): a priority/emergency insertion changes the
        # live queue state for EVERYONE waiting in this department, so we
        # re-forecast the whole department, not just add a fixed penalty.
        reforecast_department(db, department_id, reason="PRIORITY_PATIENT_ADDED")
    else:
        # A normal arrival still changes queue_size for those already ahead
        # is unaffected, but we generate this patient's own first prediction.
        predict_for_entry(db, entry, reason="PATIENT_REGISTERED")

    db.refresh(entry)
    return entry


def call_next(db: Session, department_id: int, doctor_id: Optional[int] = None) -> Optional[models.QueueEntry]:
    q = db.query(models.QueueEntry).filter(
        models.QueueEntry.department_id == department_id,
        models.QueueEntry.status == models.QueueStatus.WAITING,
    )
    if doctor_id:
        q = q.filter(models.QueueEntry.doctor_id == doctor_id)
    # Priority patients first, then earliest registration — this ordering
    # reflects hospital policy; the ML model still predicts wait times, it
    # does not decide queue order.
    entry = q.order_by(models.QueueEntry.priority.desc(), models.QueueEntry.registration_time.asc()).first()
    if not entry:
        return None
    entry.status = models.QueueStatus.CALLED
    entry.called_time = datetime.utcnow()
    db.commit()
    _log_event(db, "PATIENT_CALLED", entry.id)
    db.refresh(entry)
    return entry


def start_consultation(db: Session, entry_id: int) -> models.QueueEntry:
    entry = db.query(models.QueueEntry).get(entry_id)
    if not entry:
        raise ValueError("Unknown queue entry")
    entry.status = models.QueueStatus.IN_CONSULTATION
    entry.consultation_start = datetime.utcnow()
    db.commit()
    _log_event(db, "CONSULTATION_STARTED", entry.id)
    reforecast_department(db, entry.department_id, reason="CONSULTATION_STARTED")
    db.refresh(entry)
    return entry


def complete_consultation(db: Session, entry_id: int) -> models.QueueEntry:
    entry = db.query(models.QueueEntry).get(entry_id)
    if not entry:
        raise ValueError("Unknown queue entry")
    entry.status = models.QueueStatus.COMPLETED
    entry.consultation_end = datetime.utcnow()
    db.commit()
    _log_event(db, "CONSULTATION_COMPLETED", entry.id)

    # Record ground truth for prediction-accuracy tracking (section 11)
    # and append to the historical training table for future retraining.
    actual_wait = None
    if entry.consultation_start and entry.registration_time:
        actual_wait = (entry.consultation_start - entry.registration_time).total_seconds() / 60.0
        last_pred = (
            db.query(models.Prediction)
            .filter(models.Prediction.queue_entry_id == entry.id, models.Prediction.prediction_status == "OK")
            .order_by(models.Prediction.generated_at.desc())
            .first()
        )
        if last_pred and last_pred.predicted_wait_minutes is not None:
            last_pred.actual_wait_minutes = actual_wait
            last_pred.error_minutes = last_pred.predicted_wait_minutes - actual_wait

    duration = None
    if entry.consultation_end and entry.consultation_start:
        duration = (entry.consultation_end - entry.consultation_start).total_seconds() / 60.0

    db.add(models.HistoricalRecord(
        source="LIVE",
        department=entry.department.name if entry.department else "UNKNOWN",
        doctor=entry.doctor.name if entry.doctor else None,
        token_number=entry.token_number,
        registration_time=entry.registration_time,
        consultation_start=entry.consultation_start,
        consultation_end=entry.consultation_end,
        priority=entry.priority,
        emergency=entry.emergency,
        actual_wait_minutes=actual_wait,
        consultation_duration_minutes=duration,
    ))
    db.commit()

    reforecast_department(db, entry.department_id, reason="CONSULTATION_COMPLETED")
    db.refresh(entry)
    return entry


def cancel_entry(db: Session, entry_id: int, no_show: bool = False) -> models.QueueEntry:
    entry = db.query(models.QueueEntry).get(entry_id)
    if not entry:
        raise ValueError("Unknown queue entry")
    entry.status = models.QueueStatus.NO_SHOW if no_show else models.QueueStatus.CANCELLED
    db.commit()
    _log_event(db, "PATIENT_NO_SHOW" if no_show else "PATIENT_CANCELLED", entry.id)
    reforecast_department(db, entry.department_id, reason="PATIENT_NO_SHOW" if no_show else "PATIENT_CANCELLED")
    db.refresh(entry)
    return entry


def set_doctor_availability(db: Session, doctor_id: int, available: bool) -> models.Doctor:
    doctor = db.query(models.Doctor).get(doctor_id)
    if not doctor:
        raise ValueError("Unknown doctor")
    doctor.is_available = available
    db.commit()
    _log_event(db, "DOCTOR_RESUMED" if available else "DOCTOR_PAUSED", detail=doctor.name)
    reforecast_department(db, doctor.department_id, reason="DOCTOR_PAUSED" if not available else "DOCTOR_RESUMED")
    db.refresh(doctor)
    return doctor
