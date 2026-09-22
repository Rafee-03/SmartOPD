import io
from datetime import datetime
from typing import List

import qrcode
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from .. import models, schemas, queue_service
from ..database import get_db

router = APIRouter(prefix="/patients", tags=["patients"])


@router.get("/departments", response_model=List[schemas.DepartmentOut])
def list_departments(db: Session = Depends(get_db)):
    return db.query(models.Department).filter(models.Department.is_active == True).all()  # noqa: E712


@router.post("/register", response_model=schemas.QueueEntryOut)
def register(payload: schemas.RegisterPatientRequest, db: Session = Depends(get_db)):
    try:
        entry = queue_service.register_patient(
            db, payload.department_id, payload.doctor_id, payload.priority, payload.emergency, payload.reason
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return entry


def _patients_ahead(db: Session, entry: models.QueueEntry) -> int:
    q = db.query(models.QueueEntry).filter(
        models.QueueEntry.department_id == entry.department_id,
        models.QueueEntry.status.in_([models.QueueStatus.WAITING, models.QueueStatus.CALLED]),
        models.QueueEntry.id != entry.id,
    )
    if entry.doctor_id:
        q = q.filter(models.QueueEntry.doctor_id == entry.doctor_id)
    # count only those effectively "ahead" per the same ordering rule used to call patients
    ahead = q.filter(
        (models.QueueEntry.priority == True) |  # noqa: E712
        ((models.QueueEntry.priority == entry.priority) & (models.QueueEntry.registration_time < entry.registration_time))
    ).count() if not entry.priority else q.filter(
        models.QueueEntry.priority == True,  # noqa: E712
        models.QueueEntry.registration_time < entry.registration_time,
    ).count()
    return ahead


@router.get("/status/{public_token}", response_model=schemas.PatientStatusOut)
def get_status(public_token: str, db: Session = Depends(get_db)):
    entry = db.query(models.QueueEntry).filter(models.QueueEntry.public_token == public_token).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Token not found")

    ahead = _patients_ahead(db, entry)

    preds = (
        db.query(models.Prediction)
        .filter(models.Prediction.queue_entry_id == entry.id)
        .order_by(models.Prediction.generated_at.desc())
        .limit(2)
        .all()
    )
    latest = preds[0] if preds else None
    previous = preds[1] if len(preds) > 1 else None

    if entry.prediction_status == "INSUFFICIENT_DATA":
        message = "Insufficient historical data for a reliable AI wait-time estimate yet. Your position in the queue is still tracked live."
    else:
        message = "Estimate based on live queue conditions and model validation results. It may change if priority or emergency patients are added."

    return schemas.PatientStatusOut(
        token_number=entry.token_number,
        status=entry.status.value,
        estimated_wait_minutes=entry.estimated_wait_minutes,
        estimated_consultation_time=entry.estimated_consultation_time,
        patients_ahead=ahead,
        prediction_status=entry.prediction_status,
        prediction_message=message,
        last_updated=latest.generated_at if latest else entry.updated_at,
        previous_estimated_wait_minutes=previous.predicted_wait_minutes if previous else None,
    )


@router.get("/status/{public_token}/qrcode")
def get_qrcode(public_token: str, db: Session = Depends(get_db)):
    entry = db.query(models.QueueEntry).filter(models.QueueEntry.public_token == public_token).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Token not found")
    # QR encodes only the opaque random public_token identifier, never PII.
    img = qrcode.make(f"/patients/status/{public_token}")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@router.get("/queue/{department_id}/live")
def live_queue(department_id: int, db: Session = Depends(get_db)):
    """Public live queue display — token numbers only, no patient names (section 17/27)."""
    serving = (
        db.query(models.QueueEntry)
        .filter(models.QueueEntry.department_id == department_id, models.QueueEntry.status == models.QueueStatus.IN_CONSULTATION)
        .order_by(models.QueueEntry.consultation_start.desc())
        .first()
    )
    called = (
        db.query(models.QueueEntry)
        .filter(models.QueueEntry.department_id == department_id, models.QueueEntry.status == models.QueueStatus.CALLED)
        .order_by(models.QueueEntry.called_time.desc())
        .first()
    )
    upcoming = (
        db.query(models.QueueEntry)
        .filter(models.QueueEntry.department_id == department_id, models.QueueEntry.status == models.QueueStatus.WAITING)
        .order_by(models.QueueEntry.priority.desc(), models.QueueEntry.registration_time.asc())
        .limit(6)
        .all()
    )
    return {
        "currently_serving": serving.token_number if serving else (called.token_number if called else None),
        "upcoming_tokens": [e.token_number for e in upcoming],
    }
