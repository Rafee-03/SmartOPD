from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .. import models, schemas, queue_service, security
from ..database import get_db

router = APIRouter(prefix="/staff", tags=["staff"], dependencies=[Depends(security.require_staff)])


@router.get("/queue/{department_id}", response_model=List[schemas.QueueEntryOut])
def view_queue(department_id: int, db: Session = Depends(get_db)):
    return (
        db.query(models.QueueEntry)
        .filter(models.QueueEntry.department_id == department_id)
        .filter(models.QueueEntry.status.in_([models.QueueStatus.WAITING, models.QueueStatus.CALLED, models.QueueStatus.IN_CONSULTATION]))
        .order_by(models.QueueEntry.priority.desc(), models.QueueEntry.registration_time.asc())
        .all()
    )


class CallNextRequest(BaseModel):
    department_id: int
    doctor_id: Optional[int] = None


@router.post("/queue/call-next", response_model=Optional[schemas.QueueEntryOut])
def call_next(payload: CallNextRequest, db: Session = Depends(get_db)):
    entry = queue_service.call_next(db, payload.department_id, payload.doctor_id)
    return entry


@router.post("/queue/{entry_id}/start", response_model=schemas.QueueEntryOut)
def start(entry_id: int, db: Session = Depends(get_db)):
    try:
        return queue_service.start_consultation(db, entry_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/queue/{entry_id}/complete", response_model=schemas.QueueEntryOut)
def complete(entry_id: int, db: Session = Depends(get_db)):
    try:
        return queue_service.complete_consultation(db, entry_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


class CancelRequest(BaseModel):
    no_show: bool = False


@router.post("/queue/{entry_id}/cancel", response_model=schemas.QueueEntryOut)
def cancel(entry_id: int, payload: CancelRequest, db: Session = Depends(get_db)):
    try:
        return queue_service.cancel_entry(db, entry_id, payload.no_show)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/queue/add-priority", response_model=schemas.QueueEntryOut)
def add_priority(payload: schemas.RegisterPatientRequest, db: Session = Depends(get_db)):
    """Staff-registered emergency / priority walk-in (section 9 — the core feature)."""
    payload.priority = True
    try:
        return queue_service.register_patient(
            db, payload.department_id, payload.doctor_id, True, payload.emergency, payload.reason
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class DoctorAvailabilityRequest(BaseModel):
    available: bool


@router.post("/doctors/{doctor_id}/availability", response_model=None)
def set_availability(doctor_id: int, payload: DoctorAvailabilityRequest, db: Session = Depends(get_db)):
    try:
        doctor = queue_service.set_doctor_availability(db, doctor_id, payload.available)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"id": doctor.id, "name": doctor.name, "is_available": doctor.is_available}
