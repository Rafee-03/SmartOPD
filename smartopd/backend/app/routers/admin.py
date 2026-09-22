import io
import json
from datetime import datetime
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from .. import models, schemas, security
from ..database import get_db
from ..ml.data_validation import validate_dataframe
from ..ml import train as train_module

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(security.require_admin)])


# ---------- Departments / Doctors ----------
class DepartmentCreate(BaseModel):
    name: str
    token_prefix: str = "A"


@router.post("/departments", response_model=schemas.DepartmentOut)
def create_department(payload: DepartmentCreate, db: Session = Depends(get_db)):
    dept = models.Department(name=payload.name, token_prefix=payload.token_prefix)
    db.add(dept)
    db.commit()
    db.refresh(dept)
    return dept


class DoctorCreate(BaseModel):
    name: str
    department_id: int


@router.post("/doctors")
def create_doctor(payload: DoctorCreate, db: Session = Depends(get_db)):
    doc = models.Doctor(name=payload.name, department_id=payload.department_id)
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return {"id": doc.id, "name": doc.name, "department_id": doc.department_id}


# ---------- Staff accounts ----------
class UserCreate(BaseModel):
    username: str
    password: str
    role: str  # "ADMIN" or "STAFF"
    full_name: Optional[str] = None


@router.post("/users")
def create_user(payload: UserCreate, db: Session = Depends(get_db)):
    if payload.role not in ("ADMIN", "STAFF"):
        raise HTTPException(status_code=400, detail="role must be ADMIN or STAFF")
    if db.query(models.User).filter(models.User.username == payload.username).first():
        raise HTTPException(status_code=400, detail="username already exists")
    user = models.User(
        username=payload.username,
        password_hash=security.hash_password(payload.password),
        role=models.RoleEnum(payload.role),
        full_name=payload.full_name,
    )
    db.add(user)
    db.commit()
    return {"id": user.id, "username": user.username, "role": user.role.value}


# ---------- Dataset upload + validation (section 4) ----------
_LAST_UPLOAD_CACHE: dict = {}  # in-memory holding area between /upload and /train (single-process dev use)


@router.post("/dataset/upload", response_model=schemas.DataQualityReport)
async def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)):
    content = await file.read()
    try:
        if file.filename.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse file: {e}")

    report = validate_dataframe(df)
    clean_df = report.pop("clean_df", None)
    _LAST_UPLOAD_CACHE["clean_df"] = clean_df
    _LAST_UPLOAD_CACHE["filename"] = file.filename

    db.add(models.AuditLog(action="DATASET_UPLOAD", detail=json.dumps({"filename": file.filename, **report})))
    db.commit()
    return schemas.DataQualityReport(**report)


@router.post("/dataset/import-live-history")
def import_live_history_into_cache(db: Session = Depends(get_db)):
    """
    Pulls completed live queue entries already stored in historical_records
    (populated automatically as consultations complete) so they can be
    combined with / used instead of an uploaded CSV for retraining.
    """
    rows = db.query(models.HistoricalRecord).filter(models.HistoricalRecord.actual_wait_minutes.isnot(None)).all()
    if not rows:
        raise HTTPException(status_code=400, detail="No completed live records yet.")
    df = pd.DataFrame([{
        "department": r.department,
        "doctor": r.doctor,
        "registration_time_parsed": r.registration_time,
        "consultation_start_parsed": r.consultation_start,
        "consultation_end_parsed": r.consultation_end,
        "priority": r.priority,
        "emergency": r.emergency,
        "actual_wait_minutes": r.actual_wait_minutes,
        "consultation_duration_minutes": r.consultation_duration_minutes,
        "patients_ahead_at_registration": 0,
        "priority_patients_ahead_at_registration": 0,
    } for r in rows])
    _LAST_UPLOAD_CACHE["clean_df"] = df
    _LAST_UPLOAD_CACHE["filename"] = "live_history"
    return {"rows_loaded": len(df)}


@router.post("/train", response_model=schemas.ModelVersionOut)
def train_model(db: Session = Depends(get_db)):
    clean_df = _LAST_UPLOAD_CACHE.get("clean_df")
    if clean_df is None or len(clean_df) == 0:
        raise HTTPException(status_code=400, detail="Upload/validate a dataset first via /admin/dataset/upload.")

    run = models.TrainingRun(status="RUNNING")
    db.add(run)
    db.commit()
    db.refresh(run)

    try:
        pipeline, metrics, metadata = train_module.train(clean_df)
    except train_module.InsufficientDataError as e:
        run.status = "FAILED"
        run.log = str(e)
        run.finished_at = datetime.utcnow()
        db.commit()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        run.status = "FAILED"
        run.log = str(e)
        run.finished_at = datetime.utcnow()
        db.commit()
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")

    existing = [v.version_name for v in db.query(models.ModelVersion).all()]
    version_name = train_module.next_version_name(existing)
    file_path = train_module.save_model(pipeline, metadata, version_name)

    mv = models.ModelVersion(
        version_name=version_name,
        model_type=metadata["model_type"],
        file_path=file_path,
        features_used=json.dumps(metadata["features_used"]),
        dataset_size=metadata["training_rows"] + metadata["test_rows"],
        mae=metrics["mae"],
        rmse=metrics["rmse"],
        r2=metrics["r2"],
        is_active=False,  # admin must explicitly activate (section 12 — no silent auto-replace)
    )
    db.add(mv)
    run.status = "SUCCESS"
    run.finished_at = datetime.utcnow()
    run.model_version_id = None  # set after flush below
    db.commit()
    db.refresh(mv)
    run.model_version_id = mv.id
    db.commit()

    return mv


@router.get("/models", response_model=List[schemas.ModelVersionOut])
def list_models(db: Session = Depends(get_db)):
    return db.query(models.ModelVersion).order_by(models.ModelVersion.training_date.desc()).all()


@router.post("/models/{model_id}/activate", response_model=schemas.ModelVersionOut)
def activate_model(model_id: int, db: Session = Depends(get_db)):
    mv = db.query(models.ModelVersion).get(model_id)
    if not mv:
        raise HTTPException(status_code=404, detail="Model not found")
    db.query(models.ModelVersion).update({models.ModelVersion.is_active: False})
    mv.is_active = True
    db.add(models.AuditLog(action="MODEL_ACTIVATED", detail=mv.version_name))
    db.commit()
    db.refresh(mv)
    return mv


# ---------- Analytics (section 20) ----------
@router.get("/analytics/opd")
def opd_analytics(db: Session = Depends(get_db)):
    total = db.query(models.QueueEntry).count()
    completed = db.query(models.QueueEntry).filter(models.QueueEntry.status == models.QueueStatus.COMPLETED).all()
    waits = [
        (e.consultation_start - e.registration_time).total_seconds() / 60.0
        for e in completed if e.consultation_start
    ]
    no_shows = db.query(models.QueueEntry).filter(models.QueueEntry.status == models.QueueStatus.NO_SHOW).count()
    emergencies = db.query(models.QueueEntry).filter(models.QueueEntry.emergency == True).count()  # noqa: E712

    return {
        "total_patients": total,
        "completed": len(completed),
        "average_wait_minutes": (sum(waits) / len(waits)) if waits else None,
        "median_wait_minutes": (sorted(waits)[len(waits) // 2] if waits else None),
        "max_wait_minutes": max(waits) if waits else None,
        "no_show_count": no_shows,
        "emergency_count": emergencies,
    }


@router.get("/analytics/predictions")
def prediction_accuracy(db: Session = Depends(get_db)):
    preds = db.query(models.Prediction).filter(models.Prediction.error_minutes.isnot(None)).all()
    errors = [p.error_minutes for p in preds]
    abs_errors = [abs(e) for e in errors]
    return {
        "predictions_evaluated": len(preds),
        "mae": (sum(abs_errors) / len(abs_errors)) if abs_errors else None,
        "mean_error_minutes": (sum(errors) / len(errors)) if errors else None,  # bias: over/under-predicting
    }


@router.get("/logs/events")
def event_logs(limit: int = 100, db: Session = Depends(get_db)):
    rows = db.query(models.EventLog).order_by(models.EventLog.created_at.desc()).limit(limit).all()
    return [{"id": r.id, "type": r.event_type, "detail": r.detail, "at": r.created_at, "queue_entry_id": r.queue_entry_id} for r in rows]


@router.get("/logs/audit")
def audit_logs(limit: int = 100, db: Session = Depends(get_db)):
    rows = db.query(models.AuditLog).order_by(models.AuditLog.created_at.desc()).limit(limit).all()
    return [{"id": r.id, "action": r.action, "detail": r.detail, "at": r.created_at, "user_id": r.user_id} for r in rows]
