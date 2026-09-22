"""
Real relational schema (SQLAlchemy ORM). Works unchanged on SQLite or
Postgres. Foreign keys + constraints are used throughout, per spec.
"""
import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, ForeignKey,
    Enum, Text, UniqueConstraint
)
from sqlalchemy.orm import relationship

from .database import Base


def gen_uuid():
    return str(uuid.uuid4())


class RoleEnum(str, enum.Enum):
    ADMIN = "ADMIN"
    STAFF = "STAFF"


class QueueStatus(str, enum.Enum):
    WAITING = "WAITING"
    CALLED = "CALLED"
    IN_CONSULTATION = "IN_CONSULTATION"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    NO_SHOW = "NO_SHOW"


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    role = Column(Enum(RoleEnum), nullable=False)
    full_name = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Department(Base):
    __tablename__ = "departments"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    token_prefix = Column(String, nullable=False, default="A")
    is_active = Column(Boolean, default=True)

    doctors = relationship("Doctor", back_populates="department")


class Doctor(Base):
    __tablename__ = "doctors"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    department_id = Column(Integer, ForeignKey("departments.id"), nullable=False)
    is_available = Column(Boolean, default=True)  # paused/resumed by staff

    department = relationship("Department", back_populates="doctors")


class QueueEntry(Base):
    """The live, database-backed queue (NOT localStorage)."""
    __tablename__ = "queue_entries"
    id = Column(Integer, primary_key=True)
    public_token = Column(String, default=gen_uuid, unique=True, index=True)  # for QR / status lookups
    token_number = Column(String, nullable=False)  # e.g. "A-127", shown publicly
    department_id = Column(Integer, ForeignKey("departments.id"), nullable=False)
    doctor_id = Column(Integer, ForeignKey("doctors.id"), nullable=True)

    priority = Column(Boolean, default=False)       # hospital-policy priority patient
    emergency = Column(Boolean, default=False)       # emergency walk-in

    status = Column(Enum(QueueStatus), default=QueueStatus.WAITING, nullable=False)

    registration_time = Column(DateTime, default=datetime.utcnow, nullable=False)
    called_time = Column(DateTime, nullable=True)
    consultation_start = Column(DateTime, nullable=True)
    consultation_end = Column(DateTime, nullable=True)

    # Latest cached prediction (denormalized for fast reads; source of truth is Prediction table)
    estimated_wait_minutes = Column(Float, nullable=True)
    estimated_consultation_time = Column(DateTime, nullable=True)
    prediction_status = Column(String, default="PENDING")  # PENDING/OK/INSUFFICIENT_DATA

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    department = relationship("Department")
    doctor = relationship("Doctor")
    predictions = relationship("Prediction", back_populates="queue_entry")


class PriorityEvent(Base):
    __tablename__ = "priority_events"
    id = Column(Integer, primary_key=True)
    queue_entry_id = Column(Integer, ForeignKey("queue_entries.id"), nullable=False)
    reason = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id = Column(Integer, primary_key=True)
    version_name = Column(String, unique=True, nullable=False)  # "model_v1", "model_v2"...
    model_type = Column(String, nullable=False)                  # e.g. "HistGradientBoostingRegressor"
    file_path = Column(String, nullable=False)
    features_used = Column(Text, nullable=False)                 # JSON list
    training_date = Column(DateTime, default=datetime.utcnow)
    dataset_size = Column(Integer, nullable=False)
    mae = Column(Float, nullable=False)
    rmse = Column(Float, nullable=False)
    r2 = Column(Float, nullable=True)
    is_active = Column(Boolean, default=False)


class TrainingRun(Base):
    __tablename__ = "training_runs"
    id = Column(Integer, primary_key=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)
    status = Column(String, default="RUNNING")  # RUNNING/SUCCESS/FAILED
    data_quality_report = Column(Text, nullable=True)  # JSON
    model_version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=True)
    log = Column(Text, nullable=True)


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True)
    queue_entry_id = Column(Integer, ForeignKey("queue_entries.id"), nullable=False)
    generated_at = Column(DateTime, default=datetime.utcnow)
    predicted_wait_minutes = Column(Float, nullable=True)
    predicted_consultation_time = Column(DateTime, nullable=True)
    model_version = Column(String, nullable=True)   # "model_v2" or "INSUFFICIENT_DATA"
    queue_state_snapshot = Column(Text, nullable=True)  # JSON of the features used
    reason = Column(String, nullable=True)  # what triggered this re-forecast
    prediction_status = Column(String, default="OK")  # OK / INSUFFICIENT_DATA

    # filled in once the patient's consultation actually happens, for error tracking
    actual_wait_minutes = Column(Float, nullable=True)
    error_minutes = Column(Float, nullable=True)

    queue_entry = relationship("QueueEntry", back_populates="predictions")


class EventLog(Base):
    """Append-only queue event log (section 30) — also useful for future ML features."""
    __tablename__ = "event_log"
    id = Column(Integer, primary_key=True)
    queue_entry_id = Column(Integer, ForeignKey("queue_entries.id"), nullable=True)
    event_type = Column(String, nullable=False)  # PATIENT_REGISTERED, PRIORITY_PATIENT_ADDED, ...
    detail = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(String, nullable=False)
    detail = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class HistoricalRecord(Base):
    """
    Stores validated historical OPD records imported from CSV/Excel and
    completed live queue entries. This is the ML training dataset table.
    """
    __tablename__ = "historical_records"
    id = Column(Integer, primary_key=True)
    source = Column(String, default="IMPORT")  # IMPORT / LIVE / SYNTHETIC_DEV
    department = Column(String, nullable=False)
    doctor = Column(String, nullable=True)
    token_number = Column(String, nullable=True)

    registration_time = Column(DateTime, nullable=False)
    consultation_start = Column(DateTime, nullable=True)
    consultation_end = Column(DateTime, nullable=True)

    priority = Column(Boolean, default=False)
    emergency = Column(Boolean, default=False)

    patients_ahead_at_registration = Column(Integer, nullable=True)
    priority_patients_ahead_at_registration = Column(Integer, nullable=True)

    actual_wait_minutes = Column(Float, nullable=True)   # target variable
    consultation_duration_minutes = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
