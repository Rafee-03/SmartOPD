from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


class LoginRequest(BaseModel):
    username: str
    password: str


class DepartmentOut(BaseModel):
    id: int
    name: str
    token_prefix: str

    class Config:
        from_attributes = True


class RegisterPatientRequest(BaseModel):
    department_id: int
    doctor_id: Optional[int] = None
    priority: bool = False
    emergency: bool = False
    reason: Optional[str] = None  # staff note, e.g. why emergency


class QueueEntryOut(BaseModel):
    id: int
    public_token: str
    token_number: str
    department_id: int
    doctor_id: Optional[int]
    priority: bool
    emergency: bool
    status: str
    registration_time: datetime
    estimated_wait_minutes: Optional[float]
    estimated_consultation_time: Optional[datetime]
    prediction_status: str

    class Config:
        from_attributes = True


class PatientStatusOut(BaseModel):
    token_number: str
    status: str
    estimated_wait_minutes: Optional[float]
    estimated_consultation_time: Optional[datetime]
    patients_ahead: int
    prediction_status: str
    prediction_message: str
    last_updated: datetime
    previous_estimated_wait_minutes: Optional[float] = None


class PredictionOut(BaseModel):
    id: int
    generated_at: datetime
    predicted_wait_minutes: Optional[float]
    model_version: Optional[str]
    reason: Optional[str]
    prediction_status: str

    class Config:
        from_attributes = True


class ModelVersionOut(BaseModel):
    id: int
    version_name: str
    model_type: str
    training_date: datetime
    dataset_size: int
    mae: float
    rmse: float
    r2: Optional[float]
    is_active: bool
    features_used: str

    class Config:
        from_attributes = True


class DataQualityReport(BaseModel):
    total_records: int
    valid_records: int
    missing_consultation_times: int
    invalid_timestamps: int
    duplicate_records: int
    notes: List[str] = []
