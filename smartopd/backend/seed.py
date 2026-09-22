"""
Run once after first deploy to create starter departments/doctors:
    python seed.py
Edit the lists below to match your actual hospital's departments first.
"""
from app.database import SessionLocal, Base, engine
from app import models

DEPARTMENTS = [
    {"name": "General Medicine", "token_prefix": "A", "doctors": ["Dr. Rao", "Dr. Iyer"]},
    {"name": "Pediatrics", "token_prefix": "B", "doctors": ["Dr. Sharma"]},
    {"name": "Orthopedics", "token_prefix": "C", "doctors": ["Dr. Khan"]},
]

Base.metadata.create_all(bind=engine)
db = SessionLocal()
try:
    for d in DEPARTMENTS:
        existing = db.query(models.Department).filter(models.Department.name == d["name"]).first()
        if existing:
            dept = existing
        else:
            dept = models.Department(name=d["name"], token_prefix=d["token_prefix"])
            db.add(dept)
            db.commit()
            db.refresh(dept)
        for doc_name in d["doctors"]:
            if not db.query(models.Doctor).filter(models.Doctor.name == doc_name, models.Doctor.department_id == dept.id).first():
                db.add(models.Doctor(name=doc_name, department_id=dept.id))
    db.commit()
    print("Seed complete.")
finally:
    db.close()
