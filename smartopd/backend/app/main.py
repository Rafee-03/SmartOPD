import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import models, security
from .database import Base, engine, SessionLocal
from .routers import auth, patients, staff, admin

app = FastAPI(title="SmartOPD AI", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your frontend's real origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(patients.router)
app.include_router(staff.router)
app.include_router(admin.router)


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    # Seed a default admin account ONLY if none exists yet, so a fresh
    # deployment is usable immediately. CHANGE THIS PASSWORD after first login.
    db = SessionLocal()
    try:
        if not db.query(models.User).filter(models.User.role == models.RoleEnum.ADMIN).first():
            default_password = os.getenv("DEFAULT_ADMIN_PASSWORD", "ChangeMe123!")
            db.add(models.User(
                username="admin",
                password_hash=security.hash_password(default_password),
                role=models.RoleEnum.ADMIN,
                full_name="Default Admin",
            ))
            db.commit()
    finally:
        db.close()


@app.get("/health")
def health():
    return {"status": "ok"}
