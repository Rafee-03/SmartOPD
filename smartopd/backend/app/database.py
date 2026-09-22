"""
Database connection setup.

Defaults to a local SQLite file so the whole system runs with zero setup
(useful for developing from a phone / free hosting with no DB server).
Set DATABASE_URL env var to a Postgres URL (e.g. from Neon/Supabase free
tier, or Render's free Postgres) to switch to Postgres in production —
no code changes required elsewhere.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./smartopd.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
