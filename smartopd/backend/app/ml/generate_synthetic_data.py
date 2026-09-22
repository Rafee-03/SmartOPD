"""
*** SYNTHETIC / DEVELOPMENT DATA ONLY ***

This generates a fake OPD dataset so you can test the training pipeline
and prediction engine end-to-end BEFORE your hospital has real historical
data. Every record produced here is tagged source="SYNTHETIC_DEV" in the
database and the admin dashboard displays that tag prominently.

Do NOT use this to make real predictions for real patients. Once you
have real historical records (imported CSV or completed live queue
entries), retrain the model on real data and deactivate any model that
was trained on synthetic data.

Run:  python -m app.ml.generate_synthetic_data --rows 3000 --out synthetic_opd_data.csv
"""
import argparse
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

DEPARTMENTS = {
    "General Medicine": ["Dr. Rao", "Dr. Iyer"],
    "Pediatrics": ["Dr. Sharma"],
    "Orthopedics": ["Dr. Khan", "Dr. Verma"],
    "ENT": ["Dr. Das"],
}

# Base consultation-time behavior per doctor (minutes) — used ONLY to
# generate believable synthetic data, never used inside the actual model.
DOCTOR_BASE_MINUTES = {
    "Dr. Rao": 8, "Dr. Iyer": 10, "Dr. Sharma": 6,
    "Dr. Khan": 12, "Dr. Verma": 9, "Dr. Das": 7,
}


def generate(rows: int, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    np.random.seed(seed)

    records = []
    start_date = datetime(2025, 1, 6)  # a Monday
    days = rows // 40 + 1

    for day_i in range(days):
        day = start_date + timedelta(days=day_i)
        if day.weekday() == 6:  # skip Sundays (OPD closed)
            continue
        opd_start = day.replace(hour=9, minute=0, second=0, microsecond=0)

        # per-doctor running queue state for the day
        queue_state = {dept: {"count": 0, "priority_count": 0} for dept in DEPARTMENTS}

        n_today = rng.randint(25, 55)
        t = opd_start
        for _ in range(n_today):
            dept = rng.choice(list(DEPARTMENTS.keys()))
            doctor = rng.choice(DEPARTMENTS[dept])
            t = t + timedelta(minutes=rng.randint(1, 6))
            priority = rng.random() < 0.08
            emergency = rng.random() < 0.04
            if emergency:
                priority = True

            patients_ahead = queue_state[dept]["count"]
            priority_ahead = queue_state[dept]["priority_count"]

            base = DOCTOR_BASE_MINUTES[doctor]
            # Synthetic "true" generative process (nonlinear + noisy) —
            # this stands in for the unknown real-world relationship the
            # ML model is supposed to learn. Real deployments must replace
            # this entire file's output with actual hospital records.
            effective_position = max(patients_ahead - (2 if priority else 0), 0)
            wait = (
                effective_position * base * np.random.uniform(0.8, 1.3)
                + priority_ahead * base * 0.5
                + np.random.normal(0, base * 0.4)
            )
            wait = max(wait, 1.0)

            consultation_start = t + timedelta(minutes=wait)
            duration = max(np.random.normal(base, base * 0.3), 2.0)
            consultation_end = consultation_start + timedelta(minutes=duration)

            records.append({
                "department": dept,
                "doctor": doctor,
                "token_number": f"{dept[:1]}-{len(records)+1}",
                "registration_time": t.isoformat(),
                "consultation_start": consultation_start.isoformat(),
                "consultation_end": consultation_end.isoformat(),
                "priority": priority,
                "emergency": emergency,
                "patients_ahead_at_registration": patients_ahead,
                "priority_patients_ahead_at_registration": priority_ahead,
            })

            queue_state[dept]["count"] += 1
            if priority:
                queue_state[dept]["priority_count"] += 1

    df = pd.DataFrame(records)
    df["_source"] = "SYNTHETIC_DEV"
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument("--out", type=str, default="synthetic_opd_data.csv")
    args = parser.parse_args()
    df = generate(args.rows)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} SYNTHETIC_DEV rows to {args.out}")
    print("Reminder: this is fake data for pipeline testing only.")
