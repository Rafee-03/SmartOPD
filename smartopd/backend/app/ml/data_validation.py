"""
Validates an uploaded historical-OPD CSV and produces a data-quality
report. Does NOT silently delete bad rows — it flags them and returns
both the report and the cleaned dataframe so the admin can see exactly
what happened (section 4).
"""
import pandas as pd

REQUIRED_COLUMNS = [
    "department", "registration_time", "consultation_start",
]
OPTIONAL_COLUMNS = [
    "doctor", "token_number", "consultation_end", "priority", "emergency",
    "patients_ahead_at_registration", "priority_patients_ahead_at_registration",
]


def validate_dataframe(df: pd.DataFrame) -> dict:
    notes = []
    total = len(df)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        return {
            "total_records": total,
            "valid_records": 0,
            "missing_consultation_times": 0,
            "invalid_timestamps": 0,
            "duplicate_records": 0,
            "notes": [f"Missing required columns: {missing_cols}. Upload rejected."],
            "clean_df": None,
        }

    df = df.copy()

    # Parse timestamps; invalid ones become NaT (flagged, not dropped yet).
    # format="mixed" parses each value independently instead of locking to
    # the first row's exact format (real hospital exports often mix
    # "...:00" and "...:00.123456" style timestamps in the same column).
    for col in ["registration_time", "consultation_start", "consultation_end"]:
        if col in df.columns:
            try:
                df[col + "_parsed"] = pd.to_datetime(df[col], errors="coerce", format="mixed")
            except (TypeError, ValueError):
                df[col + "_parsed"] = pd.to_datetime(df[col], errors="coerce")

    invalid_timestamps = int(df["registration_time_parsed"].isna().sum())
    if "consultation_start_parsed" in df.columns:
        invalid_timestamps += int(df["consultation_start_parsed"].isna().sum())

    missing_consultation = 0
    if "consultation_start_parsed" in df.columns:
        missing_consultation = int(df["consultation_start_parsed"].isna().sum())

    dedup_cols = [c for c in ["department", "token_number", "registration_time"] if c in df.columns]
    duplicate_mask = df.duplicated(subset=dedup_cols, keep="first") if dedup_cols else pd.Series([False] * total)
    duplicates = int(duplicate_mask.sum())

    valid_mask = (
        df["registration_time_parsed"].notna()
        & df["consultation_start_parsed"].notna()
        & (~duplicate_mask)
    )
    valid_df = df[valid_mask].copy()

    # Compute the real target from actual observed timestamps (section 3),
    # never a manually invented formula.
    valid_df["actual_wait_minutes"] = (
        valid_df["consultation_start_parsed"] - valid_df["registration_time_parsed"]
    ).dt.total_seconds() / 60.0

    # Drop physically impossible waits (negative, or absurdly long e.g. >1 day)
    # — flagged as a note, not hidden.
    before = len(valid_df)
    valid_df = valid_df[(valid_df["actual_wait_minutes"] >= 0) & (valid_df["actual_wait_minutes"] <= 24 * 60)]
    impossible = before - len(valid_df)
    if impossible:
        notes.append(f"{impossible} records had a negative or >24h wait and were excluded as physically invalid.")

    if "consultation_end_parsed" in valid_df.columns:
        valid_df["consultation_duration_minutes"] = (
            valid_df["consultation_end_parsed"] - valid_df["consultation_start_parsed"]
        ).dt.total_seconds() / 60.0

    notes.append("Rows with unparseable timestamps or missing consultation_start were excluded, not silently deleted — see counts above.")

    return {
        "total_records": total,
        "valid_records": int(len(valid_df)),
        "missing_consultation_times": missing_consultation,
        "invalid_timestamps": invalid_timestamps,
        "duplicate_records": duplicates,
        "notes": notes,
        "clean_df": valid_df,
    }
