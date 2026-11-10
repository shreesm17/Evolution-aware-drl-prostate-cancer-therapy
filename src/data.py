"""Validated Bruchovsky intermittent-ADT series; no inferred clinical records."""
from pathlib import Path
import re
import json
import numpy as np
import pandas as pd

COLUMNS = ["patient_id", "day", "psa_ng_ml", "androgen_reported", "cycle", "treatment_on"]


def load_patients(raw_dir="data/raw"):
    paths = sorted(Path(raw_dir).glob("patient*.txt"))
    if not paths:
        raise FileNotFoundError(f"No patient*.txt files in {raw_dir}")
    patients, audit = {}, []
    for path in paths:
        name_id = int(re.fullmatch(r"patient(\d+)", path.stem).group(1))
        frame = pd.read_csv(path, header=None, names=COLUMNS, dtype=str).apply(lambda c: c.str.strip())
        if frame.shape[1] != 6 or frame.isna().any().any():
            raise ValueError(f"Missing or extra fields: {path}")
        # Corrections are restricted to literal source typos, with an audit trail.
        if path.name == "patient7.txt" and frame.loc[9, "treatment_on"] == "1.00s":
            frame.loc[9, "treatment_on"] = "1.00"
            audit.append({"file": path.name, "row": 10, "field": "treatment_on", "correction": "1.00s -> 1.00"})
        if path.name == "patient17.txt" and frame.loc[1, "patient_id"] == "00":
            frame.loc[1, "patient_id"] = "17.00"
            audit.append({"file": path.name, "row": 2, "field": "patient_id", "correction": "00 -> 17.00"})
        for column in COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        embedded = set(frame.patient_id.astype(int))
        if len(embedded) != 1:
            raise ValueError(f"Mixed patient identities: {path}")
        patient_id = embedded.pop()
        if patient_id != name_id:
            if path.name == "patient200.txt" and patient_id == 30:
                audit.append({"file": path.name, "field": "patient_id", "correction": "Excluded: filename 200, embedded ID 30"})
                continue
            raise ValueError(f"Filename/identity mismatch: {path}")
        if patient_id in patients:
            raise ValueError(f"Duplicate patient {patient_id}")
        if (frame.day < 0).any() or not frame.day.is_monotonic_increasing or frame.day.duplicated().any():
            raise ValueError(f"Invalid timestamps: {path}")
        if (frame.psa_ng_ml < 0).any() or (frame.androgen_reported < 0).any():
            raise ValueError(f"Negative biomarker: {path}")
        if not set(frame.treatment_on).issubset({0, 1}):
            raise ValueError(f"Invalid treatment status: {path}")
        if not np.all(frame.cycle.to_numpy() == frame.cycle.astype(int).to_numpy()):
            raise ValueError(f"Invalid cycle: {path}")
        frame.patient_id = frame.patient_id.astype(int)
        frame.treatment_on = frame.treatment_on.astype(int)
        frame.cycle = frame.cycle.astype(int)
        patients[patient_id] = frame.reset_index(drop=True)
    if not patients:
        raise ValueError("No valid patient series")
    return patients, audit


def split_patients(patients, seed=42, fractions=(0.7, 0.15, 0.15)):
    if not np.isclose(sum(fractions), 1):
        raise ValueError("Split fractions must sum to one")
    ids = np.array(sorted(patients), dtype=int)
    np.random.default_rng(seed).shuffle(ids)
    a, b = int(len(ids) * fractions[0]), int(len(ids) * sum(fractions[:2]))
    return {"train": ids[:a].tolist(), "validation": ids[a:b].tolist(), "test": ids[b:].tolist()}


def calibration_prefix(frame, fraction=0.3, minimum=5):
    """Only an early historical prefix can condition the virtual twin."""
    n = min(len(frame), max(minimum, int(np.ceil(len(frame) * fraction))))
    return frame.iloc[:n].copy()


def prepare(raw_dir="data/raw", output_dir="data/processed", seed=42):
    patients, audit = load_patients(raw_dir)
    splits = split_patients(patients, seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(patients.values(), ignore_index=True).to_csv(out / "observations.csv", index=False)
    (out / "splits.json").write_text(json.dumps(splits, indent=2))
    (out / "audit.json").write_text(json.dumps(audit, indent=2))
    return patients, splits, audit
