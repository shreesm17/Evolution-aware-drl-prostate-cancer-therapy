"""Local research API. All outcomes beyond recorded histories are simulations."""
from concurrent.futures import ThreadPoolExecutor
from copy import copy, deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from threading import RLock, Event
from typing import Literal, Any
from uuid import uuid4
import csv
import io
import json
import hashlib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict
from src.data import load_patients, calibration_prefix, split_patients
from src.experiment import ExperimentRunner, load_ppo, OBSERVATION_SCHEMA, policy_diagnostics
from src.tumor_model import Twin, state_at_end, historical_prediction, psa
from src.utils import load_config, calibrated_twin, ContinuousPolicy, intermittent_policy

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "results" / "studio"
Policy = Literal["ppo", "continuous", "intermittent"]
app = FastAPI(title="OncoEvo Research Studio", version="1.0")
pool = ThreadPoolExecutor(max_workers=2)
sessions: dict[str, dict] = {}
jobs: dict[str, dict] = {}
guard = RLock()


@app.exception_handler(ValueError)
async def invalid_configuration(request, exc):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(RuntimeError)
async def invalid_transition(request, exc):
    return JSONResponse(status_code=409, content={"detail": str(exc)})


def scientific_fingerprint():
    """Refuse exact reproduction if code, calibration inputs or configuration changed."""
    paths = [ROOT / "config.yaml", *sorted((ROOT / "data/raw").glob("patient*.txt"))]
    paths += [ROOT / "src" / name for name in ("tumor_model.py", "data.py", "environment.py", "experiment.py", "utils.py", "api.py")]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


RUNTIME_FINGERPRINT = scientific_fingerprint()


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Scenario(Schema):
    patient_id: int | None = 1  # None explicitly selects a synthetic virtual patient.
    policy: Policy = "ppo"
    seed: int = Field(default=42, ge=0, le=2147483647)
    horizon_days: int = Field(default=728, ge=28, le=1820)
    decision_days: int = Field(default=28, ge=7, le=90)
    growth_multiplier: float = Field(default=1, ge=.5, le=1.5)
    resistance_multiplier: float = Field(default=1, ge=.5, le=2)
    kill_multiplier: float = Field(default=1, ge=.5, le=1.5)
    safety_multiplier: float = Field(default=1.5, ge=1, le=3)
    progression_multiplier: float = Field(default=2, ge=1.1, le=4)
    synthetic_psa: float = Field(default=10, ge=.1, le=60)


class Command(Schema):
    operation: Literal["step", "run", "pause", "resume", "reset", "replay"]
    steps: int = Field(default=1, ge=1, le=260)
    action: Literal[0, 1] | None = None


class Snapshot(Schema):
    day: int
    psa: float
    observed_psa: float
    sensitive: float
    resistant: float
    treatment: int
    reward: float
    exposure_days: float


class DecisionRecord(Schema):
    day: int
    observed_state: list[float]
    selected_action: int
    executed_action: int
    safety_override: bool
    reward: float
    reward_components: dict[str, float]
    next_state: list[float]
    policy_info: dict[str, Any]
    terminated: bool
    truncated: bool
    simulated: bool = True


class Simulation(Schema):
    id: str
    scenario: Scenario
    paused: bool
    done: bool
    frames: list[Snapshot]
    trace: list[DecisionRecord]
    calibration: dict[str, Any]


class PatientSummary(Schema):
    id: int
    measurements: int
    followup_days: float
    initial_psa: float
    latest_psa: float
    on_visits: int
    split: str
    psa_trend: list[float]


class PatientObservation(Schema):
    patient_id: int
    day: float
    psa_ng_ml: float
    androgen_reported: float
    cycle: int
    treatment_on: int


class ForecastPoint(Schema):
    day: float
    psa: float


class PatientHistory(Schema):
    id: int
    observations: list[PatientObservation]
    calibration: dict[str, Any]
    forecast: list[ForecastPoint]
    forecast_log_rmse: float | None


class ModelStatus(Schema):
    available: bool
    checkpoint: str | None = None
    sha256: str | None = None
    observation_schema: list[str] | None = None
    preprocessing: str | None = None
    action_mapping: dict[int, str] | None = None
    error: str | None = None
    training_timesteps: int | None = None
    reward_warning: str | None = None


class EngineStatus(Schema):
    model: ModelStatus
    cohort_size: int
    observations: int
    audit: list[dict[str, Any]]
    observation_schema: list[str]
    notice: str


class ExperimentSpec(Schema):
    name: str = Field(default="Untitled experiment", min_length=1, max_length=80)
    scenario: Scenario = Field(default_factory=Scenario)
    policies: list[Policy] = Field(default=["ppo", "continuous", "intermittent"], min_length=1, max_length=3)
    seeds: list[int] = Field(default=[42], min_length=1, max_length=5)
    sensitivity: bool = True


@lru_cache(maxsize=1)
def cohort():
    try:
        return load_patients(ROOT / "data/raw")
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(503, str(exc)) from exc


def configuration(scenario):
    if scientific_fingerprint() != RUNTIME_FINGERPRINT:
        raise HTTPException(409, "Scientific files changed while the API was running. Restart the local backend before starting or reproducing an experiment.")
    config = load_config(ROOT / "config.yaml")
    for key in ("decision_days", "horizon_days", "safety_multiplier", "progression_multiplier"):
        config[key] = getattr(scenario, key)
    return config


@lru_cache(maxsize=1)
def ppo_checkpoint(digest):
    return load_ppo(ROOT / "models/ppo_oncoevo.zip")


def policy_for(name, config):
    if name == "continuous":
        return ContinuousPolicy()
    if name == "intermittent":
        return intermittent_policy(config)
    path = ROOT / "models/ppo_oncoevo.zip"
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return ppo_checkpoint(digest)
    except Exception as exc:
        raise HTTPException(409, f"PPO unavailable or incompatible: {exc}") from exc


@lru_cache(maxsize=128)
def patient_twin(patient_id):
    patients, _ = cohort()
    if patient_id not in patients:
        raise HTTPException(404, "Patient not found")
    config = load_config(ROOT / "config.yaml")
    prefix = calibration_prefix(patients[patient_id], config["calibration_fraction"])
    twin, fit = calibrated_twin(prefix, config)
    return twin, state_at_end(prefix, twin), {**fit, "cutoff_day": float(prefix.day.iloc[-1]), "kind": "calibrated estimate"}


def new_runner(scenario):
    config = configuration(scenario)
    if scenario.patient_id is None:
        twin = Twin(scenario.synthetic_psa, **config.get("simulator", {}))
        state, fit = twin.initial_state(), {"kind": "explicitly synthetic", "parameters": asdict(twin)}
    else:
        twin, state, fit = patient_twin(scenario.patient_id)
    twin = replace(twin, growth=twin.growth * scenario.growth_multiplier, kill=twin.kill * scenario.kill_multiplier,
                   resistant_growth=twin.resistant_growth * scenario.resistance_multiplier)
    fit = deepcopy(fit)
    fit["scenario_parameters"] = asdict(twin)
    fit["warnings"] = ["Instantaneous drug effect; no androgen recovery or PSA clearance dynamics.",
                       "Simulated populations are unobserved PSA-scaled burdens, not measured cells."]
    if twin.competition >= 1 and sum(state) <= twin.carrying_capacity and psa(state,twin) * scenario.progression_multiplier > twin.carrying_capacity + twin.floor_psa:
        fit["warnings"].append("Progression threshold exceeds the model carrying-capacity ceiling; apparent non-progression is structurally uninformative.")
    if scenario.horizon_days != 728 or scenario.decision_days != 28 or scenario.growth_multiplier != 1 or scenario.resistance_multiplier != 1 or scenario.kill_multiplier != 1:
        fit["warnings"].append("Scenario differs from the legacy checkpoint training settings; out-of-distribution policy behavior is unvalidated.")
    return ExperimentRunner(twin, state, config, policy_for(scenario.policy, config), scenario.seed), fit


def frame(runner, exposure=0):
    return {"day": runner.env.elapsed, "psa": psa(runner.env.state, runner.twin),
            "observed_psa": runner.env.observed_psa, "sensitive": float(runner.env.state[0]),
            "resistant": float(runner.env.state[1]), "treatment": runner.env.last_action,
            "reward": runner.trace[-1]["reward"] if runner.trace else 0., "exposure_days": exposure}


def step_runner(runner, frames, action=None):
    # A manual intervention is explicitly recorded, without editing PPO weights.
    previous_policy = runner.policy
    if action is not None:
        runner.policy = lambda observation, info, previous: action
    try:
        record = runner.step()
        if action is not None:
            record["policy_info"] = {"name": "manual intervention", "action": action}
            runner.trace[-1] = record
    finally:
        runner.policy = previous_policy
    frames.append(frame(runner, frames[-1]["exposure_days"] + runner.info["days"] * runner.previous_action))


def snapshot(key, entry):
    runner = entry["runner"]
    return {"id": key, "scenario": entry["scenario"], "paused": runner.paused, "done": runner.done,
            "frames": entry["frames"], "trace": runner.trace, "calibration": entry["calibration"]}


@app.get("/api/status", response_model=EngineStatus)
def status() -> dict[str, Any]:
    try:
        policy = policy_for("ppo", load_config(ROOT / "config.yaml"))
        model = {"available": True, **policy.oncoevo_provenance}
    except HTTPException as exc:
        model = {"available": False, "error": exc.detail}
    patients, audit = cohort()
    return {"model": model, "cohort_size": len(patients), "observations": sum(map(len, patients.values())),
            "audit": audit, "observation_schema": OBSERVATION_SCHEMA,
            "notice": "Computational research prototype. Counterfactual simulations are not treatment recommendations."}


@app.get("/api/patients", response_model=list[PatientSummary])
def patients() -> list[dict[str, Any]]:
    patients, _ = cohort()
    splits = split_patients(patients, load_config(ROOT / "config.yaml")["seed"])
    return [{"id": pid, "measurements": len(f), "followup_days": float(f.day.max()),
             "initial_psa": float(f.psa_ng_ml.iloc[0]), "latest_psa": float(f.psa_ng_ml.iloc[-1]),
             "on_visits": int(f.treatment_on.sum()), "split": next(k for k,v in splits.items() if pid in v),
             "psa_trend": f.psa_ng_ml.tolist()} for pid, f in patients.items()]


@app.get("/api/patients/{patient_id}", response_model=PatientHistory)
def history(patient_id: int) -> dict[str, Any]:
    twin, _, fit = patient_twin(patient_id)
    f = cohort()[0][patient_id]
    prediction = historical_prediction(f, twin)
    return {"id": patient_id, "observations": f.to_dict("records"), "calibration": fit,
            "forecast": [{"day": float(day), "psa": float(value)} for day,value in zip(f.day, prediction)],
            "forecast_log_rmse": float(np.sqrt(np.mean((np.log1p(prediction[len_prediction:]) - np.log1p(f.psa_ng_ml.to_numpy()[len_prediction:]))**2)))
            if (len_prediction := fit["n_prefix"]) < len(f) else None}


@app.post("/api/simulations", response_model=Simulation)
def create_simulation(scenario: Scenario):
    runner, fit = new_runner(scenario)
    key = uuid4().hex
    with guard:
        if len(sessions) >= 64:
            raise HTTPException(409, "Session capacity reached; restart the local API to clear sessions")
        sessions[key] = {"runner": runner, "frames": [frame(runner)], "scenario": scenario.model_dump(), "calibration": fit, "lock": RLock()}
    return snapshot(key, sessions[key])


def get_session(key):
    if key not in sessions:
        raise HTTPException(404, "Simulation expired; reset the scenario")
    return sessions[key]


@app.delete("/api/simulations/{key}", status_code=204)
def release_simulation(key: str):
    entry = get_session(key)
    with entry["lock"], guard:
        sessions.pop(key, None)
    return Response(status_code=204)


@app.post("/api/simulations/{key}/command", response_model=Simulation)
def command(key: str, command: Command):
    entry = get_session(key)
    with entry["lock"]:
        runner = entry["runner"]
        if command.operation in ("reset", "replay"):
            runner.reset()
            entry["frames"] = [frame(runner)]
        if command.operation == "pause":
            runner.pause()
        elif command.operation == "resume":
            runner.resume()
        elif command.operation in ("step", "run", "replay"):
            if runner.paused or runner.done:
                raise HTTPException(409, "Simulation paused or finished; resume or reset it")
            count = 1 if command.operation == "step" else command.steps
            for _ in range(count):
                if runner.done:
                    break
                step_runner(runner, entry["frames"], command.action)
        return snapshot(key, entry)


@app.post("/api/simulations/{key}/branches")
def branches(key: str) -> dict[str, Any]:
    entry = get_session(key)
    with entry["lock"]:
        original = entry["runner"]
        if original.done:
            raise HTTPException(409, "Reset before branching a finished simulation")
        outcomes = []
        for action in (0, 1):
            runner = copy(original)
            runner.env, runner.trace, runner.info = deepcopy(original.env), deepcopy(original.trace), deepcopy(original.info)
            runner.paused = False
            frames = deepcopy(entry["frames"])
            step_runner(runner, frames, action)
            outcomes.append({"action": action, "frame": frames[-1], "decision": runner.trace[-1]})
        return {"source_day": original.env.elapsed, "outcomes": outcomes}


def stored(key):
    if len(key) != 32 or any(c not in "0123456789abcdef" for c in key):
        raise HTTPException(404, "Experiment not found")
    path = STORE / f"{key}.json"
    with guard:
        if not path.exists():
            raise HTTPException(404, "Experiment not found")
        return json.loads(path.read_text(encoding="utf-8"))


def persist(document):
    # Windows cannot replace a file while another request holds it open.
    # Use the same lock for reads and atomic writes; never expose partial JSON.
    with guard:
        STORE.mkdir(parents=True, exist_ok=True)
        path = STORE / f"{document['id']}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(document, allow_nan=False), encoding="utf-8")
        temporary.replace(path)


def run_job(key, spec, event):
    document = stored(key)
    try:
        config = spec.scenario.model_dump()
        # One-at-a-time perturbations avoid confounding growth, response and resistance.
        variants = [(1.,1.,1.)]
        if spec.sensitivity:
            variants += [(.8,1.,1.), (1.2,1.,1.), (1.,.8,1.), (1.,1.2,1.), (1.,1.,.5), (1.,1.,2.)]
        total = len(spec.seeds) * len(set(spec.policies)) * len(variants)
        for multiplier, kill_factor, resistance_factor in variants:
            for seed in spec.seeds:
                for policy in dict.fromkeys(spec.policies):
                    if event.is_set():
                        document["status"] = "cancelled"
                        persist(document)
                        return
                    scenario = Scenario(**{**config, "policy": policy, "seed": seed,
                        "growth_multiplier": min(1.5, max(.5, config["growth_multiplier"] * multiplier)),
                        "kill_multiplier": min(1.5, max(.5, config["kill_multiplier"] * kill_factor)),
                        "resistance_multiplier": min(2., max(.5, config["resistance_multiplier"] * resistance_factor))})
                    runner, fit = new_runner(scenario)
                    frames = [frame(runner)]
                    while not runner.done:
                        if event.is_set():
                            document["status"] = "cancelled"
                            persist(document)
                            return
                        step_runner(runner, frames)
                    actions = [t["executed_action"] for t in runner.trace]
                    document["runs"].append({"policy": policy, "seed": seed, "sensitivity_factor": multiplier,
                        "nominal": (multiplier,kill_factor,resistance_factor) == (1.,1.,1.),
                        "kill_factor": kill_factor, "resistance_factor": resistance_factor,
                        "scenario": scenario.model_dump(), "parameters": asdict(runner.twin),
                        "initial_state": runner.initial_state.tolist(),
                        "diagnostics": policy_diagnostics(runner.trace),
                        "frames": frames, "trace": runner.trace, "calibration": fit,
                        "metrics": {"exposure_days": frames[-1]["exposure_days"],
                            "exposure_fraction": frames[-1]["exposure_days"] / frames[-1]["day"],
                            "progression": bool(runner.info["progression"]), "followup_days": frames[-1]["day"],
                            "time_to_progression": frames[-1]["day"] if runner.info["progression"] else None,
                            "safety_overrides": runner.env.overrides,
                            "policy_switches": sum(a != b for a,b in zip(actions, actions[1:]))}})
                    document["progress"] = len(document["runs"]) / total
                    persist(document)
        document["status"] = "completed"
    except Exception as exc:
        document["status"], document["error"] = "failed", str(exc)
    persist(document)


@app.post("/api/experiments", status_code=202)
def create_experiment(spec: ExperimentSpec) -> dict[str, Any]:
    if any(seed < 0 or seed > 2147483647 for seed in spec.seeds):
        raise HTTPException(422, "Seeds must be between 0 and 2147483647")
    for policy in spec.policies:
        policy_for(policy, configuration(spec.scenario))
    if spec.scenario.patient_id is not None:
        patient_twin(spec.scenario.patient_id)
    with guard:
        if sum(not job["future"].done() for job in jobs.values()) >= 4:
            raise HTTPException(409, "Local queue full; wait or cancel an experiment")
        key = uuid4().hex
        document = {"id": key, "name": spec.name, "created": datetime.now(timezone.utc).isoformat(),
            "spec": spec.model_dump(), "status": "running", "progress": 0., "runs": [],
            "model": status()["model"], "scientific_fingerprint": scientific_fingerprint(),
            "environment_config": configuration(spec.scenario),
            "interpretation": "Hypothetical simulated outcomes; not observed patient outcomes"}
        persist(document)
        event = Event()
        jobs[key] = {"event": event, "future": pool.submit(run_job, key, spec, event)}
    return document


@app.get("/api/experiments")
def experiments() -> list[dict[str, Any]]:
    with guard:
        documents = [json.loads(p.read_text(encoding="utf-8")) for p in STORE.glob("*.json")]
    for document in documents:
        if document["status"] == "running" and document["id"] not in jobs:
            document["status"] = "interrupted"
    return [{k:v for k,v in d.items() if k != "runs"} for d in sorted(documents, key=lambda d:d["created"], reverse=True)]


@app.get("/api/experiments/{key}")
def experiment(key: str) -> dict[str, Any]:
    document = stored(key)
    if document["status"] == "running" and key not in jobs:
        document["status"] = "interrupted"
    elif document["status"] == "running" and jobs[key]["future"].done():
        document["status"] = "failed"
        document["error"] = "Worker stopped before finalizing its results; completed run data remains exportable."
    return document


@app.post("/api/experiments/{key}/cancel")
def cancel(key: str) -> dict[str, Any]:
    stored(key)
    if key in jobs:
        jobs[key]["event"].set()
    return experiment(key)


@app.post("/api/experiments/{key}/reproduce", status_code=202)
def reproduce(key: str) -> dict[str, Any]:
    document = stored(key)
    if document.get("scientific_fingerprint") != scientific_fingerprint():
        raise HTTPException(409, "Scientific code, data or configuration changed; exact reproduction is unavailable. Create a new experiment explicitly.")
    if "ppo" in document["spec"]["policies"] and document["model"].get("sha256") != status()["model"].get("sha256"):
        raise HTTPException(409, "Checkpoint changed; cannot reproduce this saved run exactly")
    return create_experiment(ExperimentSpec(**document["spec"]))


@app.get("/api/experiments/{key}/metrics")
def evaluation_metrics(key: str) -> dict[str, Any]:
    document = stored(key)
    groups = {}
    for run in document["runs"]:
        groups.setdefault(run["policy"], []).append(run["metrics"])
    return {"simulated": True, "interval_type": "empirical sensitivity min/max, not confidence intervals",
            "policies": {name: {"completed_runs": len(rows),
                "mean_exposure_fraction": float(np.mean([r["exposure_fraction"] for r in rows])),
                "exposure_range": [min(r["exposure_fraction"] for r in rows), max(r["exposure_fraction"] for r in rows)],
                "modeled_progression_fraction": float(np.mean([r["progression"] for r in rows])),
                "safety_overrides": sum(r["safety_overrides"] for r in rows)} for name,rows in groups.items()}}


@app.get("/api/experiments/{key}/export")
def export(key: str, format: Literal["json", "csv", "report"] = "json"):
    document = stored(key)
    if format == "json":
        body, mime, extension = json.dumps(document, indent=2), "application/json", "json"
    elif format == "csv":
        stream = io.StringIO()
        rows = [{"experiment_id": key, "patient_id": document["spec"]["scenario"]["patient_id"],
                 "checkpoint_sha256": document["model"].get("sha256"),
                 "scientific_fingerprint": document.get("scientific_fingerprint"),
                 "policy": run["policy"], "seed": run["seed"], "sensitivity_factor": run["sensitivity_factor"],
                 "kill_factor": run.get("kill_factor",1), "resistance_factor": run.get("resistance_factor",1),
                 "scenario_json": json.dumps(run.get("scenario", document["spec"]["scenario"])),
                 "parameters_json": json.dumps(run.get("parameters", {})), **run["metrics"]} for run in document["runs"]]
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        body, mime, extension = stream.getvalue(), "text/csv", "csv"
    else:
        lines = ["OncoEvo Research Studio", document["name"], document["created"], document["interpretation"],
                 "Configuration:", json.dumps(document["spec"], indent=2),
                 "Checkpoint and scientific source identity:", json.dumps(document["model"], indent=2),
                 document.get("scientific_fingerprint", "legacy record"), "Calculated simulation metrics:"]
        lines.extend(f"{r['policy']} | seed {r['seed']} | sensitivity {r['sensitivity_factor']}: {json.dumps(r['metrics'])}" for r in document["runs"])
        lines.append("Limitations: latent populations are simulated PSA-scaled burdens, not measured cells. PSA mapping, resistance and competition are uncertain. PSA threshold progression is not clinical progression or survival. Sensitivity ranges are not confidence intervals. Historical treatment is not randomized counterfactual evidence.")
        body, mime, extension = "\n\n".join(lines), "text/plain", "txt"
    return Response(body, media_type=mime, headers={"Content-Disposition": f'attachment; filename="oncoevo-{key}.{extension}"'})
