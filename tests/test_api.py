"""Exercise the actual local API, simulator, checkpoint and durable experiments."""
import hashlib
import time
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import src.api as backend


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "STORE", tmp_path)
    with TestClient(backend.app) as connection:
        yield connection
    for job in backend.jobs.values():
        job["future"].result(timeout=30)


def finished(client, key):
    for _ in range(300):
        result = client.get(f"/api/experiments/{key}").json()
        if result["status"] != "running":
            return result
        time.sleep(.02)
    pytest.fail("Local experiment did not finish")


def test_recorded_data_real_inference_and_state_controls(client):
    path = backend.ROOT / "models/ppo_oncoevo.zip"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    assert client.get("/api/status").json()["model"]["available"]
    assert len(client.get("/api/patients").json()) == 71
    assert client.get("/api/patients/1").json()["observations"][0]["psa_ng_ml"] == 29
    assert client.get("/api/patients/9999").status_code == 404
    original = client.post("/api/simulations", json={"patient_id": None, "policy": "ppo"}).json()
    key = original["id"]
    url = f"/api/simulations/{key}/command"
    branch = client.post(f"/api/simulations/{key}/branches").json()
    assert branch["source_day"] == 0
    assert branch["outcomes"][0]["frame"]["psa"] != branch["outcomes"][1]["frame"]["psa"]
    step = client.post(url, json={"operation": "step"}).json()
    decision = step["trace"][0]
    assert decision["policy_info"]["sha256"] == before
    assert sum(decision["policy_info"]["action_probabilities"]) == pytest.approx(1)
    assert decision["reward"] == pytest.approx(sum(decision["reward_components"].values()))
    client.post(url, json={"operation": "pause"})
    assert client.post(url, json={"operation": "step"}).status_code == 409
    client.post(url, json={"operation": "reset"})
    assert client.post(url, json={"operation": "step"}).json()["frames"] == step["frames"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert client.post(url, json={"operation": "step", "action": 2}).status_code == 422
    assert client.post("/api/simulations", json={"horizon_days": -1}).status_code == 422


def test_saved_comparison_exports_and_reproduction(client):
    spec = {"name": "API deterministic comparison", "scenario": {"patient_id": None, "horizon_days": 56},
            "policies": ["ppo", "continuous", "intermittent"], "seeds": [17], "sensitivity": False}
    response = client.post("/api/experiments", json=spec)
    assert response.status_code == 202, response.text
    key = response.json()["id"]
    result = finished(client, key)
    assert result["status"] == "completed", result
    assert len(result["runs"]) == 3
    assert len({r["frames"][0]["psa"] for r in result["runs"]}) == 1
    for run in result["runs"]:
        exposure = sum((b["day"] - a["day"]) * b["treatment"] for a,b in zip(run["frames"],run["frames"][1:]))
        assert exposure == run["metrics"]["exposure_days"]
    metrics = client.get(f"/api/experiments/{key}/metrics").json()
    assert metrics["policies"]["continuous"]["mean_exposure_fraction"] == 1
    for format in ("csv", "json", "report"):
        export = client.get(f"/api/experiments/{key}/export?format={format}")
        assert export.status_code == 200 and len(export.content) > 100
    replay = client.post(f"/api/experiments/{key}/reproduce", json={})
    assert replay.status_code == 202
    assert finished(client, replay.json()["id"])["runs"] == result["runs"]


def test_cancellation_and_unavailable_checkpoint(client, monkeypatch):
    def missing(*args):
        raise FileNotFoundError("checkpoint intentionally unavailable")
    monkeypatch.setattr(backend, "ppo_checkpoint", missing)
    assert not client.get("/api/status").json()["model"]["available"]
    assert client.post("/api/simulations", json={"policy": "ppo"}).status_code == 409
    assert client.post("/api/simulations", json={"policy": "continuous", "patient_id": None}).status_code == 200
    spec = {"name": "Cancellation", "scenario": {"patient_id": None, "horizon_days": 1820},
            "policies": ["continuous", "intermittent"], "seeds": [1,2,3,4,5], "sensitivity": True}
    key = client.post("/api/experiments", json=spec).json()["id"]
    client.post(f"/api/experiments/{key}/cancel", json={})
    assert finished(client, key)["status"] in ("cancelled", "completed")


def test_complete_sensitivity_and_export_identity(client):
    spec = {"name": "One-at-a-time assumptions", "scenario": {"patient_id": None, "horizon_days": 28},
            "policies": ["continuous"], "seeds": [17], "sensitivity": True}
    key = client.post("/api/experiments", json=spec).json()["id"]
    result = finished(client, key)
    assert result["status"] == "completed"
    assert len(result["runs"]) == 7
    assert sum(run["nominal"] for run in result["runs"]) == 1
    assert len({run["parameters"]["kill"] for run in result["runs"]}) == 3
    csv = client.get(f"/api/experiments/{key}/export?format=csv").text
    assert all(field in csv for field in ("checkpoint_sha256", "patient_id", "scenario_json", "parameters_json"))


def test_runtime_source_drift_requires_restart(client, monkeypatch):
    monkeypatch.setattr(backend, "scientific_fingerprint", lambda: "modified")
    response = client.post("/api/simulations", json={"patient_id": None, "policy": "continuous"})
    assert response.status_code == 409
    assert "Restart" in response.json()["detail"]


def test_concurrent_result_reads_and_atomic_writes(client):
    from concurrent.futures import ThreadPoolExecutor
    key = "a" * 32
    document = {"id": key, "status": "running", "runs": []}
    backend.persist(document)
    def reader():
        for _ in range(50):
            assert backend.stored(key)["id"] == key
    def writer():
        for i in range(50):
            backend.persist({**document, "progress": i})
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(reader) for _ in range(3)] + [executor.submit(writer)]
        for future in futures:
            future.result(timeout=10)
    assert backend.stored(key)["progress"] == 49
