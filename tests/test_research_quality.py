"""Research integrity gates without training or altering the legacy checkpoint."""
from copy import deepcopy
import numpy as np
import pytest
from src.data import load_patients, calibration_prefix, split_patients
from src.environment import OncoEvoEnv
from src.tumor_model import Twin, fit_prefix
from src.experiment import load_ppo, ExperimentRunner
from src.utils import load_config
from src.api import Scenario, new_runner


def test_future_observations_do_not_change_calibration_or_policy():
    patients, _ = load_patients()
    original = patients[1]
    prefix = calibration_prefix(original)
    altered = original.copy()
    altered.loc[len(prefix):, "psa_ng_ml"] = 100000
    altered.loc[len(prefix):, "treatment_on"] = 1 - altered.loc[len(prefix):, "treatment_on"]
    twin1, _ = fit_prefix(prefix)
    twin2, _ = fit_prefix(calibration_prefix(altered))
    assert twin1 == twin2
    policy = load_ppo()
    config = load_config()
    one = ExperimentRunner(twin1, twin1.initial_state(), config, policy, 17)
    two = ExperimentRunner(twin2, twin2.initial_state(), config, policy, 17)
    assert one.step() == two.step()


def test_absorbing_cost_is_opt_in_and_transition_unchanged():
    twin = Twin(10)
    legacy = OncoEvoEnv(twin, progression_multiplier=1.001)
    corrected = OncoEvoEnv(twin, progression_multiplier=1.001, absorbing_failure_cost=True)
    legacy.reset(seed=17)
    corrected.reset(seed=17)
    a, b = legacy.step(0), corrected.step(0)
    np.testing.assert_array_equal(a[0], b[0])
    assert a[2] and b[2]
    assert "failure_continuation" not in a[4]["reward_components"]
    assert b[4]["reward_components"]["failure_continuation"] < 0
    assert b[1] < a[1]


def test_three_biological_perturbations_and_ood_warning():
    base = Scenario(patient_id=None, policy="continuous")
    original, _ = new_runner(base)
    for field in ("growth_multiplier", "kill_multiplier", "resistance_multiplier"):
        changed = base.model_copy(update={field: 1.2})
        runner, fit = new_runner(changed)
        assert runner.initial_state.tolist() == original.initial_state.tolist()
        runner.step()
        comparison = deepcopy(original)
        comparison.step()
        assert not np.array_equal(runner.env.state, comparison.env.state)
        assert any("out-of-distribution" in warning for warning in fit["warnings"])
    _, fit = new_runner(Scenario(patient_id=None, policy="continuous", synthetic_psa=60, progression_multiplier=4))
    assert any("carrying-capacity" in warning for warning in fit["warnings"])


def test_train_ids_are_disjoint_from_evaluation_splits():
    import json
    from pathlib import Path
    patients, _ = load_patients()
    splits = split_patients(patients, load_config()["seed"])
    manifest = json.loads(Path("models/training_metadata.json").read_text())
    assert set(manifest["training_patient_ids"]).issubset(splits["train"])
    assert not set(manifest["training_patient_ids"]) & set(splits["validation"] + splits["test"])


def test_safety_action_cannot_be_configured_away():
    with pytest.raises(ValueError, match="force-ON"):
        OncoEvoEnv(Twin(10), valid_actions=[0])


def test_checkpoint_contract_and_preservation():
    import hashlib
    import json
    from pathlib import Path
    path = Path("models/ppo_oncoevo.zip")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "861a7ecae154cc145d04c0983772c3b32084cad20a752cdb178011e850f38d55"
    manifest = json.loads(Path("models/training_metadata.json").read_text())
    policy = load_ppo()
    assert policy.num_timesteps == manifest["training_timesteps"] == 4096
    assert policy.seed == manifest["seed"]
    assert policy.observation_space.dtype == np.float32
    assert policy.action_space.start == 0 and policy.action_space.n == 2


def test_training_output_never_replaces_existing_file(tmp_path):
    from src.train import train
    output = tmp_path / "protected.zip"
    output.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        train(output=str(output))
    assert output.read_bytes() == b"preserve"
