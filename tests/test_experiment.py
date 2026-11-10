import hashlib
from pathlib import Path
import numpy as np
import pytest
from src.experiment import ExperimentRunner, load_ppo, validate_policy
from src.tumor_model import Twin, advance
from src.environment import OncoEvoEnv
from src.utils import load_config, ContinuousPolicy, IntermittentPolicy


def test_real_checkpoint_and_deterministic_trace():
    path = Path("models/ppo_oncoevo.zip")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    policy = load_ppo(path)
    twin = Twin(10)
    runner = ExperimentRunner(twin, twin.initial_state(), load_config(), policy, 17)
    first = runner.step()
    assert first["selected_action"] in (0, 1)
    assert sum(first["policy_info"]["action_probabilities"]) == pytest.approx(1)
    assert first["reward"] == pytest.approx(sum(first["reward_components"].values()))
    runner.pause()
    assert len(runner.advance(3)) == 1
    runner.resume()
    runner.advance()
    assert runner.trace == runner.replay()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_action_effects_and_terminal_guard():
    twin = Twin(10)
    state = twin.initial_state()
    assert not np.allclose(advance(state, 28, 0, twin), advance(state, 28, 1, twin))
    env = OncoEvoEnv(twin, progression_multiplier=.01)
    env.reset(seed=1)
    assert env.step(1)[2]
    with pytest.raises(RuntimeError): env.step(1)
    env.reset(seed=1)
    env.step(1)
    with pytest.raises(ValueError): advance([np.nan, 1], 28, 0, twin)


def test_matched_baselines_and_missing_checkpoint():
    twin = Twin(10)
    config = load_config()
    policies = {"continuous": ContinuousPolicy(), "intermittent": IntermittentPolicy()}
    traces = ExperimentRunner.compare(twin, twin.initial_state(), config, policies, 17)
    assert traces["continuous"][0]["observed_state"] == traces["intermittent"][0]["observed_state"]
    assert all(r["executed_action"] == 1 for r in traces["continuous"])
    with pytest.raises(FileNotFoundError): load_ppo("models/absent.zip")
    model = load_ppo()
    model.observation_space.low[:] = -1
    with pytest.raises(ValueError): validate_policy(model)
