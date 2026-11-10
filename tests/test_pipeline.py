import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from src.data import load_patients, split_patients, calibration_prefix
from src.tumor_model import Twin, advance, fit_prefix, historical_prediction
from src.environment import OncoEvoEnv
from src.evaluate import replay
from src.utils import ContinuousPolicy, IntermittentPolicy, load_config


def test_data_audit_and_split():
    patients, audit = load_patients()
    assert len(patients) == 71 and len(audit) == 3
    assert 200 not in patients
    assert all(f.day.is_monotonic_increasing for f in patients.values())
    splits = split_patients(patients)
    assert len(set(sum(splits.values(), []))) == len(patients)


def test_model_and_prefix():
    patients, _ = load_patients()
    prefix = calibration_prefix(patients[1])
    twin, fit = fit_prefix(prefix)
    assert fit["n_prefix"] < len(patients[1])
    assert np.all(np.isfinite(historical_prediction(prefix, twin)))
    assert np.all(advance(twin.initial_state(), 100, 1, twin) >= 0)


def test_environment_api_seed_and_safety():
    twin = Twin(10)
    env = OncoEvoEnv(twin, decision_days=28, horizon_days=56, safety_multiplier=0.1)
    check_env(env, skip_render_check=True)
    obs1,_ = env.reset(seed=7)
    step1=env.step(0)
    obs2,_ = env.reset(seed=7)
    step2=env.step(0)
    np.testing.assert_array_equal(obs1,obs2)
    np.testing.assert_array_equal(step1[0],step2[0])
    assert step1[4]["safety_override"] and step1[4]["applied_action"] == 1
    with pytest.raises(ValueError):env.step(2)


def test_baselines_and_replay():
    config=load_config()
    twin=Twin(10)
    trace,metrics=replay(twin,twin.initial_state(),config,ContinuousPolicy(),7)
    assert len(trace)>1 and 0<=metrics["exposure_fraction"]<=1
    policy=IntermittentPolicy()
    assert policy(None,{"psa_ng_ml":4,"reference_psa":10},1)==0
    assert policy(None,{"psa_ng_ml":10,"reference_psa":10},0)==1


def test_dashboard_import():
    import ast
    with open("app.py",encoding="utf-8") as handle:ast.parse(handle.read())
