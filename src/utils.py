"""Shared configuration and baseline policies."""
from pathlib import Path
import yaml


def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dirs():
    for name in ("data/processed", "models", "results"):
        Path(name).mkdir(parents=True, exist_ok=True)


class ContinuousPolicy:
    def __call__(self, observation, info, previous_action):
        return 1


class IntermittentPolicy:
    """PSA 50%-of-baseline stop and baseline restart (Zhang et al. 2022)."""
    def __init__(self, stop_ratio=0.5, restart_ratio=1.0):
        if not 0 < stop_ratio < restart_ratio:
            raise ValueError("Stop threshold must be below restart threshold")
        self.stop_ratio, self.restart_ratio = stop_ratio, restart_ratio

    def __call__(self, observation, info, previous_action):
        ratio = info["psa_ng_ml"] / info["reference_psa"]
        if previous_action == 1 and ratio <= self.stop_ratio:
            return 0
        if previous_action == 0 and ratio >= self.restart_ratio:
            return 1
        return previous_action


def calibrated_twin(prefix, config):
    from src.tumor_model import Twin, fit_prefix
    return fit_prefix(prefix, Twin(initial_psa=float(prefix.psa_ng_ml.iloc[0]), **config.get("simulator", {})))


def intermittent_policy(config):
    return IntermittentPolicy(config.get("intermittent_stop_ratio", .5), config.get("intermittent_restart_ratio", 1.0))
