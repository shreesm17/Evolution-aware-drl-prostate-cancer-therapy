from src.utils import calibrated_twin, intermittent_policy
"""Train a local PPO policy on virtual patients calibrated from train IDs only."""
import argparse
import json
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from src.data import prepare, calibration_prefix
from src.tumor_model import fit_prefix, state_at_end
from src.environment import OncoEvoEnv
from src.utils import load_config, ensure_dirs
from src.experiment import OBSERVATION_SCHEMA
import hashlib


class PatientSampler(OncoEvoEnv):
    def __init__(self, twins, config, seed):
        self.twins = twins
        self.sample_rng = np.random.default_rng(seed)
        twin, state = twins[0]
        super().__init__(twin, state, config["decision_days"], config["horizon_days"],
                         config["progression_multiplier"], config["safety_multiplier"], config["observation_noise_sd"], config.get("reward_weights"), config.get("valid_actions", [0, 1]), config.get("absorbing_failure_cost",False), config.get("reward_gamma",.99))

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.sample_rng = np.random.default_rng(seed)
        self.twin, self.start_state = self.twins[int(self.sample_rng.integers(len(self.twins)))]
        return super().reset(seed=seed, options=options)


class CurveCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rows = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.rows.append({"step": self.num_timesteps, "episode_reward": float(info["episode"]["r"]),
                                  "episode_length": int(info["episode"]["l"])})
        return True


def train(config_path="config.yaml", timesteps=None, output="models/ppo_oncoevo.zip", absorbing_failure_cost=False):
    checkpoint = Path(output).with_suffix(".zip")
    if checkpoint.exists():
        raise FileExistsError("Existing checkpoint preserved. Move it and its metadata to an archive before training a replacement.")
    config = load_config(config_path)
    if absorbing_failure_cost:
        config["absorbing_failure_cost"] = True
    ensure_dirs()
    patients, splits, _ = prepare(seed=config["seed"])
    ids = splits["train"][:int(config["training_patients"])]
    twins = []
    for patient_id in ids:
        prefix = calibration_prefix(patients[patient_id], config["calibration_fraction"])
        twin, _ = calibrated_twin(prefix, config)
        twins.append((twin, state_at_end(prefix, twin)))
    env = Monitor(PatientSampler(twins, config, config["seed"]))
    model = PPO("MlpPolicy", env, seed=config["seed"], device=config.get("training_device", "cpu"), verbose=0,
                n_steps=256, batch_size=64, learning_rate=3e-4, gamma=config.get("reward_gamma",.99))
    callback = CurveCallback()
    steps = int(timesteps or config["training_timesteps"])
    model.learn(total_timesteps=steps, callback=callback)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(checkpoint))
    (Path("results") / (checkpoint.stem + "_training_curve.json")).write_text(json.dumps(callback.rows, indent=2))
    metadata = {"seed": config["seed"], "training_patient_ids": ids, "training_timesteps": int(model.num_timesteps),
                "device": str(model.device), "config": config, "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "observation_schema": OBSERVATION_SCHEMA, "splits": splits, "gamma": model.gamma}
    checkpoint.with_name("training_metadata.json" if checkpoint.stem == "ppo_oncoevo" else checkpoint.stem + "_metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--output", default="models/ppo_oncoevo.zip")
    parser.add_argument("--absorbing-failure-cost", action="store_true", help="Opt into discounted absorbing failure costs for NEW checkpoints only")
    args = parser.parse_args()
    print(json.dumps(train(args.config, args.timesteps, args.output, args.absorbing_failure_cost), indent=2))
