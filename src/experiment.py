"""Stateful counterfactual experiments and explicit legacy checkpoint contract."""
from copy import deepcopy
import hashlib
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from src.environment import OncoEvoEnv

OBSERVATION_SCHEMA = ["log1p(observed PSA ng/mL)", "clip(log((current PSA+.1)/(previous PSA+.1)),-5,5)",
                      "previous executed action", "elapsed days / horizon days"]


def validate_policy(model):
    if (model.observation_space.shape != (4,) or getattr(model.action_space, "n", None) != 2
            or not np.all(model.observation_space.low == -10)
            or not np.all(model.observation_space.high == 10)):
        raise ValueError("Checkpoint incompatible with legacy four-feature ON/OFF contract")


def load_ppo(path="models/ppo_oncoevo.zip"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"PPO checkpoint unavailable: {path}")
    model = PPO.load(str(path), device="cpu")
    validate_policy(model)
    model.oncoevo_provenance = {"checkpoint": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "observation_schema": OBSERVATION_SCHEMA,
        "preprocessing": "float32; no VecNormalize; default SB3 MlpPolicy feature flattening",
        "action_mapping": {0: "OFF", 1: "ON"}, "training_timesteps": int(model.num_timesteps),
        "reward_warning": "Legacy all-negative rewards and early termination can favor shorter failed episodes. Episode return is not a disease-control ranking."}
    return model


def policy_diagnostics(trace, gamma=.99):
    """Observable behavior only; constant actions do not prove reward exploitation."""
    selected = [r["selected_action"] for r in trace]
    executed = [r["executed_action"] for r in trace]
    return {"decisions": len(trace),
            "proposed_on_fraction": sum(selected) / len(selected) if selected else None,
            "executed_on_fraction": sum(executed) / len(executed) if executed else None,
            "constant_proposal": ("always_on" if selected[0] else "always_off") if selected and len(set(selected)) == 1 else "mixed",
            "episode_return": float(sum(r["reward"] for r in trace)),
            "discounted_return": float(sum(gamma**i * r["reward"] for i,r in enumerate(trace))),
            "reward_risk": "Early termination removes future negative rewards; compare progression and exposure, not raw episode returns."}


def make_environment(twin, initial_state, config):
    return OncoEvoEnv(deepcopy(twin), initial_state, config["decision_days"], config["horizon_days"],
        config["progression_multiplier"], config["safety_multiplier"], config["observation_noise_sd"],
        config.get("reward_weights"), config.get("valid_actions", [0, 1]),
        config.get("absorbing_failure_cost", False), config.get("reward_gamma", .99))


class ExperimentRunner:
    """Pause is cooperative: advance stops immediately while paused. Replay resets RNG."""
    def __init__(self, twin, initial_state, config, policy, seed=42):
        self.twin, self.initial_state, self.config = deepcopy(twin), np.array(initial_state).copy(), deepcopy(config)
        self.policy, self.seed = policy, seed
        if isinstance(policy, PPO):
            validate_policy(policy)
        self.reset()

    def reset(self):
        self.env = make_environment(self.twin, self.initial_state, self.config)
        self.observation, self.info = self.env.reset(seed=self.seed)
        self.reference = self.info["psa_ng_ml"]
        self.info["reference_psa"] = self.reference
        self.previous_action = 1  # Published intermittent baseline begins ON.
        self.trace, self.paused, self.done = [], False, False
        return self.observation.copy()

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def step(self):
        if self.paused or self.done:
            raise RuntimeError("Experiment paused or finished")
        before = self.observation.copy()
        policy_info = {"name": type(self.policy).__name__}
        if isinstance(self.policy, PPO):
            action = int(self.policy.predict(before, deterministic=True)[0])
            with torch.no_grad():
                tensor, _ = self.policy.policy.obs_to_tensor(before)
                probabilities = self.policy.policy.get_distribution(tensor).distribution.probs.cpu().numpy()[0]
            policy_info.update(getattr(self.policy, "oncoevo_provenance", {}))
            policy_info["action_probabilities"] = probabilities.tolist()
        else:
            action = int(self.policy(before, self.info, self.previous_action))
        self.observation, reward, terminated, truncated, self.info = self.env.step(action)
        self.info["reference_psa"] = self.reference
        self.previous_action = self.info["applied_action"]
        self.done = terminated or truncated
        record = {"day": self.info["day"], "observed_state": before.tolist(), "selected_action": action,
            "executed_action": self.previous_action, "safety_override": self.info["safety_override"],
            "reward": reward, "reward_components": self.info["reward_components"],
            "next_state": self.observation.tolist(), "policy_info": policy_info,
            "terminated": terminated, "truncated": truncated, "simulated": True}
        self.trace.append(record)
        return deepcopy(record)

    def advance(self, steps=None):
        count = 0
        while not self.paused and not self.done and (steps is None or count < steps):
            self.step()
            count += 1
        return deepcopy(self.trace)

    def replay(self):
        self.reset()
        return self.advance()

    @classmethod
    def compare(cls, twin, initial_state, config, policies, seed=42):
        return {name: cls(twin, initial_state, config, policy, seed).advance()
                for name, policy in policies.items()}
