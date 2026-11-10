"""Causal PSA-only treatment environment; latent populations remain hidden."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.tumor_model import advance, psa


class OncoEvoEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, twin, initial_state=None, decision_days=28, horizon_days=728,
                 progression_multiplier=2.0, safety_multiplier=1.5, noise_sd=0.03, reward_weights=None, valid_actions=(0, 1),
                 absorbing_failure_cost=False, reward_gamma=.99):
        super().__init__()
        self.twin = twin
        self.start_state = np.asarray(initial_state if initial_state is not None else twin.initial_state(), dtype=float)
        self.decision_days = int(decision_days)
        self.horizon_days = int(horizon_days)
        self.progression_multiplier = float(progression_multiplier)
        self.safety_multiplier = float(safety_multiplier)
        self.noise_sd = float(noise_sd)
        self.absorbing_failure_cost = bool(absorbing_failure_cost)
        self.reward_gamma = float(reward_gamma)
        if not 0 < self.reward_gamma <= 1:
            raise ValueError("Reward discount must be in (0,1]")
        self.reward_weights = {"disease": 1.0, "exposure": .18, "progression": 3.0, "override": .25}
        self.reward_weights.update(reward_weights or {})
        self.valid_actions = tuple(valid_actions)
        if not self.valid_actions or not set(self.valid_actions).issubset({0, 1}):
            raise ValueError("Valid actions must be a nonempty subset of [0, 1]")
        if 1 not in self.valid_actions:
            raise ValueError("Safety force-ON requires treatment ON to remain permitted")
        if any(not np.isfinite(v) or v < 0 for v in self.reward_weights.values()):
            raise ValueError("Reward weights must be finite and nonnegative")
        if (self.start_state.shape != (2,) or not np.isfinite(self.start_state).all()
                or (self.start_state < 0).any() or self.start_state.sum() <= 0):
            raise ValueError("Initial simulated populations must be finite, nonnegative and nonzero")
        if not np.isfinite([noise_sd, progression_multiplier, safety_multiplier]).all() or noise_sd < 0 or progression_multiplier <= 0 or safety_multiplier <= 0:
            raise ValueError("Invalid noise or safety thresholds")
        if self.decision_days <= 0 or self.horizon_days <= 0:
            raise ValueError("Intervals must be positive")
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-10, 10, shape=(4,), dtype=np.float32)

    def _observation(self):
        observation = np.array([np.log1p(self.observed_psa),
                         np.clip(np.log((self.observed_psa + .1) / (self.previous_psa + .1)), -5, 5),
                         float(self.last_action), self.elapsed / self.horizon_days], dtype=np.float32)
        if not self.observation_space.contains(observation):
            raise RuntimeError("Observation exceeded the saved policy domain; reset with valid simulation parameters")
        return observation

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.start_state.copy()
        self.done = False
        self.elapsed = 0
        self.last_action = 0
        self.overrides = 0
        self.reference_psa = psa(self.state, self.twin)
        self.observed_psa = self.reference_psa
        self.previous_psa = self.observed_psa
        return self._observation(), {"psa_ng_ml": self.observed_psa, "simulated_state": self.state.copy()}

    def step(self, action):
        if not self.action_space.contains(action) or int(action) not in self.valid_actions:
            raise ValueError("Action must be 0 (OFF) or 1 (ON)")
        if self.done:
            raise RuntimeError("Episode ended; call reset")
        observed_state = self._observation().copy()
        action = int(action)
        forced = action == 0 and self.observed_psa >= self.safety_multiplier * self.reference_psa
        applied = 1 if forced else action
        days = min(self.decision_days, self.horizon_days - self.elapsed)
        self.state = advance(self.state, days, applied, self.twin)
        self.elapsed += days
        self.previous_psa = self.observed_psa
        true_psa = psa(self.state, self.twin)
        self.observed_psa = max(0.0, true_psa * np.exp(self.np_random.normal(0, self.noise_sd)))
        self.last_action = applied
        self.overrides += int(forced)
        progression = true_psa >= self.progression_multiplier * self.reference_psa
        components = {"disease": -self.reward_weights["disease"] * np.log1p(true_psa / self.reference_psa),
                      "exposure": -self.reward_weights["exposure"] * applied,
                      "progression": -self.reward_weights["progression"] * progression,
                      "override": -self.reward_weights["override"] * forced}
        if progression and self.absorbing_failure_cost:
            # Charge the discounted remaining failed-state costs at termination.
            # This prevents escaping future penalties by ending an episode early.
            remaining = int(np.ceil((self.horizon_days - self.elapsed) / self.decision_days))
            discount_sum = sum(self.reward_gamma ** i for i in range(1, remaining + 1))
            failed_cost = self.reward_weights["disease"] * np.log1p(self.progression_multiplier) + self.reward_weights["progression"]
            components["failure_continuation"] = -float(failed_cost * discount_sum)
        reward = sum(components.values())
        terminated = bool(progression)
        truncated = bool(self.elapsed >= self.horizon_days and not terminated)
        self.done = terminated or truncated
        info = {"observed_state": observed_state, "selected_action": action, "reward_components": components, "day": self.elapsed, "psa_ng_ml": self.observed_psa, "true_psa_ng_ml": true_psa,
                "simulated_state": self.state.copy(), "applied_action": applied,
                "safety_override": bool(forced), "progression": bool(progression), "days": days}
        return self._observation(), float(reward), terminated, truncated, info
