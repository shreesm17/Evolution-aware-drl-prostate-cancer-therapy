"""Two-population competitive ODE; cell states are latent simulations.

Competition is inspired by Zhang et al., Nat Commun 2017,
https://www.nature.com/articles/s41467-017-01968-5 . This is a simplified
two-population adaptation, not their three-population/PSA-clearance model.
No pharmacokinetics, testosterone recovery, mutation or spatial structure is modeled.
PSA is an imperfect proxy, modeled as the sum of scaled populations.
Only net sensitive growth and treatment kill are fit from early PSA.
Resistant fraction, growth, competition and PSA scale are fixed assumptions.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


@dataclass
class Twin:
    initial_psa: float
    growth: float = 0.006
    kill: float = 0.014
    resistant_growth: float = 0.002
    resistant_fraction: float = 0.03
    carrying_capacity: float = 100.0
    competition: float = 1.0
    floor_psa: float = 0.05

    def __post_init__(self):
        if not np.isfinite(list(asdict(self).values())).all():
            raise ValueError("Twin parameters must be finite")
        if min(self.initial_psa, self.growth, self.kill, self.resistant_growth, self.competition, self.floor_psa) < 0 or self.carrying_capacity <= 0 or not 0 <= self.resistant_fraction <= 1:
            raise ValueError("Invalid biological parameter range")

    def initial_state(self):
        total = max(self.initial_psa - self.floor_psa, 0.05)
        return np.array([total * (1 - self.resistant_fraction), total * self.resistant_fraction])


def advance(state, days, treatment_on, twin):
    state = np.asarray(state, dtype=float)
    if state.shape != (2,) or not np.isfinite(state).all() or (state < 0).any():
        raise ValueError("Invalid simulated populations")
    if not np.isfinite(days) or days < 0 or treatment_on not in (0, 1):
        raise ValueError("Invalid interval or action")
    if days == 0:
        return np.asarray(state, dtype=float).copy()
    def rhs(_, y):
        s, r = np.maximum(y, 0)
        k = twin.carrying_capacity
        return [twin.growth * s * (1 - (s + twin.competition * r) / k) - twin.kill * treatment_on * s,
                twin.resistant_growth * r * (1 - (r + twin.competition * s) / k)]
    result = solve_ivp(rhs, (0, days), state, method="RK45", rtol=1e-6, atol=1e-8)
    if not result.success or not np.isfinite(result.y[:, -1]).all():
        raise RuntimeError(result.message)
    return np.maximum(result.y[:, -1], 0)


def psa(state, twin):
    return float(np.sum(state) + twin.floor_psa)


def historical_prediction(frame, twin):
    """Status at row i applies from day i through the next observation."""
    state = twin.initial_state()
    values = [psa(state, twin)]
    for i in range(len(frame) - 1):
        state = advance(state, float(frame.day.iloc[i + 1] - frame.day.iloc[i]), int(frame.treatment_on.iloc[i]), twin)
        values.append(psa(state, twin))
    return np.asarray(values)


def state_at_end(frame, twin):
    state = twin.initial_state()
    for i in range(len(frame) - 1):
        state = advance(state, float(frame.day.iloc[i + 1] - frame.day.iloc[i]), int(frame.treatment_on.iloc[i]), twin)
    return state


def fit_prefix(frame, template=None):
    """Fit observable effective rates; hold growth fixed for ON-only prefixes."""
    if len(frame) < 3:
        raise ValueError("At least three early observations required")
    twin = template or Twin(initial_psa=float(frame.psa_ng_ml.iloc[0]))
    twin.initial_psa = float(frame.psa_ng_ml.iloc[0])
    def residual(x):
        twin.growth, twin.kill = map(float, x)
        return np.log1p(historical_prediction(frame, twin)) - np.log1p(frame.psa_ng_ml.to_numpy())
    statuses = set(frame.treatment_on.iloc[:-1].astype(int))
    if statuses == {1}:
        def one_rate(x):
            twin.kill = float(x[0])
            return residual([twin.growth, twin.kill])
        fit = least_squares(one_rate, [twin.kill], bounds=([0.0001], [0.08]), max_nfev=80)
        twin.kill = float(fit.x[0])
        fitted = ["kill"]
    elif statuses == {0}:
        def one_rate(x):
            twin.growth = float(x[0])
            return residual([twin.growth, twin.kill])
        fit = least_squares(one_rate, [twin.growth], bounds=([0.0001], [0.03]), max_nfev=80)
        twin.growth = float(fit.x[0])
        fitted = ["growth"]
    else:
        fit = least_squares(residual, [twin.growth, twin.kill], bounds=([0.0001, 0.0001], [0.03, 0.08]), max_nfev=80)
        twin.growth, twin.kill = map(float, fit.x)
        fitted = ["growth", "kill"]
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    condition = float(singular[0] / singular[-1]) if len(singular) and singular[-1] > 1e-12 else None
    return twin, {"prefix_log_rmse": float(np.sqrt(np.mean(fit.fun ** 2))), "n_prefix": len(frame),
                  "parameters": asdict(twin), "fitted_parameters": fitted, "fit_success": bool(fit.success),
                  "at_parameter_bound": bool(np.any(fit.active_mask)), "jacobian_condition": condition,
                  "identifiability_note": "Conditional effective rates only. Fit quality and local Jacobian conditioning do not establish biological identifiability."}
