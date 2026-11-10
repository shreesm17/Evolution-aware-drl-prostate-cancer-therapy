from src.utils import calibrated_twin, intermittent_policy
"""Held-out observed PSA forecast and matched virtual-patient policy comparisons."""
import argparse
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from src.data import prepare, calibration_prefix
from src.tumor_model import fit_prefix, state_at_end, historical_prediction
from src.experiment import ExperimentRunner, load_ppo, policy_diagnostics
from src.utils import load_config, ContinuousPolicy, IntermittentPolicy, ensure_dirs


def replay(twin, initial_state, config, policy, seed):
    runner = ExperimentRunner(twin, initial_state, config, policy, seed)
    rows = [{"day": 0, "psa_ng_ml": runner.reference, "sensitive_simulated": initial_state[0],
             "resistant_simulated": initial_state[1], "treatment_on": 0}]
    exposure = 0
    while not runner.done:
        record = runner.step()
        info = runner.info
        exposure += info["days"] * info["applied_action"]
        rows.append({**record, "psa_ng_ml": info["true_psa_ng_ml"],
                     "sensitive_simulated": info["simulated_state"][0],
                     "resistant_simulated": info["simulated_state"][1], "treatment_on": info["applied_action"]})
    actions = [r["executed_action"] for r in runner.trace]
    return rows, {"time_to_progression_days": info["day"] if info["progression"] else np.nan,
                  **policy_diagnostics(runner.trace, config.get("reward_gamma", .99)), "progression": int(info["progression"]), "exposure_fraction": exposure / info["day"],
                  "treatment_days": exposure, "policy_switches": sum(a != b for a,b in zip(actions, actions[1:])),
                  "safety_overrides": runner.env.overrides, "horizon_days": info["day"]}


def evaluate(config_path="config.yaml", split="test", max_patients=None, output_dir="results", checkpoint_path="models/ppo_oncoevo.zip", nominal=False):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    ensure_dirs()
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError("Train first: python -m src.train")
    patients, splits, _ = prepare(seed=config["seed"])
    model = load_ppo(checkpoint)
    ids = splits[split][:max_patients]
    metadata_path = checkpoint.with_name("training_metadata.json" if checkpoint.stem == "ppo_oncoevo" else checkpoint.stem + "_metadata.json")
    if not metadata_path.exists():
        raise ValueError("Training metadata required to verify patient separation")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("checkpoint_sha256", model.oncoevo_provenance["sha256"]) != model.oncoevo_provenance["sha256"]:
        raise ValueError("Training manifest does not match checkpoint hash")
    if metadata["training_timesteps"] != model.num_timesteps:
        raise ValueError("Training manifest timestep count does not match checkpoint")
    # Compare rewards under the checkpoint's training objective, not a new UI objective.
    config["absorbing_failure_cost"] = metadata["config"].get("absorbing_failure_cost", False)
    config["reward_gamma"] = model.gamma
    overlap = set(ids) & set(metadata["training_patient_ids"])
    if overlap:
        raise ValueError(f"Held-out split overlaps training patients: {sorted(overlap)}")
    metrics, traces, forecasts = [], [], []
    for patient_id in ids:
        frame = patients[patient_id]
        prefix = calibration_prefix(frame, config["calibration_fraction"])
        twin, fit = calibrated_twin(prefix, config)
        state = state_at_end(prefix, twin)
        prediction = historical_prediction(frame, twin)
        for i in range(len(prefix), len(frame)):
            forecasts.append({"patient_id": patient_id, "day": frame.day.iloc[i],
                              "observed_psa_ng_ml": frame.psa_ng_ml.iloc[i], "forecast_psa_ng_ml": prediction[i]})
        for factor in ([1.] if nominal else config["sensitivity_growth_factors"]):
            for resistance_factor in ([1.] if nominal else config["sensitivity_resistance_factors"]):
                for kill_factor in ([1.] if nominal else config.get("sensitivity_kill_factors", [.8,1.,1.2])):
                    variant = replace(twin, growth=twin.growth * factor,
                                      resistant_growth=twin.resistant_growth * resistance_factor, kill=twin.kill * kill_factor)
                    for seed in config["evaluation_seeds"]:
                        for name, policy in (("PPO", model), ("Continuous", ContinuousPolicy()),
                                             ("Intermittent threshold", intermittent_policy(config))):
                            trace, summary = replay(variant, state, config, policy, seed)
                            metrics.append({"patient_id": patient_id, "policy": name, "seed": seed,
                                            "growth_factor": factor, "kill_factor": kill_factor, "resistance_factor": resistance_factor,
                                            "prefix_log_rmse": fit["prefix_log_rmse"], **summary})
                            if kill_factor == 1.0 and factor == 1.0 and resistance_factor == 1.0 and seed == config["evaluation_seeds"][0]:
                                traces.extend({"patient_id": patient_id, "policy": name, **row} for row in trace)
    pd.DataFrame(metrics).to_csv(output / "policy_metrics.csv", index=False)
    pd.DataFrame(traces).to_csv(output / "traces.csv", index=False)
    pd.DataFrame(forecasts).to_csv(output / "observed_forecasts.csv", index=False)
    forecast = pd.DataFrame(forecasts)
    fit_summary = {"heldout_log_rmse": float(np.sqrt(np.mean((np.log1p(forecast.forecast_psa_ng_ml) -
                                                                np.log1p(forecast.observed_psa_ng_ml)) ** 2))) if len(forecast) else None,
                   "checkpoint": model.oncoevo_provenance, "effective_config": config,
                   "manifest_hash_verified": "checkpoint_sha256" in metadata,
                   "training_patient_overlap": sorted(overlap), "n_patients": len(ids), "n_future_observations": len(forecast), "split": split,
                   "interpretation": "Historical PSA forecast; policy metrics are counterfactual simulations only."}
    (output / "evaluation_summary.json").write_text(json.dumps(fit_summary, indent=2))
    return fit_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--max-patients", type=int)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--checkpoint", default="models/ppo_oncoevo.zip")
    parser.add_argument("--nominal", action="store_true", help="Skip biological perturbations for a lightweight check")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.config, args.split, args.max_patients, args.output_dir, args.checkpoint, args.nominal), indent=2))
