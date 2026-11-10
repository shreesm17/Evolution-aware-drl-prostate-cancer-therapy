export type Policy = "ppo" | "continuous" | "intermittent";

export type Scenario = {
  patient_id: number | null;

  policy: Policy;

  seed: number;

  horizon_days: number;

  decision_days: number;

  growth_multiplier: number;

  resistance_multiplier: number;

  kill_multiplier: number;

  safety_multiplier: number;

  progression_multiplier: number;

  synthetic_psa: number;
};

export type Patient = {
  id: number;

  measurements: number;

  followup_days: number;

  initial_psa: number;

  latest_psa: number;

  on_visits: number;

  split: string;

  psa_trend: number[];
};

export type History = {
  id: number;

  observations: {
    day: number;

    psa_ng_ml: number;

    androgen_reported: number;

    treatment_on: number;

    cycle: number;
  }[];

  forecast: { day: number; psa: number }[];

  calibration: {
    cutoff_day: number;

    n_prefix: number;

    prefix_log_rmse: number;

    parameters: Record<string, number>;

    at_parameter_bound?: boolean;

    jacobian_condition?: number | null;
  };

  forecast_log_rmse: number | null;
};

export type Frame = {
  day: number;

  psa: number;

  observed_psa: number;

  sensitive: number;

  resistant: number;

  treatment: number;

  reward: number;

  exposure_days: number;
};

export type Decision = {
  day: number;

  observed_state: number[];

  selected_action: number;

  executed_action: number;

  safety_override: boolean;

  reward: number;

  reward_components: Record<string, number>;

  next_state: number[];

  policy_info: {
    name: string;

    action_probabilities?: number[];

    sha256?: string;
  };

  terminated: boolean;

  truncated: boolean;
};

export type Simulation = {
  id: string;

  scenario: Scenario;

  paused: boolean;

  done: boolean;

  frames: Frame[];

  trace: Decision[];

  calibration: Record<string, unknown>;
};

export type Status = {
  model: { available: boolean; sha256?: string; error?: string };

  cohort_size: number;

  observations: number;

  audit: { file: string; correction: string; field: string }[];

  observation_schema: string[];
};

export type Spec = {
  name: string;

  scenario: Scenario;

  policies: Policy[];

  seeds: number[];

  sensitivity: boolean;
};

export type Run = {
  policy: Policy;

  seed: number;

  sensitivity_factor: number;

  nominal?: boolean;

  kill_factor?: number;

  resistance_factor?: number;

  diagnostics?: {
    constant_proposal: string;
    proposed_on_fraction: number;
    reward_risk: string;
  };

  frames: Frame[];

  trace: Decision[];

  metrics: {
    exposure_days: number;

    exposure_fraction: number;

    progression: boolean;

    followup_days: number;

    time_to_progression: number | null;

    safety_overrides: number;

    policy_switches: number;
  };
};

export type Experiment = {
  id: string;

  name: string;

  created: string;

  spec: Spec;

  status: string;

  progress: number;

  runs?: Run[];

  error?: string;
};

export const defaultScenario: Scenario = {
  patient_id: 1,

  policy: "ppo",

  seed: 42,

  horizon_days: 728,

  decision_days: 28,

  growth_multiplier: 1,

  resistance_multiplier: 1,

  kill_multiplier: 1,

  safety_multiplier: 1.5,

  progression_multiplier: 2,

  synthetic_psa: 10,
};

export const names: Record<Policy, string> = {
  ppo: "PPO adaptive",

  continuous: "Continuous",

  intermittent: "Intermittent protocol",
};

export const colors: Record<Policy, string> = {
  ppo: "#5ce2ca",

  continuous: "#7397ff",

  intermittent: "#b49af7",
};

export async function api<T>(path: string, body?: unknown): Promise<T> {
  const r = await fetch("/api" + path, {
    signal: AbortSignal.timeout(30000),

    method: body === undefined ? "GET" : "POST",

    headers: { "Content-Type": "application/json" },

    body: body === undefined ? undefined : JSON.stringify(body),
  });

  if (!r.ok) {
    let msg = r.statusText;

    try {
      const e = await r.json();

      msg = typeof e.detail === "string" ? e.detail : JSON.stringify(e.detail);
    } catch {}

    throw new Error(msg);
  }

  return r.json();
}
