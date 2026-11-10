import type { ReactNode } from "react";

import { Microscope, ArrowRight } from "lucide-react";

import { Card } from "./components/ui/button";

import { names, type Decision, type Scenario, type Status } from "./api";

export const fmt = (v: number | undefined, digits = 2) =>
  v === undefined
    ? "—"
    : v.toLocaleString(undefined, { maximumFractionDigits: digits });

export function Tag({
  children,

  tone = "teal",
}: {
  children: ReactNode;

  tone?: string;
}) {
  return <span className={"tag " + tone}>{children}</span>;
}

export function Metric({
  label,

  value,

  detail,

  color,
}: {
  label: string;

  value: ReactNode;

  detail: string;

  color?: string;
}) {
  return (
    <Card className="metric">
      <div className="eyebrow">{label}</div>

      <strong style={{ color }}>{value}</strong>

      <small>{detail}</small>
    </Card>
  );
}

export function Empty({
  title,

  children,
}: {
  title: string;

  children: ReactNode;
}) {
  return (
    <div className="empty">
      <Microscope size={32} />

      <h3>{title}</h3>

      <p>{children}</p>
    </div>
  );
}

export function Field({
  label,

  children,

  hint,
}: {
  label: string;

  children: ReactNode;

  hint?: string;
}) {
  return (
    <label className="field">
      <span>{label}</span>

      {children}

      {hint && <small>{hint}</small>}
    </label>
  );
}

export function Settings({
  scenario,

  setScenario,

  status,
}: {
  scenario: Scenario;

  setScenario: (s: Scenario) => void;

  status: Status | null;
}) {
  const set = (key: keyof Scenario, value: number | string) =>
    setScenario({ ...scenario, [key]: value });

  return (
    <fieldset className="settings">
      <Field label="Decision policy">
        <select
          aria-label="Decision policy"
          value={scenario.policy}
          onChange={(e) => set("policy", e.target.value)}
        >
          {(["ppo", "continuous", "intermittent"] as const).map((p) => (
            <option
              key={p}
              value={p}
              disabled={p === "ppo" && !status?.model.available}
            >
              {names[p]}

              {p === "ppo" && !status?.model.available ? " · unavailable" : ""}
            </option>
          ))}
        </select>
      </Field>

      <div className="two-fields">
        <Field label="Horizon · days">
          <input
            aria-label="Horizon days"
            type="number"
            min={28}
            max={1820}
            value={scenario.horizon_days}
            onChange={(e) => set("horizon_days", +e.target.value)}
          />
        </Field>

        <Field label="Random seed">
          <input
            aria-label="Random seed"
            type="number"
            min={0}
            max={2147483647}
            value={scenario.seed}
            onChange={(e) => set("seed", +e.target.value)}
          />
        </Field>
      </div>

      <Field
        label={`Sensitive growth · ${scenario.growth_multiplier.toFixed(2)}×`}
        hint="Changes simulated biology; PPO weights stay fixed"
      >
        <input
          aria-label="Sensitive growth multiplier"
          type="range"
          min={0.5}
          max={1.5}
          step={0.05}
          value={scenario.growth_multiplier}
          onChange={(e) => set("growth_multiplier", +e.target.value)}
        />
      </Field>

      <Field
        label={`Resistant growth · ${scenario.resistance_multiplier.toFixed(2)}×`}
      >
        <input
          aria-label="Resistant growth multiplier"
          type="range"
          min={0.5}
          max={2}
          step={0.05}
          value={scenario.resistance_multiplier}
          onChange={(e) => set("resistance_multiplier", +e.target.value)}
        />
      </Field>

      <Field
        label={`Treatment response · ${scenario.kill_multiplier.toFixed(2)}×`}
        hint="Assumed drug-effect perturbation, not measured efficacy"
      >
        <input
          aria-label="Treatment response multiplier"
          type="range"
          min={0.5}
          max={1.5}
          step={0.05}
          value={scenario.kill_multiplier}
          onChange={(e) => set("kill_multiplier", +e.target.value)}
        />
      </Field>

      <div className="two-fields">
        <Field label="Decision days">
          <input
            aria-label="Decision days"
            type="number"
            min={7}
            max={90}
            value={scenario.decision_days}
            onChange={(e) => set("decision_days", +e.target.value)}
          />
        </Field>

        <Field label="Force ON · PSA ratio">
          <input
            aria-label="Safety threshold"
            type="number"
            min={1}
            max={3}
            step={0.1}
            value={scenario.safety_multiplier}
            onChange={(e) => set("safety_multiplier", +e.target.value)}
          />
        </Field>
      </div>

      <Field label="Progression · starting PSA ratio">
        <input
          aria-label="Progression threshold"
          type="number"
          min={1.1}
          max={4}
          step={0.1}
          value={scenario.progression_multiplier}
          onChange={(e) => set("progression_multiplier", +e.target.value)}
        />
      </Field>

      {scenario.patient_id === null && (
        <Field label="Synthetic starting PSA · ng/mL">
          <input
            aria-label="Synthetic PSA"
            type="number"
            min={0.1}
            max={60}
            step={0.1}
            value={scenario.synthetic_psa}
            onChange={(e) => set("synthetic_psa", +e.target.value)}
          />
        </Field>
      )}
    </fieldset>
  );
}

export function DecisionEvidence({
  decision,

  schema,
}: {
  decision: Decision;

  schema: string[];
}) {
  const probs = decision.policy_info.action_probabilities;

  return (
    <div className="evidence">
      <div className="decision-banner">
        <div>
          <span className="eyebrow">DECISION AT DAY {decision.day}</span>

          <h3>
            {decision.selected_action ? "Treatment ON" : "Treatment OFF"}{" "}
            <ArrowRight size={16} />{" "}
            {decision.executed_action ? "Executed ON" : "Executed OFF"}
          </h3>
        </div>

        <Tag tone={decision.safety_override ? "amber" : "teal"}>
          {decision.safety_override ? "Safety override" : "Action accepted"}
        </Tag>
      </div>

      <div className="two-fields">
        <div>
          <h4>Available observation → next state</h4>

          {decision.observed_state.map((v, i) => (
            <div className="evidence-row" key={i}>
              <span title={schema[i]}>
                {
                  [
                    "Log PSA",

                    "Log PSA change",

                    "Previous treatment",

                    "Elapsed / horizon",
                  ][i]
                }
              </span>

              <code>{fmt(v, 4)}</code>

              <ArrowRight size={12} />

              <code>{fmt(decision.next_state[i], 4)}</code>
            </div>
          ))}
        </div>

        <div>
          <h4>Calculated reward terms</h4>

          {Object.entries(decision.reward_components).map(([name, value]) => (
            <div className="reward-row" key={name}>
              <span>{name}</span>

              <div>
                <i
                  style={{ width: `${Math.min(100, Math.abs(value) * 30)}%` }}
                />
              </div>

              <code>{fmt(value, 3)}</code>
            </div>
          ))}

          <div className="evidence-row">
            <b>Total reward</b>

            <strong>{fmt(decision.reward, 3)}</strong>
          </div>
        </div>
      </div>

      {probs ? (
        <div className="probability">
          <div className="flex justify-between">
            <h4>Checkpoint action distribution</h4>

            <small>Deterministic argmax</small>
          </div>

          <div className="probability-bar">
            <div style={{ width: `${probs[0] * 100}%` }}>
              OFF {(probs[0] * 100).toFixed(1)}%
            </div>

            <div style={{ width: `${probs[1] * 100}%` }}>
              ON {(probs[1] * 100).toFixed(1)}%
            </div>
          </div>

          <p className="caption">
            Policy probability is not confidence in clinical benefit.
            Neural-network reasoning is not inferred.
          </p>
        </div>
      ) : (
        <p className="caption">
          {decision.policy_info.name} · Rule or explicit intervention; no neural
          policy probabilities.
        </p>
      )}
    </div>
  );
}
