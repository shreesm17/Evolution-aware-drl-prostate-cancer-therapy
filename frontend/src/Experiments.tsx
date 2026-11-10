import { useState } from "react";

import {
  BookOpen,
  Download,
  RefreshCw,
  Square,
  ChevronRight,
} from "lucide-react";

import { Button } from "./components/ui/button";

import { ArenaCharts, Chart } from "./Charts";

import { Empty, Tag, fmt } from "./Panels";

import {
  api,
  colors,
  names,
  type Experiment,
  type Policy,
  type Status,
} from "./api";

export function PolicyToggles({
  selected,

  setSelected,

  status,
}: {
  selected: Policy[];

  setSelected: (p: Policy[]) => void;

  status: Status | null;
}) {
  return (
    <div className="policy-toggles">
      {(["ppo", "continuous", "intermittent"] as const).map((p) => (
        <label key={p} className={selected.includes(p) ? "checked" : ""}>
          <input
            aria-label={`Include ${names[p]}`}
            type="checkbox"
            checked={selected.includes(p)}
            disabled={p === "ppo" && !status?.model.available}
            onChange={(e) =>
              setSelected(
                e.target.checked
                  ? [...selected, p]
                  : selected.filter((v) => v !== p),
              )
            }
          />

          <i style={{ background: colors[p] }} />

          {names[p]}

          {p === "ppo" && !status?.model.available ? " · unavailable" : ""}
        </label>
      ))}
    </div>
  );
}

export function ExperimentLibrary({
  experiments,

  selected,

  open,
}: {
  experiments: Experiment[];

  selected?: string;

  open: (id: string) => void;
}) {
  return experiments.length ? (
    <div className="experiment-library">
      {experiments.map((e) => (
        <button
          key={e.id}
          onClick={() => open(e.id)}
          className={selected === e.id ? "active" : ""}
        >
          <BookOpen size={15} />

          <span>
            {e.name}

            <small>{new Date(e.created).toLocaleString()}</small>
          </span>

          <Tag>{e.status}</Tag>

          <ChevronRight size={14} />
        </button>
      ))}
    </div>
  ) : (
    <p className="caption">
      No saved Studio experiments yet. Configure and save a local run to begin.
    </p>
  );
}

export function ExperimentPanel({
  experiment,

  setExperiment,

  refresh,

  fail,
}: {
  experiment: Experiment;

  setExperiment: (e: Experiment) => void;

  refresh: () => void;

  fail: (e: unknown) => void;
}) {
  const [busy, setBusy] = useState(false);

  const runs = experiment.runs || [];

  const ps = experiment.spec.policies.filter((p) =>
    runs.some((r) => r.policy === p),
  );

  const act = async (operation: string) => {
    setBusy(true);

    try {
      const e = await api<Experiment>(
        `/experiments/${experiment.id}/${operation}`,

        {},
      );

      setExperiment(e);

      refresh();
    } catch (e) {
      fail(e);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <div className="panel-heading">
        <div>
          <span className="eyebrow">SAVED LOCAL EXPERIMENT</span>

          <h3>{experiment.name}</h3>
        </div>

        <Tag tone={experiment.status === "failed" ? "amber" : "teal"}>
          {experiment.status}
        </Tag>
      </div>

      {experiment.status === "running" && (
        <div className="job-progress">
          <div style={{ width: `${experiment.progress * 100}%` }} />

          <span>
            {Math.round(experiment.progress * 100)}% · computing locally
          </span>
        </div>
      )}

      {experiment.error && (
        <p role="alert" className="error-text">
          {experiment.error}
        </p>
      )}

      <p className="notice">
        Research limitations: latent populations and instantaneous treatment
        effects are assumptions. Progression is checked at decision times, not
        continuously. The legacy negative-reward objective can favor shorter
        failed episodes; do not rank disease control by episode return.
      </p>

      {runs.some(
        (r) =>
          r.policy === "ppo" &&
          r.diagnostics?.constant_proposal !== "mixed" &&
          r.diagnostics,
      ) && (
        <p className="notice">
          Constant PPO proposals detected in at least one completed run. Inspect
          the exported diagnostics and safety overrides before interpreting
          adaptive behavior.
        </p>
      )}

      {runs.length > 0 ? (
        <>
          <div className="arena-cards">
            {ps.map((p) => {
              const rs = runs.filter(
                (r) =>
                  r.policy === p && (r.nominal ?? r.sensitivity_factor === 1),
              );

              const exposure =
                rs.reduce((sum, r) => sum + r.metrics.exposure_fraction, 0) /
                rs.length;

              return (
                <div key={p} style={{ borderTopColor: colors[p] }}>
                  <span>{names[p]}</span>

                  <strong>
                    {Number.isFinite(exposure)
                      ? `${(exposure * 100).toFixed(1)}%`
                      : "Pending"}
                  </strong>

                  <small>mean nominal treatment exposure</small>

                  <p>
                    {rs.filter((r) => r.metrics.progression).length} /{" "}
                    {rs.length} modeled progressions
                  </p>
                </div>
              );
            })}
          </div>

          <ArenaCharts runs={runs} />

          <div className="metrics-table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Policy / seed</th>

                  <th>Growth / response / resistance</th>

                  <th>Exposure</th>

                  <th>Progression day</th>

                  <th>Overrides</th>

                  <th>Switches</th>
                </tr>
              </thead>

              <tbody>
                {runs.map((r, i) => (
                  <tr key={i}>
                    <td>
                      <i
                        className="color-dot"
                        style={{ background: colors[r.policy] }}
                      />
                      {names[r.policy]} <small>#{r.seed}</small>
                    </td>

                    <td>
                      {r.sensitivity_factor.toFixed(1)} /{" "}
                      {(r.kill_factor ?? 1).toFixed(1)} /{" "}
                      {(r.resistance_factor ?? 1).toFixed(1)}×
                    </td>

                    <td>{(r.metrics.exposure_fraction * 100).toFixed(1)}%</td>

                    <td>
                      {r.metrics.time_to_progression ??
                        `≥ ${r.metrics.followup_days} (censored)`}
                    </td>

                    <td>{r.metrics.safety_overrides}</td>

                    <td>{r.metrics.policy_switches}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <Chart
            label="Calculated treatment exposure sensitivity ranges"
            height={220}
            option={{
              grid: { left: 55, right: 30, top: 25, bottom: 35 },

              xAxis: {
                type: "category",

                data: ps.map((p) => names[p]),

                axisLabel: { color: "#b2c1d6" },
              },

              yAxis: {
                type: "value",

                name: "Exposure %",

                axisLabel: { color: "#b2c1d6" },

                splitLine: { lineStyle: { color: "#233149" } },
              },

              tooltip: { trigger: "axis" },

              series: [
                {
                  name: "Minimum completed variant",

                  type: "bar",

                  data: ps.map((p) =>
                    Math.min(
                      ...runs

                        .filter((r) => r.policy === p)

                        .map((r) => r.metrics.exposure_fraction * 100),
                    ),
                  ),

                  itemStyle: { color: "#5ce2ca" },
                },

                {
                  name: "Maximum completed variant",

                  type: "bar",

                  data: ps.map((p) =>
                    Math.max(
                      ...runs

                        .filter((r) => r.policy === p)

                        .map((r) => r.metrics.exposure_fraction * 100),
                    ),
                  ),

                  itemStyle: { color: "#b49af7" },
                },
              ],
            }}
          />

          <p className="caption">
            Min/max across completed seeds and biological variants: sensitivity
            ranges, not confidence intervals. Progression is a modeled PSA
            threshold; censored runs are not observed progression events.
          </p>
        </>
      ) : (
        <Empty title="Computing the matched experiment">
          Results appear as each virtual run completes. Cancellation retains
          completed runs.
        </Empty>
      )}

      <div className="button-row">
        {(["csv", "json", "report"] as const).map((format) => (
          <Button key={format} variant="outline" size="sm" asChild>
            <a
              href={`/api/experiments/${experiment.id}/export?format=${format}`}
              download
            >
              <Download size={13} />

              {format === "report" ? "Readable report" : format.toUpperCase()}
            </a>
          </Button>
        ))}

        {experiment.status === "running" ? (
          <Button
            disabled={busy}
            variant="destructive"
            size="sm"
            onClick={() => act("cancel")}
          >
            <Square size={12} />
            Cancel run
          </Button>
        ) : (
          <Button
            disabled={busy}
            variant="outline"
            size="sm"
            onClick={() => act("reproduce")}
          >
            <RefreshCw size={13} />
            Reproduce saved run
          </Button>
        )}
      </div>

      <details>
        <summary>Inspect saved configuration and provenance</summary>

        <pre>{JSON.stringify(experiment.spec, null, 2)}</pre>

        <p className="caption">
          Export JSON includes every computed frame, reward component, decision
          trace, and model checkpoint hash.
        </p>
      </details>

      {runs[0] && (
        <details>
          <summary>
            Inspect computed decision trace · first completed run
          </summary>

          <pre>{JSON.stringify(runs[0].trace, null, 2)}</pre>
        </details>
      )}
    </>
  );
}
