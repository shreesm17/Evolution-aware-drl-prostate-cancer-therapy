import { useCallback, useEffect, useState, useRef } from "react";

import {
  Activity,
  Atom,
  ArrowRight,
  ChevronRight,
  FlaskConical,
  GitBranch,
  LayoutDashboard,
  Loader2,
  Pause,
  Play,
  Plus,
  RefreshCw,
  RotateCcw,
  Search,
  ShieldCheck,
  SkipForward,
  Workflow,
  X,
} from "lucide-react";

import { Button, Card } from "./components/ui/button";

import Cells from "./Cells";

import { ObservedChart, SimulationCharts } from "./Charts";

import {
  ExperimentPanel,
  ExperimentLibrary,
  PolicyToggles,
} from "./Experiments";

import {
  Settings,
  DecisionEvidence,
  Empty,
  Metric,
  Tag,
  Field,
  fmt,
} from "./Panels";

import {
  api,
  defaultScenario,
  names,
  type Experiment,
  type Frame,
  type History,
  type Patient,
  type Policy,
  type Scenario,
  type Simulation,
  type Spec,
  type Status,
} from "./api";

const tabs = [
  {
    name: "Command Center",

    subtitle: "Cohort intelligence",

    icon: LayoutDashboard,
  },

  {
    name: "Digital Tumor Twin",

    subtitle: "Interactive simulation",

    icon: Atom,
  },

  {
    name: "RL Decision Laboratory",

    subtitle: "Policy evidence",

    icon: Workflow,
  },

  { name: "Strategy Arena", subtitle: "Matched comparisons", icon: GitBranch },

  {
    name: "Experiment Center",

    subtitle: "Reproducible research",

    icon: FlaskConical,
  },
];

export default function App() {
  const generation = useRef(0);

  const [workspace, setWorkspace] = useState(0);

  const [status, setStatus] = useState<Status | null>(null);

  const [patients, setPatients] = useState<Patient[]>([]);

  const [scenario, setScenario] = useState<Scenario>(defaultScenario);

  const [history, setHistory] = useState<History | null>(null);

  const [simulation, setSimulation] = useState<Simulation | null>(null);

  const [inspect, setInspect] = useState(0);

  const [autoplay, setAutoplay] = useState(false);

  const [busy, setBusy] = useState(false);

  const [error, setError] = useState("");

  const [search, setSearch] = useState("");

  const [split, setSplit] = useState("all");

  const [experiments, setExperiments] = useState<Experiment[]>([]);

  const [experiment, setExperiment] = useState<Experiment | null>(null);

  const [experimentName, setExperimentName] = useState(
    "Adaptive therapy study",
  );

  const [selectedPolicies, setSelectedPolicies] = useState<Policy[]>([
    "ppo",

    "continuous",

    "intermittent",
  ]);

  const [seedText, setSeedText] = useState("42, 73, 101");

  const [sensitivity, setSensitivity] = useState(true);

  const [branches, setBranches] = useState<{
    source_day: number;

    outcomes: { action: number; frame: Frame }[];
  } | null>(null);

  const [loading, setLoading] = useState(true);

  const fail = useCallback((e: unknown) => {
    setError(e instanceof Error ? e.message : String(e));

    setAutoplay(false);
  }, []);

  const refreshExperiments = useCallback(
    () => api<Experiment[]>("/experiments").then(setExperiments).catch(fail),

    [fail],
  );

  useEffect(() => {
    let live = true;

    Promise.all([api<Status>("/status"), api<Patient[]>("/patients")])

      .then(([s, p]) => {
        if (!live) return;

        setStatus(s);

        setPatients(p);

        if (!s.model.available) {
          setScenario((v) => ({ ...v, policy: "continuous" }));

          setSelectedPolicies(["continuous", "intermittent"]);
        }
      })

      .catch(fail)

      .finally(() => {
        if (live) setLoading(false);
      });

    refreshExperiments();

    return () => {
      live = false;
    };
  }, [fail, refreshExperiments]);

  useEffect(() => {
    let live = true;

    setHistory(null);

    if (scenario.patient_id !== null)
      api<History>(`/patients/${scenario.patient_id}`)
        .then((h) => {
          if (live) setHistory(h);
        })

        .catch(fail);

    return () => {
      live = false;
    };
  }, [scenario.patient_id, fail]);

  useEffect(() => {
    if (!experiment || experiment.status !== "running") return;

    const timer = setInterval(() => {
      api<Experiment>(`/experiments/${experiment.id}`)
        .then((e) => {
          setExperiment(e);

          if (e.status !== "running") refreshExperiments();
        })

        .catch(fail);
    }, 700);

    return () => clearInterval(timer);
  }, [experiment?.id, experiment?.status, fail, refreshExperiments]);

  const perform = async (fn: () => Promise<void>) => {
    setError("");

    setBusy(true);

    try {
      await fn();
    } catch (e) {
      fail(e);
    } finally {
      setBusy(false);
    }
  };

  const command = useCallback(
    async (operation: string, steps = 1) => {
      if (!simulation) return;

      const revision = generation.current;

      const s = await api<Simulation>(`/simulations/${simulation.id}/command`, {
        operation,

        steps,
      });

      if (revision !== generation.current) return;

      setSimulation(s);

      setInspect(s.frames.length - 1);

      setBranches(null);

      if (s.done) setAutoplay(false);
    },

    [simulation?.id],
  );

  useEffect(() => {
    if (
      !autoplay ||
      busy ||
      !simulation ||
      simulation.paused ||
      simulation.done
    )
      return;

    const timer = setTimeout(() => {
      setBusy(true);

      command("step")
        .catch(fail)

        .finally(() => setBusy(false));
    }, 650);

    return () => clearTimeout(timer);
  }, [autoplay, busy, simulation, command, fail]);

  const updateScenario = (s: Scenario) => {
    setScenario(s);

    setAutoplay(false);
  };

  const selectPatient = (id: number | null) => {
    generation.current++;

    updateScenario({ ...scenario, patient_id: id });

    setSimulation(null);

    setBranches(null);
  };

  const create = () =>
    perform(async () => {
      setAutoplay(false);

      const revision = ++generation.current;

      const s = await api<Simulation>("/simulations", scenario);

      if (revision !== generation.current) return;

      if (simulation)
        fetch(`/api/simulations/${simulation.id}`, { method: "DELETE" }).catch(
          () => {},
        );

      setSimulation(s);

      setInspect(0);

      setBranches(null);
    });

  const runExperiment = (arena = false) =>
    perform(async () => {
      if (!selectedPolicies.length)
        throw new Error("Select at least one policy");

      const seeds = arena
        ? [scenario.seed]
        : seedText

            .split(",")

            .map((s) => s.trim())

            .filter(Boolean)

            .map(Number);

      if (
        !seeds.length ||
        seeds.some((s) => !Number.isInteger(s) || s < 0) ||
        seeds.length > 5
      )
        throw new Error("Enter one to five nonnegative integer seeds");

      const spec: Spec = {
        name: arena ? "Strategy arena comparison" : experimentName,

        scenario,

        policies: selectedPolicies,

        seeds,

        sensitivity,
      };

      setExperiment(await api<Experiment>("/experiments", spec));

      refreshExperiments();
    });

  const openExperiment = (id: string) =>
    perform(async () => {
      generation.current++;

      const e = await api<Experiment>(`/experiments/${id}`);

      setExperiment(e);

      setScenario({ ...defaultScenario, ...e.spec.scenario });

      setExperimentName(e.name);

      setSelectedPolicies(e.spec.policies);

      setSeedText(e.spec.seeds.join(", "));

      setSensitivity(e.spec.sensitivity);

      setSimulation(null);

      setAutoplay(false);

      setWorkspace(4);
    });

  const currentFrame =
    simulation?.frames[Math.min(inspect, simulation.frames.length - 1)];

  const decision = simulation?.trace[Math.max(0, inspect - 1)];

  const dirty =
    !!simulation &&
    Object.keys(defaultScenario).some(
      (key) =>
        simulation.scenario[key as keyof Scenario] !==
        scenario[key as keyof Scenario],
    );

  const currentPatient = patients.find((p) => p.id === scenario.patient_id);

  const patientSelect = (label: string) => (
    <Field label={label}>
      <select
        aria-label={label}
        value={scenario.patient_id ?? "synthetic"}
        onChange={(e) =>
          selectPatient(e.target.value === "synthetic" ? null : +e.target.value)
        }
      >
        <option value="synthetic">Explicit synthetic virtual patient</option>

        {patients.map((p) => (
          <option key={p.id} value={p.id}>
            Patient P-{String(p.id).padStart(3, "0")} · {p.split}
          </option>
        ))}
      </select>
    </Field>
  );

  return (
    <div className="studio-shell">
      <aside className="sidebar">
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();

            setWorkspace(0);
          }}
        >
          <div className="brand-mark">
            <Atom size={25} />
          </div>

          <div>
            OncoEvo<span>RESEARCH STUDIO</span>
          </div>
        </a>

        <div className="nav-caption">RESEARCH WORKSPACES</div>

        <nav>
          {tabs.map((tab, i) => (
            <button
              key={tab.name}
              onClick={() => setWorkspace(i)}
              aria-current={workspace === i ? "page" : undefined}
              className={workspace === i ? "active" : ""}
            >
              <tab.icon size={18} />

              <span>
                {tab.name}

                <small>{tab.subtitle}</small>
              </span>

              {workspace === i && <ChevronRight size={14} />}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="local-status">
            <span className={status ? "live-dot" : "offline-dot"} />

            <span>
              {status ? "Local engine connected" : "Connecting to local engine"}
            </span>
          </div>

          <p>
            Private by architecture.
            <br />
            Your data stays on this machine.
          </p>

          <div className="research-label">
            <ShieldCheck size={15} />
            RESEARCH USE ONLY
          </div>
        </div>
      </aside>

      <div className="main-shell">
        <header className="topbar">
          <div className="breadcrumb">
            Workspace <ChevronRight size={13} />
            <span>{tabs[workspace].name}</span>
          </div>

          <div className="top-actions">
            <Tag tone="blue">LOCAL / OFFLINE</Tag>

            <span className="version">v1.0</span>

            <div className="avatar">OE</div>
          </div>
        </header>

        <main>
          <div className="page-title">
            <div>
              <div className="eyebrow">
                COMPUTATIONAL ONCOLOGY /{" "}
                {String(workspace + 1).padStart(2, "0")}
              </div>

              <h1>{tabs[workspace].name}</h1>

              <p>
                {
                  [
                    "Explore the cohort. Ground every experiment in the evidence.",

                    "An evolving virtual tumor. A controlled space for discovery.",

                    "Inspect what the policy sees, selects, and changes.",

                    "Compare treatment strategies under matched virtual conditions.",

                    "Save the question, the configuration, and the computed evidence.",
                  ][workspace]
                }
              </p>
            </div>

            <div className="title-status">
              <span className="live-dot" />

              <span>
                {status?.model.available
                  ? "PPO checkpoint verified"
                  : "PPO unavailable"}
              </span>

              <small>
                {status?.model.sha256?.slice(0, 12) ||
                  "Baseline policies available"}
              </small>
            </div>
          </div>

          <div className="integrity-strip">
            <ShieldCheck size={15} />

            <span>
              Computational research prototype · not a medical treatment
              recommender
            </span>

            <div className="data-key">
              <span>
                <i className="observed" />
                Recorded
              </span>

              <span>
                <i className="estimated" />
                Calibrated estimate
              </span>

              <span>
                <i className="hypothetical" />
                Hypothetical simulation
              </span>
            </div>
          </div>

          {error && (
            <div className="error-banner" role="alert">
              <span>{error}</span>

              <Button
                variant="ghost"
                size="icon"
                aria-label="Dismiss error"
                onClick={() => setError("")}
              >
                <X size={16} />
              </Button>
            </div>
          )}

          {loading ? (
            <Card>
              <Empty title="Connecting to the scientific engine">
                <Loader2 className="animate-spin inline" /> Loading real cohort
                and validating the checkpoint…
              </Empty>
            </Card>
          ) : (
            <>
              {workspace === 0 && (
                <>
                  <div className="metric-grid">
                    <Metric
                      label="Available cohort"
                      value={status?.cohort_size ?? "—"}
                      detail="Validated recorded patient series"
                    />

                    <Metric
                      label="Clinical observations"
                      value={fmt(status?.observations, 0)}
                      detail="Actual longitudinal measurements"
                    />

                    <Metric
                      label="Data audit entries"
                      value={status?.audit.length ?? "—"}
                      detail="Corrections and explicit exclusions"
                      color="#b49af7"
                    />

                    <Metric
                      label="Saved experiments"
                      value={experiments.length}
                      detail="Local, reproducible configurations"
                      color="#7397ff"
                    />
                  </div>

                  <div className="command-grid">
                    <Card className="cohort-panel">
                      <div className="panel-heading">
                        <div>
                          <span className="eyebrow">
                            RECORDED CLINICAL DATA
                          </span>

                          <h3>Patient cohort</h3>
                        </div>

                        <Tag>{patients.length} series</Tag>
                      </div>

                      <div className="table-controls">
                        <label className="search">
                          <Search size={15} />

                          <input
                            aria-label="Search patients"
                            placeholder="Search patient ID…"
                            value={search}
                            onChange={(e) => setSearch(e.target.value)}
                          />
                        </label>

                        <select
                          aria-label="Filter patient split"
                          value={split}
                          onChange={(e) => setSplit(e.target.value)}
                        >
                          <option value="all">All splits</option>

                          {["train", "validation", "test"].map((s) => (
                            <option key={s}>{s}</option>
                          ))}
                        </select>
                      </div>

                      <div className="cohort-table">
                        <table>
                          <thead>
                            <tr>
                              <th>Patient</th>

                              <th>PSA trend</th>

                              <th>Visits</th>

                              <th>Follow-up</th>

                              <th>Split</th>
                            </tr>
                          </thead>

                          <tbody>
                            {patients

                              .filter(
                                (p) =>
                                  String(p.id).includes(search) &&
                                  (split === "all" || p.split === split),
                              )

                              .map((p) => (
                                <tr
                                  key={p.id}
                                  className={
                                    scenario.patient_id === p.id
                                      ? "selected"
                                      : ""
                                  }
                                >
                                  <td>
                                    <button
                                      className="patient-link"
                                      onClick={() => selectPatient(p.id)}
                                    >
                                      P-{String(p.id).padStart(3, "0")}
                                    </button>
                                  </td>

                                  <td>
                                    <svg
                                      width="85"
                                      height="26"
                                      role="img"
                                      aria-label={`Recorded PSA trend patient ${p.id}`}
                                    >
                                      <polyline
                                        fill="none"
                                        stroke={
                                          p.latest_psa > p.initial_psa
                                            ? "#b49af7"
                                            : "#5ce2ca"
                                        }
                                        strokeWidth="1.5"
                                        points={p.psa_trend

                                          .map(
                                            (v, i) =>
                                              `${(i / Math.max(1, p.psa_trend.length - 1)) * 80},${24 - (v / Math.max(...p.psa_trend, 1)) * 22}`,
                                          )

                                          .join(" ")}
                                      />
                                    </svg>
                                  </td>

                                  <td>{p.measurements}</td>

                                  <td>{fmt(p.followup_days, 0)} d</td>

                                  <td>
                                    <span className="split-label">
                                      {p.split}
                                    </span>
                                  </td>
                                </tr>
                              ))}
                          </tbody>
                        </table>
                      </div>
                    </Card>

                    <div className="stack">
                      <Card>
                        <div className="panel-heading">
                          <div>
                            <span className="eyebrow">
                              SELECTED PATIENT / RECORDED
                            </span>

                            <h3>
                              {currentPatient
                                ? `Patient P-${String(currentPatient.id).padStart(3, "0")}`
                                : "Select a recorded patient"}
                            </h3>
                          </div>

                          <Tag>Observed</Tag>
                        </div>

                        {history ? (
                          <>
                            <ObservedChart history={history} />

                            <div className="small-stat-row">
                              <span>
                                Initial PSA{" "}
                                <b>{fmt(currentPatient?.initial_psa)} ng/mL</b>
                              </span>

                              <span>
                                Forecast log RMSE{" "}
                                <b>
                                  {fmt(
                                    history.forecast_log_rmse ?? undefined,

                                    3,
                                  )}
                                </b>
                              </span>
                            </div>

                            <p className="caption">
                              Shaded bands: recorded treatment ON. Dashed curve:
                              calibrated estimate and forecast under recorded
                              treatment.
                            </p>

                            <Button
                              variant="outline"
                              onClick={() => setWorkspace(1)}
                            >
                              Open digital twin <ArrowRight size={14} />
                            </Button>
                          </>
                        ) : (
                          <Empty title="No recorded history loaded">
                            Select a recorded patient or wait for calibration to
                            finish.
                          </Empty>
                        )}
                      </Card>

                      <Card>
                        <div className="panel-heading">
                          <h3>Source quality & model readiness</h3>

                          <ShieldCheck size={17} className="teal" />
                        </div>

                        <div className="audit-items">
                          {status?.audit.map((a, i) => (
                            <div key={i}>
                              <span className="audit-number">0{i + 1}</span>

                              <span>
                                <b>{a.file}</b>

                                <small>{a.correction}</small>
                              </span>
                            </div>
                          ))}
                        </div>

                        {!status?.model.available && (
                          <p className="error-text">{status?.model.error}</p>
                        )}

                        <p className="caption">
                          Source: Weber / Bruchovsky intermittent androgen
                          suppression series. No clinical records are generated.
                        </p>
                      </Card>
                    </div>
                  </div>

                  <Card>
                    <div className="panel-heading">
                      <h3>Recent research experiments</h3>

                      <Button variant="ghost" onClick={() => setWorkspace(4)}>
                        View experiment center <ArrowRight size={14} />
                      </Button>
                    </div>

                    <ExperimentLibrary
                      experiments={experiments.slice(0, 4)}
                      selected={experiment?.id}
                      open={openExperiment}
                    />
                  </Card>
                </>
              )}

              {workspace === 1 && (
                <>
                  <div className="twin-grid">
                    <Card className="twin-stage">
                      <div className="panel-heading">
                        <div>
                          <span className="eyebrow">
                            HYPOTHETICAL SIMULATION
                          </span>

                          <h3>
                            {scenario.patient_id === null
                              ? "Synthetic virtual patient"
                              : `Patient P-${String(scenario.patient_id).padStart(3, "0")} · calibrated twin`}
                          </h3>
                        </div>

                        <Tag tone="violet">
                          {simulation?.done
                            ? "Completed"
                            : simulation?.paused
                              ? "Paused"
                              : autoplay
                                ? "Running"
                                : "Ready"}
                        </Tag>
                      </div>

                      {currentFrame ? (
                        <>
                          <Cells frame={currentFrame} playing={autoplay} />

                          <div className="twin-metrics">
                            <div>
                              <span>Simulated PSA</span>

                              <b>
                                {fmt(currentFrame.psa)} <small>ng/mL</small>
                              </b>
                            </div>

                            <div>
                              <span>Cumulative exposure</span>

                              <b>
                                {fmt(currentFrame.exposure_days, 0)}{" "}
                                <small>days ON</small>
                              </b>
                            </div>

                            <div>
                              <span>Last executed action</span>

                              <b className="teal">
                                {inspect === 0
                                  ? "Not yet applied"
                                  : currentFrame.treatment
                                    ? "ON"
                                    : "OFF"}
                              </b>
                            </div>
                          </div>

                          <div className="transport">
                            <Button
                              aria-label="Start or resume simulation"
                              disabled={busy || dirty || simulation?.done}
                              onClick={() =>
                                perform(async () => {
                                  if (simulation?.paused)
                                    await command("resume");

                                  setAutoplay(true);
                                })
                              }
                            >
                              <Play size={15} />
                              Start / Resume
                            </Button>

                            <Button
                              variant="outline"
                              aria-label="Pause simulation"
                              disabled={!simulation || simulation.done}
                              onClick={() => {
                                setAutoplay(false);

                                perform(() => command("pause"));
                              }}
                            >
                              <Pause size={15} />
                            </Button>

                            <Button
                              variant="outline"
                              aria-label="Step simulation"
                              disabled={
                                busy ||
                                dirty ||
                                simulation?.paused ||
                                simulation?.done ||
                                autoplay
                              }
                              onClick={() => perform(() => command("step"))}
                            >
                              <SkipForward size={15} />
                              Step
                            </Button>

                            <Button
                              variant="ghost"
                              aria-label="Reset simulation"
                              disabled={busy}
                              onClick={() => {
                                setAutoplay(false);

                                perform(() => command("reset"));
                              }}
                            >
                              <RotateCcw size={16} />
                            </Button>

                            <Button
                              variant="ghost"
                              aria-label="Replay simulation"
                              disabled={busy || dirty}
                              onClick={() => {
                                setAutoplay(false);

                                perform(() => command("replay", 260));
                              }}
                            >
                              <RefreshCw size={15} />
                              Replay
                            </Button>
                          </div>

                          <div className="timeline">
                            <span>Inspect history</span>

                            <input
                              type="range"
                              aria-label="Simulation timeline"
                              min={0}
                              max={simulation!.frames.length - 1}
                              value={inspect}
                              onChange={(e) => setInspect(+e.target.value)}
                            />

                            <code>DAY {currentFrame.day}</code>
                          </div>
                        </>
                      ) : (
                        <Empty title="Initialize your virtual experiment">
                          Configure a patient and scenario, then initialize the
                          twin. All response curves are computed by the Python
                          simulator.
                        </Empty>
                      )}
                    </Card>

                    <Card className="scenario-card">
                      <div className="panel-heading">
                        <h3>Scenario configuration</h3>

                        <Atom size={17} className="teal" />
                      </div>

                      {patientSelect("Patient source")}

                      <Settings
                        scenario={scenario}
                        setScenario={updateScenario}
                        status={status}
                      />

                      {dirty && (
                        <p className="notice">
                          Draft parameters changed. Initialize to apply them;
                          the active simulation retains its saved configuration.
                        </p>
                      )}

                      <Button
                        className="w-full"
                        disabled={busy || !status}
                        onClick={create}
                      >
                        {busy ? (
                          <Loader2 size={14} className="animate-spin" />
                        ) : (
                          <Plus size={14} />
                        )}
                        Initialize twin
                      </Button>

                      <p className="caption">
                        Simulation day 0 begins at the historical calibration
                        cutoff for recorded patients. Defaults assume equal PSA
                        contribution and fixed competition and resistance
                        fraction. Changing the horizon or biology may move
                        beyond the PPO training distribution.
                      </p>
                    </Card>
                  </div>

                  {simulation && (
                    <>
                      <Card>
                        <div className="panel-heading">
                          <div>
                            <span className="eyebrow">
                              CONTROLLED COUNTERFACTUAL
                            </span>

                            <h3>Compare the next decision</h3>
                          </div>

                          <Button
                            variant="outline"
                            disabled={busy || simulation.done || dirty}
                            onClick={() =>
                              perform(async () =>
                                setBranches(
                                  await api(
                                    `/simulations/${simulation.id}/branches`,

                                    {},
                                  ),
                                ),
                              )
                            }
                          >
                            <GitBranch size={14} />
                            Branch ON vs OFF
                          </Button>
                        </div>

                        <p className="caption">
                          Both actions start from the current saved state and
                          identical RNG state. The active simulation is
                          unchanged. Safety rules may make executed actions
                          equal.
                        </p>

                        {branches && (
                          <div className="branch-results">
                            {branches.outcomes.map((o) => (
                              <div key={o.action}>
                                <Tag tone={o.action ? "teal" : "violet"}>
                                  Requested {o.action ? "ON" : "OFF"}
                                </Tag>

                                <strong>{fmt(o.frame.psa)} ng/mL</strong>

                                <small>
                                  Day {o.frame.day} · simulated next-state PSA
                                </small>
                              </div>
                            ))}
                          </div>
                        )}
                      </Card>

                      <Card>
                        <div className="panel-heading">
                          <h3>Synchronized simulation telemetry</h3>

                          <Tag tone="violet">Hypothetical</Tag>
                        </div>

                        {Array.isArray(simulation.calibration.warnings) && (
                          <ul className="caption">
                            {(simulation.calibration.warnings as string[]).map(
                              (w) => (
                                <li key={w}>{w}</li>
                              ),
                            )}
                          </ul>
                        )}
                        <SimulationCharts
                          frames={simulation.frames}
                          day={currentFrame?.day || 0}
                        />
                      </Card>
                    </>
                  )}
                </>
              )}

              {workspace === 2 && (
                <>
                  {simulation ? (
                    <>
                      <Card className="lab-context">
                        <div>
                          <span className="eyebrow">ACTIVE EXPERIMENT</span>

                          <h3>
                            {names[simulation.scenario.policy]} ·{" "}
                            {simulation.scenario.patient_id === null
                              ? "Synthetic"
                              : `Patient ${simulation.scenario.patient_id}`}
                          </h3>

                          <p>
                            Seed {simulation.scenario.seed} ·{" "}
                            {simulation.scenario.decision_days}-day intervals ·
                            force ON ≥ {simulation.scenario.safety_multiplier}×
                            starting PSA
                          </p>
                        </div>

                        <Button
                          disabled={
                            busy ||
                            simulation.done ||
                            simulation.paused ||
                            dirty
                          }
                          onClick={() => perform(() => command("step"))}
                        >
                          <SkipForward size={14} />
                          Next decision
                        </Button>
                      </Card>

                      <div className="lab-grid">
                        <Card>
                          <div className="panel-heading">
                            <h3>Decision inspector</h3>

                            <Tag tone="blue">Inference evidence</Tag>
                          </div>

                          {decision ? (
                            <DecisionEvidence
                              decision={decision}
                              schema={status?.observation_schema || []}
                            />
                          ) : (
                            <Empty title="Awaiting first decision">
                              Step the simulator to inspect observation,
                              selected action, executed action, reward, and
                              transition.
                            </Empty>
                          )}

                          {simulation.trace.length > 0 && (
                            <Field label="Inspect decision">
                              <input
                                aria-label="Decision timeline"
                                type="range"
                                min={1}
                                max={simulation.frames.length - 1}
                                value={Math.max(1, inspect)}
                                onChange={(e) => setInspect(+e.target.value)}
                              />
                            </Field>
                          )}
                        </Card>

                        <Card>
                          <div className="panel-heading">
                            <h3>Policy contract</h3>

                            <ShieldCheck size={17} />
                          </div>

                          <div className="contract">
                            <span>
                              Action space<b>0 = OFF / 1 = ON</b>
                            </span>

                            <span>
                              Observation dimension<b>4 · float32</b>
                            </span>

                            <span>
                              Latent cell state<b>Hidden from policy</b>
                            </span>

                            <span>
                              Inference mode<b>Deterministic</b>
                            </span>

                            <span>
                              Weight updates<b>Disabled in application</b>
                            </span>
                          </div>

                          <p className="caption">
                            Legacy reward warning: early termination removes
                            future negative rewards. A higher episode return
                            does not establish better disease control. The value
                            function is not displayed. PPO action probabilities
                            come directly from the saved categorical policy
                            distribution.
                          </p>

                          <details>
                            <summary>Exact observation preprocessing</summary>

                            <ol>
                              {status?.observation_schema.map((s) => (
                                <li key={s}>{s}</li>
                              ))}
                            </ol>
                          </details>
                        </Card>
                      </div>

                      <Card>
                        <div className="panel-heading">
                          <h3>Chronological decision history</h3>

                          <span className="caption">
                            Select a row to inspect
                          </span>
                        </div>

                        <div className="metrics-table-wrap">
                          <table>
                            <thead>
                              <tr>
                                <th>Decision</th>

                                <th>Selected</th>

                                <th>Executed</th>

                                <th>Reward</th>

                                <th>Safety override</th>

                                <th>Outcome</th>
                              </tr>
                            </thead>

                            <tbody>
                              {simulation.trace.map((t, i) => (
                                <tr
                                  key={i}
                                  className={
                                    inspect === i + 1 ? "selected" : ""
                                  }
                                >
                                  <td>
                                    <button
                                      className="patient-link"
                                      onClick={() => setInspect(i + 1)}
                                    >
                                      Day {t.day}
                                    </button>
                                  </td>

                                  <td>{t.selected_action ? "ON" : "OFF"}</td>

                                  <td>{t.executed_action ? "ON" : "OFF"}</td>

                                  <td>{fmt(t.reward, 3)}</td>

                                  <td>{t.safety_override ? "Yes" : "No"}</td>

                                  <td>
                                    {t.terminated
                                      ? "Modeled progression"
                                      : t.truncated
                                        ? "Horizon / censored"
                                        : "Continuing"}
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </Card>
                    </>
                  ) : (
                    <Card>
                      <Empty title="No active decision trace">
                        Initialize a twin and step the simulation to inspect
                        actual model decisions.
                      </Empty>

                      <div className="center">
                        <Button onClick={() => setWorkspace(1)}>
                          Open digital tumor twin <ArrowRight size={14} />
                        </Button>
                      </div>
                    </Card>
                  )}
                </>
              )}

              {workspace === 3 && (
                <>
                  <Card>
                    <div className="panel-heading">
                      <div>
                        <span className="eyebrow">
                          MATCHED VIRTUAL CONDITIONS
                        </span>

                        <h3>Strategy comparison</h3>
                      </div>

                      <Button
                        disabled={busy || experiment?.status === "running"}
                        onClick={() => runExperiment(true)}
                      >
                        <Play size={14} />
                        Run comparison
                      </Button>
                    </div>

                    <p className="caption">
                      Patient {scenario.patient_id ?? "synthetic"} · seed{" "}
                      {scenario.seed} · horizon {scenario.horizon_days} days ·
                      identical calibrated initial state and RNG seed for every
                      policy.
                    </p>

                    <PolicyToggles
                      selected={selectedPolicies}
                      setSelected={setSelectedPolicies}
                      status={status}
                    />

                    <label className="checkbox-label">
                      <input
                        type="checkbox"
                        checked={sensitivity}
                        onChange={(e) => setSensitivity(e.target.checked)}
                      />
                      Include growth, response and resistance sensitivity
                      (bounded to permitted range)
                    </label>

                    <div className="baseline-definitions">
                      <span>
                        <b>Continuous</b> ON each decision
                      </span>

                      <span>
                        <b>Intermittent</b> Stop at ≤50% starting PSA; restart
                        at baseline
                      </span>

                      <span>
                        <b>PPO adaptive</b> Checkpoint inference; fixed weights
                      </span>
                    </div>

                    <p className="caption">
                      The intermittent PSA rule is a literature-inspired
                      research baseline transferred from adaptive abiraterone
                      studies; it is not a reconstruction of the historical
                      cohort protocol. Safety limits are model constraints, not
                      validated clinical safety rules.
                    </p>

                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setWorkspace(1)}
                    >
                      Adjust patient and scenario <ArrowRight size={13} />
                    </Button>
                  </Card>

                  <Card>
                    {experiment ? (
                      <ExperimentPanel
                        experiment={experiment}
                        setExperiment={setExperiment}
                        refresh={refreshExperiments}
                        fail={fail}
                      />
                    ) : (
                      <Empty title="Design a fair comparison">
                        Select policies and run a small local comparison. No
                        policies are substituted if PPO is unavailable.
                      </Empty>
                    )}
                  </Card>
                </>
              )}

              {workspace === 4 && (
                <div className="experiment-layout">
                  <div className="stack">
                    <Card>
                      <div className="panel-heading">
                        <div>
                          <span className="eyebrow">EXPERIMENT BUILDER</span>

                          <h3>Ask a reproducible question</h3>
                        </div>

                        <FlaskConical size={18} />
                      </div>

                      <Field label="Experiment name">
                        <input
                          aria-label="Experiment name"
                          maxLength={80}
                          value={experimentName}
                          onChange={(e) => setExperimentName(e.target.value)}
                        />
                      </Field>

                      {patientSelect("Experiment patient")}

                      <Settings
                        scenario={scenario}
                        setScenario={updateScenario}
                        status={status}
                      />

                      <h4>Comparison policies</h4>

                      <PolicyToggles
                        selected={selectedPolicies}
                        setSelected={setSelectedPolicies}
                        status={status}
                      />

                      <Field
                        label="Random seeds"
                        hint="One to five comma-separated integers"
                      >
                        <input
                          aria-label="Experiment seeds"
                          value={seedText}
                          onChange={(e) => setSeedText(e.target.value)}
                        />
                      </Field>

                      <label className="checkbox-label">
                        <input
                          type="checkbox"
                          checked={sensitivity}
                          onChange={(e) => setSensitivity(e.target.checked)}
                        />
                        Include growth, treatment-response and resistance
                        sensitivity
                      </label>

                      <Button
                        className="w-full mt-4"
                        disabled={
                          busy ||
                          !experimentName.trim() ||
                          experiment?.status === "running"
                        }
                        onClick={() => runExperiment()}
                      >
                        <Play size={14} />
                        Save & run experiment
                      </Button>

                      <p className="caption">
                        Runs in a local worker. Results persist in
                        results/studio. Training remains a separate terminal
                        operation.
                      </p>
                    </Card>

                    <Card>
                      <div className="panel-heading">
                        <h3>Experiment library</h3>

                        <Button
                          variant="ghost"
                          size="icon"
                          aria-label="Refresh experiments"
                          onClick={refreshExperiments}
                        >
                          <RefreshCw size={15} />
                        </Button>
                      </div>

                      <ExperimentLibrary
                        experiments={experiments}
                        selected={experiment?.id}
                        open={openExperiment}
                      />
                    </Card>
                  </div>

                  <Card className="experiment-output">
                    {experiment ? (
                      <ExperimentPanel
                        experiment={experiment}
                        setExperiment={setExperiment}
                        refresh={refreshExperiments}
                        fail={fail}
                      />
                    ) : (
                      <Empty title="A durable research record">
                        Save a named experiment to compare policies, inspect
                        traces, export results, and reproduce the exact
                        configuration.
                      </Empty>
                    )}
                  </Card>
                </div>
              )}
            </>
          )}

          <footer>
            <div>
              <Atom size={13} /> ONCOEVO RESEARCH STUDIO <span>·</span> Local
              computational oncology
            </div>

            <span>
              Recorded evidence. Explicit assumptions. Reproducible simulation.
            </span>
          </footer>
        </main>
      </div>
    </div>
  );
}
