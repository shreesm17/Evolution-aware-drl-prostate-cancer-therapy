import { useEffect, useRef } from "react";

import * as echarts from "echarts/core";

import { LineChart, BarChart, CustomChart } from "echarts/charts";

import {
  GridComponent,
  TooltipComponent,
  LegendComponent,
  MarkAreaComponent,
  MarkLineComponent,
  AriaComponent,
} from "echarts/components";

import { CanvasRenderer } from "echarts/renderers";

echarts.use([
  LineChart,

  BarChart,

  CustomChart,

  GridComponent,

  TooltipComponent,

  LegendComponent,

  MarkAreaComponent,

  MarkLineComponent,

  AriaComponent,

  CanvasRenderer,
]);

import type { EChartsOption } from "echarts";

import type { Frame, History, Run } from "./api";

import { colors, names } from "./api";

export function Chart({
  option,

  height = 260,

  label,

  group = "studio",
}: {
  option: EChartsOption;

  height?: number;

  label: string;

  group?: string;
}) {
  const el = useRef<HTMLDivElement>(null);

  const chart = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    const c = echarts.init(el.current!, undefined, { renderer: "canvas" });

    chart.current = c;

    c.group = group;

    echarts.connect(group);

    const ro = new ResizeObserver(() => c.resize());

    ro.observe(el.current!);

    return () => {
      ro.disconnect();

      c.dispose();
    };
  }, [group]);

  useEffect(() => {
    chart.current?.setOption(
      {
        backgroundColor: "transparent",

        animationDuration: 350,

        textStyle: {
          color: "#a9b8cd",

          fontFamily: "Inter, Segoe UI, sans-serif",
        },

        aria: { enabled: true },

        grid: { left: 50, right: 22, top: 42, bottom: 40 },

        tooltip: {
          trigger: "axis",

          backgroundColor: "#142136",

          borderColor: "#344961",

          textStyle: { color: "#e1eaf5" },
        },

        legend: {
          top: 2,

          textStyle: { color: "#adbed2" },

          itemWidth: 14,

          itemHeight: 3,
        },

        ...option,
      },

      true,
    );
  }, [option]);

  return (
    <div
      ref={el}
      role="img"
      aria-label={label}
      style={{ height, width: "100%" }}
    />
  );
}

const axis = {
  type: "value" as const,

  axisLine: { lineStyle: { color: "#29374d" } },

  splitLine: { lineStyle: { color: "#1b293d" } },

  axisLabel: { color: "#8e9fb6", fontSize: 10 },

  nameTextStyle: { color: "#8e9fb6" },
};

export function timeseries(
  series: {
    name: string;

    color: string;

    points: number[][];

    area?: boolean;

    step?: boolean;
  }[],

  yName: string,

  inspect?: number,
): EChartsOption {
  return {
    xAxis: {
      ...axis,

      name: "Simulated days",

      nameLocation: "middle",

      nameGap: 27,
    },

    yAxis: { ...axis, name: yName },

    series: series.map((s, i) => ({
      name: s.name,

      type: "line",

      showSymbol: s.points.length <= 1,

      smooth: false,

      step: s.step ? "end" : undefined,

      lineStyle: { width: 2, color: s.color },

      itemStyle: { color: s.color },

      areaStyle: s.area ? { color: s.color, opacity: 0.07 } : undefined,

      data: s.points,

      markLine:
        i === 0 && inspect !== undefined
          ? {
              symbol: "none",

              silent: true,

              label: { show: false },

              lineStyle: { color: "#657d97", type: "dashed" },

              data: [{ xAxis: inspect }],
            }
          : undefined,
    })),
  };
}

export function ObservedChart({ history }: { history: History }) {
  const obs = history.observations;

  return (
    <Chart
      label="Recorded PSA and calibrated model forecast"
      group="observed"
      height={320}
      option={{
        xAxis: { ...axis, name: "Recorded days" },

        yAxis: { ...axis, name: "PSA · ng/mL" },

        series: [
          {
            name: "Recorded observations",

            type: "line",

            showSymbol: true,

            symbolSize: 5,

            data: obs.map((p) => [p.day, p.psa_ng_ml]),

            lineStyle: { color: "#5ce2ca", width: 2 },

            itemStyle: { color: "#5ce2ca" },

            markArea: {
              silent: true,

              itemStyle: { color: "rgba(180,154,247,.07)" },

              data: obs.flatMap((p, i) =>
                p.treatment_on && obs[i + 1]
                  ? [[{ xAxis: p.day }, { xAxis: obs[i + 1].day }]]
                  : [],
              ),
            },

            markLine: {
              silent: true,

              symbol: "none",

              label: { formatter: "Calibration cutoff", color: "#b49af7" },

              lineStyle: { color: "#b49af7", type: "dashed" },

              data: [{ xAxis: history.calibration.cutoff_day }],
            },
          },

          {
            name: "Calibrated estimate / forecast",

            type: "line",

            showSymbol: false,

            data: history.forecast.map((p) => [p.day, p.psa]),

            lineStyle: { color: "#7397ff", type: "dashed", width: 1.5 },
          },
        ],
      }}
    />
  );
}

export function SimulationCharts({
  frames,

  day,
}: {
  frames: Frame[];

  day: number;
}) {
  return (
    <div className="chart-grid">
      <div>
        <h4>
          PSA trajectory <span>ng/mL · simulated</span>
        </h4>

        <Chart
          label="Simulated PSA and noisy observation"
          option={timeseries(
            [
              {
                name: "Model PSA",

                color: "#5ce2ca",

                points: frames.map((f) => [f.day, f.psa]),

                area: true,
              },

              {
                name: "Policy observation",

                color: "#7397ff",

                points: frames.map((f) => [f.day, f.observed_psa]),
              },
            ],

            "PSA",

            day,
          )}
        />
      </div>

      <div>
        <h4>
          Population dynamics <span>PSA-scaled burden</span>
        </h4>

        <Chart
          label="Latent sensitive and resistant populations"
          option={timeseries(
            [
              {
                name: "Sensitive · latent",

                color: "#5ce2ca",

                points: frames.map((f) => [f.day, f.sensitive]),

                area: true,
              },

              {
                name: "Resistant · latent",

                color: "#b49af7",

                points: frames.map((f) => [f.day, f.resistant]),

                area: true,
              },
            ],

            "Burden",

            day,
          )}
        />
      </div>

      <div>
        <h4>
          Treatment exposure <span>executed interval action</span>
        </h4>

        <Chart
          label="Treatment ON and OFF intervals"
          height={180}
          option={timeseries(
            [
              {
                name: "ON = 1 / OFF = 0",

                color: "#7397ff",

                points: frames.slice(1).flatMap((f, i) => [
                  [frames[i].day, f.treatment],

                  [f.day, f.treatment],
                ]),

                area: true,

                step: true,
              },
            ],

            "Action",

            day,
          )}
        />
      </div>

      <div>
        <h4>
          Decision reward <span>configured objective</span>
        </h4>

        <Chart
          label="Per decision calculated reward"
          height={180}
          option={timeseries(
            [
              {
                name: "Reward",

                color: "#b49af7",

                points: frames.slice(1).map((f) => [f.day, f.reward]),
              },
            ],

            "Reward",

            day,
          )}
        />
      </div>
    </div>
  );
}

export function ArenaCharts({ runs }: { runs: Run[] }) {
  const nominal = runs.filter(
    (r) =>
      (r.nominal ?? r.sensitivity_factor === 1) && r.seed === runs[0]?.seed,
  );

  const option = timeseries(
    nominal.map((r) => ({
      name: names[r.policy],

      color: colors[r.policy],

      points: r.frames.map((f) => [f.day, f.psa]),
    })),

    "PSA · ng/mL",
  );

  return (
    <>
      <Chart
        label="Matched policy PSA comparison"
        height={340}
        option={option}
      />

      <div className="chart-grid">
        <Chart
          label="Cumulative simulated treatment exposure"
          option={timeseries(
            nominal.map((r) => ({
              name: names[r.policy],

              color: colors[r.policy],

              points: r.frames.map((f) => [f.day, f.exposure_days]),
            })),

            "Days ON",
          )}
        />

        <Chart
          label="Matched policy treatment schedules"
          option={{
            xAxis: { ...axis, name: "Simulated days" },

            yAxis: {
              type: "category",

              data: nominal.map((r) => names[r.policy]),

              axisLabel: { color: "#c9d6e9", fontSize: 10 },
            },

            grid: { left: 130, right: 20, top: 20, bottom: 40 },

            series: nominal.map((r, ri) => ({
              name: names[r.policy],

              type: "custom",

              renderItem: (_params, api) => {
                const a = api.coord([api.value(0), api.value(2)]),
                  b = api.coord([api.value(1), api.value(2)]);

                return {
                  type: "rect",

                  shape: {
                    x: a[0],

                    y: a[1] - 10,

                    width: b[0] - a[0],

                    height: 20,
                  },

                  style: { fill: colors[r.policy], opacity: 0.8 },
                };
              },

              data: r.frames
                .slice(1)
                .map((f, i) => ({ frame: f, start: r.frames[i].day }))
                .filter(({ frame: f }) => f.treatment === 1)
                .map(({ frame: f, start }) => [start, f.day, ri]),
            })),
          }}
        />
      </div>
    </>
  );
}
