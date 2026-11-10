import { useEffect, useRef } from "react";
import type { Frame } from "./api";
export default function Cells({
  frame,
  playing,
}: {
  frame: Frame;
  playing: boolean;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const state = useRef({ frame, playing });
  state.current = { frame, playing };
  useEffect(() => {
    const el = canvas.current!;
    const ctx = el.getContext("2d")!;
    let raf = 0;
    let clock = 0;
    const draw = () => {
      const bounds = el.getBoundingClientRect();
      const dpr = devicePixelRatio || 1;
      if (
        el.width !== Math.round(bounds.width * dpr) ||
        el.height !== Math.round(bounds.height * dpr)
      ) {
        el.width = Math.round(bounds.width * dpr);
        el.height = Math.round(bounds.height * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const w = bounds.width,
        h = bounds.height;
      ctx.clearRect(0, 0, w, h);
      const { frame: f, playing: p } = state.current;
      if (p && !matchMedia("(prefers-reduced-motion: reduce)").matches)
        clock += 0.005;
      const total = f.sensitive + f.resistant;
      const fraction = total ? f.resistant / total : 0;
      const spread = Math.min(1, total / 40);
      ctx.strokeStyle = "#193449";
      for (let i = 1; i < 4; i++) {
        ctx.beginPath();
        ctx.ellipse(
          w / 2,
          h / 2,
          w * 0.12 * i,
          h * 0.12 * i,
          0,
          0,
          Math.PI * 2,
        );
        ctx.stroke();
      }
      for (let i = 0; i < 150; i++) {
        const resistant = i / 150 < fraction;
        const angle = i * 2.39996 + Math.sin(i) * clock * 0.07;
        const radius = Math.sqrt((i + 0.5) / 150);
        const x =
          w / 2 +
          Math.cos(angle) * radius * w * (0.24 + 0.15 * spread) +
          Math.sin(clock + i) * 3;
        const y =
          h / 2 +
          Math.sin(angle) * radius * h * (0.25 + 0.15 * spread) +
          Math.cos(clock + i) * 3;
        const r = 2 + Math.min(4, total / 12) + Math.sin(i) * 0.6;
        ctx.shadowColor = resistant ? "#a394fb" : "#4dddc2";
        ctx.shadowBlur = 8;
        ctx.fillStyle = resistant
          ? "rgba(180,154,247,.8)"
          : "rgba(81,218,192,.66)";
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.fillStyle = resistant ? "#dfd2ff" : "#bcfff1";
        ctx.beginPath();
        ctx.arc(x - r * 0.2, y - r * 0.2, r * 0.3, 0, Math.PI * 2);
        ctx.fill();
      }
      raf = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(raf);
  }, []);
  return (
    <div className="cell-field">
      <div className="cell-label">
        <span className="live-dot" /> LATENT POPULATION FIELD{" "}
        <span className="mono">DAY {frame.day}</span>
      </div>
      <canvas
        ref={canvas}
        aria-label="Abstract population animation driven by simulated burden; particles are not measured cells"
      />
      <div className="cell-legend">
        <span>
          <i style={{ background: "#5ce2ca" }} />
          Sensitive {frame.sensitive.toFixed(2)}
        </span>
        <span>
          <i style={{ background: "#b49af7" }} />
          Resistant {frame.resistant.toFixed(2)}
        </span>
      </div>
      <p>
        Abstract particles · simulated PSA-scaled burden · not measured cells
      </p>
    </div>
  );
}
