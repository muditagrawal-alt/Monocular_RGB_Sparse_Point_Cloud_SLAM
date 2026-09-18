import { Suspense, lazy, useState } from "react";
import { ArrowCounterClockwise, DownloadSimple } from "@phosphor-icons/react";
import type { ReconstructionResult } from "../lib/api";
import { Telemetry } from "./Telemetry";
import { count } from "../lib/format";

// three.js is a large dependency and is only needed once a result exists.
const Scene = lazy(() => import("./Scene").then((m) => ({ default: m.Scene })));

interface ViewerProps {
  result: ReconstructionResult;
  onReset: () => void;
}

export function Viewer({ result, onReset }: ViewerProps) {
  const [showOdometry, setShowOdometry] = useState(result.summary.drift_correction_applied);
  const [showCloud, setShowCloud] = useState(true);
  const [showKeyframes, setShowKeyframes] = useState(true);
  const [pointSize, setPointSize] = useState(0.018);

  const corrected = result.summary.drift_correction_applied;

  return (
    <div className="grid h-full min-w-0 grid-rows-[1fr_auto] overflow-hidden
                    lg:grid-cols-[minmax(0,1fr)_320px] lg:grid-rows-1">
      <div className="relative min-h-[55vh] min-w-0 overflow-hidden bg-ink-900">
        <Suspense
          fallback={
            <div className="flex h-full items-center justify-center text-sm text-ink-500">
              Loading viewer
            </div>
          }
        >
          <Scene
            result={result}
            showOdometry={showOdometry && corrected}
            showCloud={showCloud}
            showKeyframes={showKeyframes}
            pointSize={pointSize}
          />
        </Suspense>

        {/* Overlay controls. Kept to one corner so the scene stays the subject. */}
        <div className="pointer-events-none absolute inset-x-0 top-0 flex flex-wrap items-start
                        justify-between gap-2 p-3 sm:p-4">
          <div className="pointer-events-auto flex flex-wrap items-center gap-1.5">
            <Toggle active={showCloud} onClick={() => setShowCloud((v) => !v)}>
              Point cloud
            </Toggle>
            <Toggle active={showKeyframes} onClick={() => setShowKeyframes((v) => !v)}>
              Keyframes
            </Toggle>
            {corrected && (
              <Toggle active={showOdometry} onClick={() => setShowOdometry((v) => !v)}>
                Before correction
              </Toggle>
            )}
          </div>

          <div className="pointer-events-auto flex items-center gap-1.5">
            <button
              type="button"
              onClick={onReset}
              className="flex items-center gap-1.5 rounded-[--radius-panel] border border-ink-700
                         bg-ink-900/90 px-3 py-1.5 text-xs text-ink-200 backdrop-blur
                         transition-colors hover:border-ink-500 hover:text-ink-100"
            >
              <ArrowCounterClockwise size={13} weight="light" />
              New video
            </button>
          </div>
        </div>

        {/* Legend. Only present when there are two trajectories to distinguish. */}
        {corrected && showOdometry && (
          <div
            className="pointer-events-none absolute bottom-4 left-4 flex flex-col gap-1.5
                       rounded-[--radius-panel] border border-ink-800 bg-ink-900/90 px-3 py-2
                       backdrop-blur"
          >
            <LegendRow color="#fafafa" label="After loop closure" />
            <LegendRow color="#d08a2c" label="Odometry only" />
          </div>
        )}

        <div className="pointer-events-none absolute bottom-4 right-4 flex items-center gap-3">
          <label className="pointer-events-auto flex items-center gap-2 rounded-[--radius-panel]
                            border border-ink-800 bg-ink-900/90 px-3 py-1.5 backdrop-blur">
            <span className="text-xs text-ink-400">Point size</span>
            <input
              type="range"
              min={0.004}
              max={0.05}
              step={0.002}
              value={pointSize}
              onChange={(e) => setPointSize(Number(e.target.value))}
              aria-label="Point size"
              className="h-1 w-20 accent-ink-100"
            />
          </label>
        </div>
      </div>

      <aside className="min-w-0 border-t border-ink-800 bg-ink-850 lg:border-l lg:border-t-0">
        <header className="flex items-center justify-between border-b border-ink-800 px-5 py-4">
          <div>
            <h2 className="text-sm text-ink-100">Telemetry</h2>
            <p className="tnum mt-0.5 text-xs text-ink-500">
              {count(result.cloud.count)} points shown
              {result.cloud.truncated ? " (subsampled)" : ""}
            </p>
          </div>
          <a
            href={`data:application/json;charset=utf-8,${encodeURIComponent(
              JSON.stringify(result.summary, null, 2),
            )}`}
            download="slam-telemetry.json"
            className="flex items-center gap-1.5 rounded-[--radius-panel] border border-ink-700
                       px-2.5 py-1.5 text-xs text-ink-300 transition-colors hover:border-ink-500
                       hover:text-ink-100"
          >
            <DownloadSimple size={13} weight="light" />
            JSON
          </a>
        </header>
        <Telemetry summary={result.summary} />
      </aside>
    </div>
  );
}

function Toggle({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={[
        "rounded-[--radius-panel] border px-3 py-1.5 text-xs backdrop-blur transition-colors",
        active
          ? "border-ink-500 bg-ink-800/90 text-ink-100"
          : "border-ink-800 bg-ink-900/90 text-ink-400 hover:text-ink-200",
      ].join(" ")}
    >
      {children}
    </button>
  );
}

function LegendRow({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-2">
      <span className="h-px w-5" style={{ backgroundColor: color }} />
      <span className="text-xs text-ink-300">{label}</span>
    </span>
  );
}
