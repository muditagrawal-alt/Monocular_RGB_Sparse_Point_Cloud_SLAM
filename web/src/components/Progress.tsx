import type { JobStatus } from "../lib/api";
import { count, seconds, stageLabel } from "../lib/format";

/** Live pipeline state. Shows the stage, the counts as they accumulate, and
 *  elapsed time, because elapsed time against video length is the number this
 *  project is judged on. */
export function Progress({ status }: { status: JobStatus }) {
  const pct = Math.round(status.progress.fraction * 100);

  return (
    <div className="mx-auto w-full max-w-2xl">
      <div className="flex items-baseline justify-between">
        <p className="text-sm text-ink-200">{stageLabel(status.progress.stage)}</p>
        <p className="tnum text-sm text-ink-300">{pct}%</p>
      </div>

      <div
        role="progressbar"
        aria-valuenow={pct}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label="Reconstruction progress"
        className="mt-3 h-px w-full bg-ink-700"
      >
        <div
          className="h-px bg-ink-100 transition-[width] duration-300 ease-out"
          style={{ width: `${Math.max(pct, 2)}%` }}
        />
      </div>

      <dl className="mt-8 grid grid-cols-2 gap-x-8 gap-y-4 sm:grid-cols-4">
        <Stat label="Elapsed" value={seconds(status.elapsed_s)} />
        <Stat label="Frames" value={count(status.progress.frames)} />
        <Stat label="Keyframes" value={count(status.progress.keyframes)} />
        <Stat label="Landmarks" value={count(status.progress.landmarks)} />
      </dl>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs text-ink-500">{label}</dt>
      <dd className="tnum mt-1 text-lg text-ink-100">{value}</dd>
    </div>
  );
}
