import type { Summary } from "../lib/api";
import {
  count,
  intrinsicsLabel,
  milliseconds,
  pixels,
  ratio,
  seconds,
  timingLabel,
} from "../lib/format";

/** Instrument panel. Hairline-separated rows rather than cards, mono tabular
 *  numerals, and no decoration: this is a readout, not a marketing surface. */
export function Telemetry({ summary }: { summary: Summary }) {
  const stages = Object.entries(summary.timings_ms)
    .filter(([key, value]) => key !== "total" && value > 0.5)
    .sort((a, b) => b[1] - a[1]);
  const total = summary.timings_ms.total ?? 1;

  return (
    <div className="flex h-full flex-col overflow-y-auto">
      <Section title="Throughput">
        <Row label="Video length" value={seconds(summary.video_duration_s)} />
        <Row label="Processing time" value={seconds(summary.processing_time_s)} />
        <Row
          label="Realtime factor"
          value={ratio(summary.realtime_factor)}
          note={summary.within_budget ? "within budget" : "over budget"}
          emphasis={summary.within_budget ? "good" : "warn"}
        />
      </Section>

      <Section title="Reconstruction">
        <Row label="Frames" value={count(summary.frames)} />
        <Row label="Keyframes" value={count(summary.keyframes)} />
        <Row label="Landmarks" value={count(summary.landmarks)} />
        <Row label="Mean reprojection error" value={pixels(summary.mean_reproj_error_px)} />
        <Row label="Median tracks per frame" value={count(summary.median_track_count)} />
      </Section>

      <Section title="Drift correction">
        <Row label="Loop closures found" value={count(summary.loop_closures)} />
        <Row
          label="Correction applied"
          value={summary.drift_correction_applied ? "Yes" : "No"}
          emphasis={summary.drift_correction_applied ? "good" : undefined}
        />
        {summary.drift_correction_applied && (
          <Row
            label="Largest pose shift"
            value={summary.max_pose_correction.toFixed(3)}
            note="slam units"
          />
        )}
        <Row label="Tracking losses" value={count(summary.tracking_losses)} />
      </Section>

      <Section title="Stage timings">
        {stages.map(([stage, ms]) => (
          <div key={stage} className="border-t border-ink-800 py-2 first:border-t-0">
            <div className="flex items-baseline justify-between gap-4">
              <span className="text-xs text-ink-400">{timingLabel(stage)}</span>
              <span className="tnum text-xs text-ink-200">{milliseconds(ms)}</span>
            </div>
            <div className="mt-1.5 h-px w-full bg-ink-800">
              <div
                className="h-px bg-ink-500"
                style={{ width: `${Math.max((ms / total) * 100, 1)}%` }}
              />
            </div>
          </div>
        ))}
      </Section>

      <Section title="Input assumptions">
        <Row label="Camera intrinsics" value={intrinsicsLabel(summary.intrinsics_source)} />
        <Row label="Processing width" value={`${summary.applied_width}px`} />
        <Row label="Feature budget" value={count(summary.applied_max_features)} />
        {summary.quality_reduced && (
          <Row label="Quality reduced" value="Yes" note="to hold the time budget" />
        )}
      </Section>

      <p className="mt-auto px-5 py-5 text-xs leading-relaxed text-ink-500">
        A single camera cannot recover absolute scale, so distances are in consistent but arbitrary
        slam units rather than metres.
      </p>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="border-b border-ink-800 px-5 py-4">
      <h3 className="mb-2 text-xs tracking-wide text-ink-500">{title}</h3>
      <div>{children}</div>
    </section>
  );
}

function Row({
  label,
  value,
  note,
  emphasis,
}: {
  label: string;
  value: string;
  note?: string;
  emphasis?: "good" | "warn";
}) {
  const tone =
    emphasis === "good" ? "text-ink-100" : emphasis === "warn" ? "text-signal" : "text-ink-200";
  return (
    <div className="flex items-baseline justify-between gap-4 border-t border-ink-800 py-2 first:border-t-0">
      <span className="text-xs text-ink-400">{label}</span>
      <span className="text-right">
        <span className={`tnum text-sm ${tone}`}>{value}</span>
        {note && <span className="ml-2 text-xs text-ink-500">{note}</span>}
      </span>
    </div>
  );
}
