/** Formatting helpers. Every number the UI shows passes through here so units
 *  and precision stay consistent across the telemetry panel. */

export function seconds(value: number): string {
  if (!Number.isFinite(value)) return "-";
  return value < 10 ? `${value.toFixed(2)}s` : `${value.toFixed(1)}s`;
}

export function milliseconds(value: number): string {
  if (!Number.isFinite(value)) return "-";
  if (value >= 1000) return `${(value / 1000).toFixed(2)}s`;
  return `${value.toFixed(0)}ms`;
}

export function count(value: number): string {
  if (!Number.isFinite(value)) return "-";
  return value >= 10000 ? `${(value / 1000).toFixed(1)}k` : value.toLocaleString("en-US");
}

export function pixels(value: number): string {
  return Number.isFinite(value) ? `${value.toFixed(2)}px` : "-";
}

export function ratio(value: number): string {
  return Number.isFinite(value) ? `${value.toFixed(2)}x` : "-";
}

export function bytes(value: number): string {
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)} MB`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(0)} KB`;
  return `${value} B`;
}

const STAGE_LABELS: Record<string, string> = {
  queued: "Queued",
  decoding: "Decoding video",
  initialized: "Map initialised",
  tracking: "Tracking and mapping",
  loop_closure: "Searching for revisits",
  pose_graph: "Correcting drift",
  done: "Complete",
};

export function stageLabel(stage: string): string {
  return STAGE_LABELS[stage] ?? stage.replace(/_/g, " ");
}

const INTRINSICS_LABELS: Record<string, string> = {
  user: "Supplied by you",
  metadata: "Read from file metadata",
  fov_heuristic: "Assumed 60 degree field of view",
};

export function intrinsicsLabel(source: string | null): string {
  return source ? (INTRINSICS_LABELS[source] ?? source) : "Unknown";
}

/** Explicit stage labels. CSS `capitalize` renders "local_ba" as "Local Ba",
 *  which is wrong for initialisms. */
const TIMING_LABELS: Record<string, string> = {
  decode: "Decode",
  frontend: "Front end",
  initialization: "Initialisation",
  tracking: "Tracking",
  mapping: "Mapping",
  descriptors: "Descriptors",
  local_ba: "Local BA",
  loop_closure: "Loop closure",
  pose_graph: "Pose graph",
  export: "Export",
};

export function timingLabel(stage: string): string {
  return TIMING_LABELS[stage] ?? stage.replace(/_/g, " ");
}
