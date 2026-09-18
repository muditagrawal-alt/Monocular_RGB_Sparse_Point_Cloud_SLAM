/** Typed client for the SLAM service. */

export type JobState = "queued" | "running" | "completed" | "failed";

export interface JobProgress {
  stage: string;
  fraction: number;
  frames: number;
  keyframes: number;
  landmarks: number;
}

export interface JobStatus {
  job_id: string;
  state: JobState;
  progress: JobProgress;
  error: string | null;
  elapsed_s: number;
  summary: Summary | null;
}

export interface Summary {
  frames: number;
  keyframes: number;
  landmarks: number;
  loop_closures: number;
  tracking_losses: number;
  reinitializations: number;
  mean_reproj_error_px: number;
  median_track_count: number;
  processing_time_s: number;
  video_duration_s: number;
  realtime_factor: number;
  within_budget: boolean;
  drift_correction_applied: boolean;
  max_pose_correction: number;
  intrinsics_source: "user" | "metadata" | "fov_heuristic" | null;
  applied_width: number;
  applied_max_features: number;
  quality_reduced: boolean;
  timings_ms: Record<string, number>;
}

export interface TrajectoryPayload {
  positions: number[];
  count: number;
  keyframe_indices: number[];
}

export interface ReconstructionResult {
  success: boolean;
  reason: string;
  summary: Summary;
  cloud: { positions: number[]; colors: number[]; count: number; truncated: boolean };
  trajectory: TrajectoryPayload;
  odometry_trajectory: TrajectoryPayload;
  keyframes: { id: number; frame_index: number; position: number[]; rotation: number[] }[];
  camera: {
    fx: number; fy: number; cx: number; cy: number;
    width: number; height: number;
    source: "user" | "metadata" | "fov_heuristic";
  } | null;
}

export interface Limits {
  max_upload_bytes: number;
  max_duration_s: number;
  accepted_types: string[];
}

async function parseError(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
  } catch {
    /* fall through to the status text */
  }
  return `${response.status} ${response.statusText}`;
}

export async function fetchLimits(): Promise<Limits> {
  const response = await fetch("/api/limits");
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}

export async function submitVideo(
  file: File,
  options: { hfovDeg?: number } = {},
): Promise<string> {
  const form = new FormData();
  form.append("video", file);
  if (options.hfovDeg !== undefined) form.append("hfov_deg", String(options.hfovDeg));

  const response = await fetch("/api/jobs", { method: "POST", body: form });
  if (!response.ok) throw new Error(await parseError(response));
  return (await response.json()).job_id as string;
}

export async function fetchStatus(jobId: string): Promise<JobStatus> {
  const response = await fetch(`/api/jobs/${jobId}`);
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}

export async function fetchResult(jobId: string): Promise<ReconstructionResult> {
  const response = await fetch(`/api/jobs/${jobId}/result`);
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}
