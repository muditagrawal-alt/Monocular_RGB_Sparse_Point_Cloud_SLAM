import { useCallback, useEffect, useRef, useState } from "react";
import {
  fetchLimits,
  fetchResult,
  fetchStatus,
  submitVideo,
  type JobStatus,
  type Limits,
  type ReconstructionResult,
} from "./lib/api";
import { Upload } from "./components/Upload";
import { Progress } from "./components/Progress";
import { Viewer } from "./components/Viewer";

type Phase = "idle" | "working" | "done" | "error";

const POLL_INTERVAL_MS = 400;

export default function App() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [limits, setLimits] = useState<Limits | null>(null);
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [result, setResult] = useState<ReconstructionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    fetchLimits().then(setLimits).catch(() => setLimits(null));
    return () => {
      if (timer.current !== null) window.clearTimeout(timer.current);
    };
  }, []);

  const poll = useCallback((jobId: string) => {
    const tick = async () => {
      try {
        const next = await fetchStatus(jobId);
        setStatus(next);
        if (next.state === "completed") {
          setResult(await fetchResult(jobId));
          setPhase("done");
          return;
        }
        if (next.state === "failed") {
          setError(next.error ?? "Reconstruction failed.");
          setPhase("error");
          return;
        }
        timer.current = window.setTimeout(tick, POLL_INTERVAL_MS);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Lost contact with the server.");
        setPhase("error");
      }
    };
    tick();
  }, []);

  const start = useCallback(
    async (file: File, hfovDeg?: number) => {
      setError(null);
      setResult(null);
      setStatus(null);
      setPhase("working");
      try {
        poll(await submitVideo(file, { hfovDeg }));
      } catch (err) {
        setError(err instanceof Error ? err.message : "Upload failed.");
        setPhase("error");
      }
    },
    [poll],
  );

  const reset = useCallback(() => {
    if (timer.current !== null) window.clearTimeout(timer.current);
    setPhase("idle");
    setStatus(null);
    setResult(null);
    setError(null);
  }, []);

  return (
    <div className="flex h-full min-w-0 flex-col overflow-x-hidden bg-ink-900">
      <header className="flex shrink-0 items-center justify-between border-b border-ink-800 px-6 py-4">
        <div className="flex items-baseline gap-3">
          <h1 className="text-sm font-medium tracking-tight text-ink-100">Sparse SLAM</h1>
          <p className="hidden text-xs text-ink-500 sm:block">
            Monocular RGB reconstruction
          </p>
        </div>
        <p className="tnum text-xs text-ink-500">CPU only, no GPU</p>
      </header>

      <main className="min-h-0 min-w-0 flex-1 overflow-hidden">
        {phase === "done" && result ? (
          <Viewer result={result} onReset={reset} />
        ) : (
          <div className="flex h-full items-center justify-center px-6 py-12">
            <div className="w-full">
              {(phase === "idle" || phase === "error") && (
                <>
                  <div className="mx-auto mb-12 max-w-2xl">
                    <h2 className="text-2xl leading-tight tracking-tight text-ink-100 sm:text-3xl">
                      Upload. Track. Reconstruct.
                    </h2>
                    <p className="mt-3 max-w-[58ch] text-sm leading-relaxed text-ink-400">
                      Recover the camera path and a sparse 3D point cloud from a single-lens video,
                      with accumulated drift corrected wherever the camera revisits a place.
                    </p>
                  </div>
                  <Upload
                    limits={limits}
                    onSubmit={start}
                    error={phase === "error" ? error : null}
                  />
                </>
              )}

              {phase === "working" && status && <Progress status={status} />}
              {phase === "working" && !status && (
                <p className="text-center text-sm text-ink-400">Uploading</p>
              )}

              {phase === "error" && (
                <div className="mx-auto mt-8 max-w-2xl text-center">
                  <button
                    type="button"
                    onClick={reset}
                    className="text-xs text-ink-400 underline underline-offset-4 hover:text-ink-200"
                  >
                    Start over
                  </button>
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
