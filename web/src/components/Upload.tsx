import { useCallback, useRef, useState } from "react";
import { UploadSimple, Warning } from "@phosphor-icons/react";
import type { Limits } from "../lib/api";
import { bytes } from "../lib/format";

interface UploadProps {
  limits: Limits | null;
  onSubmit: (file: File, hfovDeg?: number) => void;
  disabled?: boolean;
  error?: string | null;
}

export function Upload({ limits, onSubmit, disabled, error }: UploadProps) {
  const [dragging, setDragging] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const [hfov, setHfov] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const validate = useCallback(
    (file: File): string | null => {
      const suffix = `.${file.name.split(".").pop()?.toLowerCase() ?? ""}`;
      if (limits && !limits.accepted_types.includes(suffix)) {
        return `${suffix} is not a supported video format.`;
      }
      if (limits && file.size > limits.max_upload_bytes) {
        return `File is ${bytes(file.size)}. The limit is ${bytes(limits.max_upload_bytes)}.`;
      }
      return null;
    },
    [limits],
  );

  const handle = useCallback(
    (file: File | undefined) => {
      if (!file) return;
      const problem = validate(file);
      setLocalError(problem);
      if (problem) return;
      const parsed = hfov.trim() ? Number(hfov) : undefined;
      if (parsed !== undefined && (!Number.isFinite(parsed) || parsed <= 5 || parsed >= 175)) {
        setLocalError("Field of view must be between 5 and 175 degrees.");
        return;
      }
      onSubmit(file, parsed);
    },
    [hfov, onSubmit, validate],
  );

  const shown = localError ?? error;

  return (
    <div className="mx-auto w-full max-w-2xl">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          handle(e.dataTransfer.files?.[0]);
        }}
        className={[
          "rounded-[--radius-panel] border border-dashed p-10 text-center transition-colors",
          dragging ? "border-signal bg-ink-850" : "border-ink-700 bg-ink-850/40",
          disabled ? "pointer-events-none opacity-50" : "",
        ].join(" ")}
      >
        <UploadSimple size={26} weight="light" className="mx-auto text-ink-400" />
        <p className="mt-4 text-sm text-ink-200">Drop a video here</p>
        <p className="mt-1 text-xs text-ink-400">
          {limits
            ? `Up to ${Math.round(limits.max_duration_s)} seconds, ${bytes(limits.max_upload_bytes)}`
            : "Checking limits"}
        </p>

        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          disabled={disabled}
          className="mt-6 rounded-[--radius-panel] bg-ink-100 px-5 py-2 text-sm font-medium
                     text-ink-900 transition-transform hover:bg-white active:translate-y-px
                     disabled:cursor-not-allowed"
        >
          Choose file
        </button>

        <input
          ref={inputRef}
          type="file"
          accept={limits?.accepted_types.join(",") ?? "video/*"}
          className="sr-only"
          onChange={(e) => handle(e.target.files?.[0])}
        />
      </div>

      <div className="mt-6 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <label htmlFor="hfov" className="text-xs text-ink-400">
          Horizontal field of view, if you know it
        </label>
        <div className="flex items-center gap-2">
          <input
            id="hfov"
            value={hfov}
            onChange={(e) => setHfov(e.target.value)}
            inputMode="decimal"
            placeholder="60"
            aria-describedby="hfov-help"
            className="tnum w-24 rounded-[--radius-panel] border border-ink-700 bg-ink-850
                       px-3 py-1.5 text-sm text-ink-100 placeholder:text-ink-500
                       focus:border-ink-500 focus:outline-none"
          />
          <span className="text-xs text-ink-400">degrees</span>
        </div>
      </div>
      <p id="hfov-help" className="mt-2 text-xs leading-relaxed text-ink-500">
        A video carries no calibration. Left blank, the reconstruction assumes 60 degrees, which
        suits most phone and webcam footage. The result reports which assumption it used.
      </p>

      {shown && (
        <p
          role="alert"
          className="mt-5 flex items-start gap-2 rounded-[--radius-panel] border border-ink-700
                     bg-ink-850 px-3 py-2.5 text-sm text-ink-200"
        >
          <Warning size={16} weight="light" className="mt-0.5 shrink-0 text-signal" />
          <span>{shown}</span>
        </p>
      )}
    </div>
  );
}
