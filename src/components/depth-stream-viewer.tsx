"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { decode } from "fast-png";
import { useTime } from "@/context/time-context";
import {
  type DepthStreamDescriptor,
} from "@/app/[org]/[dataset]/[episode]/fetch-data";
import {
  fetchParquetFile,
  readParquetAsObjects,
} from "@/utils/parquetUtils";

type DepthFrame = {
  rowIndex: number;
  frameIndex: number;
  timestamp: number;
};

type LoadedDepthStream = DepthStreamDescriptor & {
  frames: DepthFrame[];
};

type DepthStreamViewerProps = {
  depthStreams: DepthStreamDescriptor[];
};

function binaryToUint8Array(value: unknown): Uint8Array {
  if (value instanceof Uint8Array) return value;
  if (value instanceof ArrayBuffer) return new Uint8Array(value);
  if (ArrayBuffer.isView(value)) {
    return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  }
  if (Array.isArray(value)) return Uint8Array.from(value);
  if (typeof value === "string") {
    const bytes = new Uint8Array(value.length);
    for (let i = 0; i < value.length; i++) {
      bytes[i] = value.charCodeAt(i) & 0xff;
    }
    return bytes;
  }
  throw new Error("Unsupported parquet binary representation for depth payload");
}

function toNumber(value: unknown): number {
  if (typeof value === "number") return value;
  if (typeof value === "bigint") return Number(value);
  if (typeof value === "string") return Number(value);
  return 0;
}

function nearestFrame(frames: DepthFrame[], time: number): DepthFrame | null {
  if (frames.length === 0) return null;
  let best = frames[0];
  let bestDist = Math.abs(best.timestamp - time);
  for (let i = 1; i < frames.length; i++) {
    const dist = Math.abs(frames[i].timestamp - time);
    if (dist < bestDist) {
      best = frames[i];
      bestDist = dist;
    }
  }
  return best;
}

function depthBytesToPreviewUrl(bytes: Uint8Array): string {
  const decoded = decode(bytes);
  const { width, height, data, channels } = decoded;

  if (channels !== 1) {
    throw new Error(`Unsupported depth PNG channels: ${channels}`);
  }

  const values =
    data instanceof Uint16Array || data instanceof Uint8Array
      ? data
      : Uint16Array.from(data as Iterable<number>);

  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;

  for (let i = 0; i < values.length; i++) {
    const value = values[i];
    if (value <= 0) continue;
    if (value < min) min = value;
    if (value > max) max = value;
  }

  if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) {
    min = 0;
    max = 1;
  }

  const ramp = [
    [0.07, 0.12, 0.35],
    [0.16, 0.31, 0.73],
    [0.0, 0.67, 0.73],
    [0.24, 0.79, 0.3],
    [0.98, 0.85, 0.26],
    [0.91, 0.39, 0.18],
    [0.55, 0.04, 0.1],
  ] as const;

  const sampleRamp = (t: number): [number, number, number] => {
    const clamped = Math.max(0, Math.min(1, t));
    const scaled = clamped * (ramp.length - 1);
    const index = Math.min(ramp.length - 2, Math.floor(scaled));
    const local = scaled - index;
    const a = ramp[index];
    const b = ramp[index + 1];
    return [
      a[0] + (b[0] - a[0]) * local,
      a[1] + (b[1] - a[1]) * local,
      a[2] + (b[2] - a[2]) * local,
    ];
  };

  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < values.length; i++) {
    const value = values[i];
    const alpha = value > 0 ? 255 : 0;
    const normalized =
      value > 0 ? Math.max(0, Math.min(1, (value - min) / (max - min))) : 0;
    const [r, g, b] = sampleRamp(normalized);
    const offset = i * 4;
    rgba[offset] = Math.round(r * 255);
    rgba[offset + 1] = Math.round(g * 255);
    rgba[offset + 2] = Math.round(b * 255);
    rgba[offset + 3] = alpha;
  }

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("Failed to create canvas context for depth preview");
  }

  context.putImageData(new ImageData(rgba, width, height), 0, 0);
  return canvas.toDataURL("image/png");
}

export function DepthStreamViewer({
  depthStreams,
}: DepthStreamViewerProps) {
  const { currentTime } = useTime();
  const [loadedStreams, setLoadedStreams] = useState<LoadedDepthStream[]>([]);
  const [previewUrls, setPreviewUrls] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const pendingPreviewKeysRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    if (depthStreams.length === 0) {
      setLoadedStreams([]);
      setPreviewUrls({});
      setError(null);
      setLoading(false);
      pendingPreviewKeysRef.current.clear();
      return;
    }

    let cancelled = false;

    async function load() {
      setLoading(true);
      setError(null);
      try {
        const streams = await Promise.all(
          depthStreams.map(async (stream): Promise<LoadedDepthStream> => {
            const parquetFile = await fetchParquetFile(stream.url);
            const rows = await readParquetAsObjects(parquetFile, [
              "frame_index",
              "timestamp",
            ]);
            const firstTimestamp = rows.length
              ? toNumber(rows[0].timestamp)
              : 0;

            const frames = rows.map((row, rowIndex) => {
              return {
                rowIndex,
                frameIndex: toNumber(row.frame_index),
                timestamp: toNumber(row.timestamp) - firstTimestamp,
              };
            });

            return {
              ...stream,
              frames,
            };
          }),
        );

        if (cancelled) return;
        setLoadedStreams(streams);
      } catch (err) {
        if (cancelled) return;
        const message = err instanceof Error ? err.message : String(err);
        setError(message || "Failed to load depth sidecar");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    void load();

    return () => {
      cancelled = true;
    };
  }, [depthStreams]);

  const activeFrames = useMemo(
    () =>
      loadedStreams.map((stream) => {
        const activeFrame = nearestFrame(stream.frames, currentTime);
        return {
          ...stream,
          activeFrame,
        };
      }),
    [loadedStreams, currentTime],
  );

  useEffect(() => {
    const missingStreams = activeFrames.filter((stream) => {
      if (!stream.activeFrame) return false;
      const key = `${stream.field}:${stream.activeFrame.frameIndex}`;
      return !previewUrls[key] && !pendingPreviewKeysRef.current.has(key);
    });

    if (missingStreams.length === 0) return;

    let cancelled = false;
    for (const stream of missingStreams) {
      const key = `${stream.field}:${stream.activeFrame!.frameIndex}`;
      pendingPreviewKeysRef.current.add(key);
    }

    async function loadMissingPreviews() {
      try {
        const entries = await Promise.all(
          missingStreams.map(async (stream) => {
            const activeFrame = stream.activeFrame!;
            const parquetFile = await fetchParquetFile(stream.url);
            const rows = await readParquetAsObjects(
              parquetFile,
              ["png16_bytes"],
              {
                rowStart: activeFrame.rowIndex,
                rowEnd: activeFrame.rowIndex + 1,
                utf8: false,
              },
            );
            if (rows.length === 0) {
              throw new Error(`Depth frame missing for ${stream.field}`);
            }
            const bytes = binaryToUint8Array(rows[0].png16_bytes);
            return {
              key: `${stream.field}:${activeFrame.frameIndex}`,
              previewUrl: depthBytesToPreviewUrl(bytes),
            };
          }),
        );

        if (cancelled) return;
        setPreviewUrls((prev) => {
          const next = { ...prev };
          for (const entry of entries) {
            next[entry.key] = entry.previewUrl;
          }
          return next;
        });
      } catch (err) {
        if (cancelled) return;
        const message = err instanceof Error ? err.message : String(err);
        setError(message || "Failed to decode depth preview");
      } finally {
        for (const stream of missingStreams) {
          const key = `${stream.field}:${stream.activeFrame!.frameIndex}`;
          pendingPreviewKeysRef.current.delete(key);
        }
      }
    }

    void loadMissingPreviews();

    return () => {
      cancelled = true;
    };
  }, [activeFrames, previewUrls]);

  if (depthStreams.length === 0) return null;

  return (
    <div className="mb-6 rounded-lg border border-slate-700 bg-slate-900/60 p-4">
      <div className="mb-3 flex items-baseline justify-between">
        <div>
          <h3 className="text-sm font-semibold uppercase tracking-[0.18em] text-slate-300">
            Depth
          </h3>
          <p className="mt-1 text-sm text-slate-400">
            Lossless sidecar aligned to the published frame grid.
          </p>
        </div>
      </div>

      {loading && (
        <p className="text-sm text-slate-400">Loading depth frames…</p>
      )}

      {error && (
        <p className="text-sm text-red-300">
          Failed to load depth sidecar: {error}
        </p>
      )}

      {!loading && !error && (
        <div className="flex flex-wrap gap-x-3 gap-y-6">
          {activeFrames.map((stream) => (
            <div key={stream.field} className="max-w-96">
              <div className="flex items-center justify-between rounded-t-xl bg-slate-800 px-3 py-2 text-sm text-slate-200">
                <span>{stream.field}</span>
                <span className="text-xs text-slate-400">{stream.unit}</span>
              </div>
              {stream.activeFrame ? (
                <>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  {previewUrls[
                    `${stream.field}:${stream.activeFrame.frameIndex}`
                  ] ? (
                    <img
                      src={
                        previewUrls[
                          `${stream.field}:${stream.activeFrame.frameIndex}`
                        ]
                      }
                      alt={stream.field}
                      className="w-full rounded-b-xl border border-slate-800 bg-black object-contain"
                    />
                  ) : (
                    <div className="rounded-b-xl border border-slate-800 bg-slate-950 px-3 py-6 text-sm text-slate-400">
                      Loading preview…
                    </div>
                  )}
                  <div className="mt-2 flex items-center justify-between text-xs text-slate-400">
                    <span>frame {stream.activeFrame.frameIndex}</span>
                    <span>{stream.sourceTopic}</span>
                  </div>
                </>
              ) : (
                <div className="rounded-b-xl border border-slate-800 bg-slate-950 px-3 py-6 text-sm text-slate-400">
                  No depth frame available.
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default DepthStreamViewer;
