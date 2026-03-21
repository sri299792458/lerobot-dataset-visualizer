"use client";

import React, { useEffect, useRef } from "react";
import { useTime } from "@/context/time-context";
import {
  type DepthStreamDescriptor,
} from "@/app/[org]/[dataset]/[episode]/fetch-data";

type DepthStreamViewerProps = {
  depthStreams: DepthStreamDescriptor[];
};

const VIDEO_SYNC_TOLERANCE = 0.2;

export function DepthStreamViewer({
  depthStreams,
}: DepthStreamViewerProps) {
  const { currentTime, isPlaying } = useTime();
  const videoRefs = useRef<Record<string, HTMLVideoElement | null>>({});

  useEffect(() => {
    depthStreams.forEach((stream) => {
      if (!stream.previewUrl) return;
      const video = videoRefs.current[stream.field];
      if (!video) return;

      if (Math.abs(video.currentTime - currentTime) > VIDEO_SYNC_TOLERANCE) {
        video.currentTime = currentTime;
      }
    });
  }, [currentTime, depthStreams]);

  useEffect(() => {
    depthStreams.forEach((stream) => {
      if (!stream.previewUrl) return;
      const video = videoRefs.current[stream.field];
      if (!video) return;

      if (isPlaying) {
        void video.play().catch(() => {});
      } else {
        video.pause();
      }
    });
  }, [isPlaying, depthStreams]);

  if (depthStreams.length === 0) return null;

  return (
    <div className="mb-6 rounded-lg border border-slate-700 bg-slate-900/60 p-4">
      <div className="mb-3 flex items-baseline justify-between">
        <div>
          <h3 className="text-sm font-semibold uppercase tracking-[0.18em] text-slate-300">
            Depth
          </h3>
          <p className="mt-1 text-sm text-slate-400">
            RealSense-style preview videos generated from the lossless depth sidecar.
          </p>
        </div>
      </div>

      <div className="flex flex-wrap gap-x-3 gap-y-6">
        {depthStreams.map((stream) => (
          <div key={stream.field} className="max-w-96">
            <div className="flex items-center justify-between rounded-t-xl bg-slate-800 px-3 py-2 text-sm text-slate-200">
              <span>{stream.field}</span>
              <span className="text-xs text-slate-400">{stream.unit}</span>
            </div>
            {stream.previewUrl ? (
              <>
                <video
                  ref={(node) => {
                    videoRefs.current[stream.field] = node;
                  }}
                  src={stream.previewUrl}
                  className="w-full rounded-b-xl border border-slate-800 bg-black object-contain"
                  muted
                  playsInline
                  preload="auto"
                  onLoadedData={(event) => {
                    event.currentTarget.currentTime = currentTime;
                  }}
                />
                <div className="mt-2 flex items-center justify-between text-xs text-slate-400">
                  <span>
                    {stream.gridFps
                      ? `frame ${Math.max(0, Math.floor(currentTime * stream.gridFps + 1e-6))}`
                      : `${currentTime.toFixed(2)}s`}
                  </span>
                  <span>{stream.sourceTopic}</span>
                </div>
              </>
            ) : (
              <div className="rounded-b-xl border border-slate-800 bg-slate-950 px-3 py-6 text-sm text-slate-400">
                No depth preview video available.
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export default DepthStreamViewer;
