// hooks/useLabeling.ts
"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { DrivingLabel, LabelSegment } from "@/lib/types";
// Firebase functions are not used in this client-side hook
// import { callGetTripMeta, callLabelTrip, callListEligible, ensureAnon } from "@/lib/firebase";

export type DragKind = "resize-start" | "resize-end" | "move";

export type DragState = {
  id: string;
  kind: DragKind;
  startClientX: number;
  origStart: number;
  origEnd: number;
  leftBound: number;
  rightBound: number;
};

export function useLabeling() {
  // trips & timeline
  const [tripOptions, setTripOptions] = useState<string[]>([]);
  const [selectedTrip, setSelectedTrip] = useState<string>("");

  const [timelineStartISO, setTimelineStartISO] = useState<string>();
  const [timelineDuration, setTimelineDuration] = useState<number>(0);

  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);

  // labeling
  const [selectionStart, setSelectionStart] = useState<number | null>(null);
  const [selectionEnd, setSelectionEnd] = useState<number | null>(null);
  const [selectedLabel, setSelectedLabel] = useState<DrivingLabel>("passive");
  const [labelSegments, setLabelSegments] = useState<LabelSegment[]>([]);
  const [selectedSegmentId, setSelectedSegmentId] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  // drag
  const [drag, setDrag] = useState<DragState | null>(null);
  const MIN_BLOCK_SEC = 1;

  // timers
  const tickerRef = useRef<ReturnType<typeof setInterval>>();

  // load trips
  useEffect(() => {
    (async () => {
      try {
        await ensureAnon();
        const res: any = await callListEligible({});
        const files: string[] = res?.data?.trips || [];
        const sessions = files.filter((n) => n.endsWith("_raw.csv")).map((n) => n.replace("_raw.csv", ""));
        setTripOptions(sessions);
        if (!selectedTrip && sessions.length) setSelectedTrip(sessions[0]);
      } catch (e) {
        console.error("[listEligibleTrips] failed", e);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // fetch meta on trip change
  useEffect(() => {
    if (!selectedTrip) return;
    (async () => {
      try {
        await ensureAnon();
        const rawPath = `skyledge/raw/${selectedTrip}_raw.csv`;
        const res: any = await callGetTripMeta({ rawPath, timestampColumn: "timestamp" });
        const { startTsISO, endTsISO } = res?.data || {};
        if (!startTsISO || !endTsISO) throw new Error("missing meta");
        const startMs = new Date(startTsISO).getTime();
        const endMs = new Date(endTsISO).getTime();
        setTimelineStartISO(startTsISO);
        setTimelineDuration(Math.max(0, Math.floor((endMs - startMs) / 1000)));

        // reset per-trip
        setCurrentTime(0);
        setIsPlaying(false);
        setSelectionStart(null);
        setSelectionEnd(null);
        setLabelSegments([]);
        setSelectedSegmentId(null);
      } catch (e) {
        console.error("[getTripMeta] failed", e);
        setTimelineStartISO(undefined);
        setTimelineDuration(0);
      }
    })();
  }, [selectedTrip]);

  // playback tick
  useEffect(() => {
    if (isPlaying && timelineDuration > 0) {
      tickerRef.current = setInterval(() => {
        setCurrentTime((p) => {
          const n = p + playbackSpeed * 0.1; // 100ms tick
          if (n >= timelineDuration) {
            clearInterval(tickerRef.current!);
            tickerRef.current = undefined;
            setIsPlaying(false);
            return timelineDuration;
          }
          return n;
        });
      }, 100);
    }
    return () => {
      if (tickerRef.current) clearInterval(tickerRef.current);
      tickerRef.current = undefined;
    };
  }, [isPlaying, playbackSpeed, timelineDuration]);

  // utils
  const clamp = useCallback(
    (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v)),
    []
  );

  const formatTime = useCallback((secondsFromStart: number) => {
    if (!timelineStartISO) return "—:—:—";
    const base = new Date(timelineStartISO).getTime();
    const t = new Date(base + Math.max(0, secondsFromStart) * 1000);
    return t.toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }, [timelineStartISO]);

  const overlapsAny = useCallback(
    (start: number, end: number, excludeId?: string) =>
      labelSegments.some(
        (s) => s.id !== excludeId && Math.max(s.startTime, start) < Math.min(s.endTime, end)
      ),
    [labelSegments]
  );

  const getImmediateNeighbors = useCallback((
    all: LabelSegment[],
    selfId: string,
    start: number,
    end: number
  ) => {
    const others = all.filter(s => s.id !== selfId).sort((a, b) => a.startTime - b.startTime);
    const left = others.filter(o => o.endTime <= start).slice(-1)[0] || null;
    const right = others.find(o => o.startTime >= end) || null;
    return {
      leftEnd: left ? left.endTime : 0,
      rightStart: right ? right.startTime : timelineDuration,
    };
  }, [timelineDuration]);

  // save labels → Cloud Function
  const saveLabels = useCallback(async () => {
    if (!selectedTrip) throw new Error("Pick a trip first.");
    if (!timelineStartISO) throw new Error("Timeline isn’t ready.");
    if (labelSegments.length === 0) throw new Error("No labels to save.");

    setSaving(true);
    try {
      await ensureAnon();

      const rawPath = `skyledge/raw/${selectedTrip}_raw.csv`;
      const startMs = new Date(timelineStartISO).getTime();
      const ranges = labelSegments.map((seg) => {
        const s = startMs + Math.floor(seg.startTime * 1000);
        const e = startMs + Math.floor(seg.endTime * 1000);
        return {
          startTs: new Date(Math.min(s, e)).toISOString(),
          endTs: new Date(Math.max(s, e)).toISOString(),
          label: seg.label,
        };
      });

      const res: any = await callLabelTrip({ rawPath, timestampColumn: "timestamp", ranges });
      return res?.data;
    } finally {
      setSaving(false);
    }
  }, [labelSegments, selectedTrip, timelineStartISO]);

  // computed label color (for UI)
  const getLabelColor = useCallback((label: DrivingLabel) => (
    label === "idle" ? "bg-chart-3" : label === "passive" ? "bg-chart-1" : "bg-chart-4"
  ), []);

  return {
    // state
    tripOptions,
    selectedTrip, setSelectedTrip,

    timelineStartISO,
    timelineDuration,

    currentTime, setCurrentTime,
    isPlaying, setIsPlaying,
    playbackSpeed, setPlaybackSpeed,

    selectionStart, setSelectionStart,
    selectionEnd, setSelectionEnd,

    selectedLabel, setSelectedLabel,
    labelSegments, setLabelSegments,
    selectedSegmentId, setSelectedSegmentId,

    saving,

    // drag
    drag, setDrag,
    MIN_BLOCK_SEC,

    // helpers
    clamp,
    formatTime,
    overlapsAny,
    getImmediateNeighbors,
    saveLabels,
    getLabelColor,
  };
}
