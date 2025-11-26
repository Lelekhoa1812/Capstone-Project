// components/Timeline.tsx
"use client";

import { useEffect, useRef } from "react";
import type { LabelSegment, DrivingLabel } from "@/lib/types";
import type { DragState, DragKind } from "@/hooks/useLabeling";

type Props = {
  timelineDuration: number;
  currentTime: number;
  setCurrentTime: (n: number) => void;

  selectionStart: number | null;
  selectionEnd: number | null;
  setSelectionStart: (n: number | null) => void;
  setSelectionEnd: (n: number | null) => void;

  labelSegments: LabelSegment[];
  setLabelSegments: (updater: (prev: LabelSegment[]) => LabelSegment[]) => void;

  selectedSegmentId: string | null;
  setSelectedSegmentId: (id: string | null) => void;

  drag: DragState | null;
  setDrag: (d: DragState | null) => void;

  MIN_BLOCK_SEC: number;
  clamp: (v: number, lo: number, hi: number) => number;
  getImmediateNeighbors: (
    all: LabelSegment[],
    selfId: string,
    start: number,
    end: number
  ) => { leftEnd: number; rightStart: number; };
  getLabelColor: (l: DrivingLabel) => string;
};

export default function Timeline({
  timelineDuration,
  currentTime, setCurrentTime,
  selectionStart, selectionEnd, setSelectionStart, setSelectionEnd,
  labelSegments, setLabelSegments,
  selectedSegmentId, setSelectedSegmentId,
  drag, setDrag,
  MIN_BLOCK_SEC, clamp, getImmediateNeighbors, getLabelColor
}: Props) {
  const timelineRef = useRef<HTMLDivElement>(null);

  const pxToSec = (px: number) => {
    if (!timelineRef.current || timelineDuration <= 0) return 0;
    const w = timelineRef.current.getBoundingClientRect().width;
    return (px / w) * timelineDuration;
  };

  // drag events
  useEffect(() => {
    if (!drag) return;

    const onMove = (e: MouseEvent) => {
      const dx = e.clientX - drag.startClientX;
      const dSec = pxToSec(dx);

      setLabelSegments(prev =>
        prev.map(seg => {
          if (seg.id !== drag.id) return seg;

          if (drag.kind === "move") {
            const width = drag.origEnd - drag.origStart;
            let newStart = clamp(drag.origStart + dSec, drag.leftBound, drag.rightBound);
            let newEnd = newStart + width;
            return { ...seg, startTime: newStart, endTime: newEnd };
          }

          if (drag.kind === "resize-start") {
            let newStart = clamp(drag.origStart + dSec, drag.leftBound, drag.rightBound);
            return { ...seg, startTime: newStart };
          }

          if (drag.kind === "resize-end") {
            let newEnd = clamp(drag.origEnd + dSec, drag.leftBound, drag.rightBound);
            return { ...seg, endTime: newEnd };
          }

          return seg;
        })
      );
    };

    const onUp = () => setDrag(null);

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    window.addEventListener("mouseleave", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      window.removeEventListener("mouseleave", onUp);
    };
  }, [drag, setDrag, setLabelSegments]); // eslint-disable-line

  const handleTimelineClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!timelineRef.current || timelineDuration <= 0) return;
    const rect = timelineRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const t = clamp((x / rect.width) * timelineDuration, 0, timelineDuration);
    if (e.shiftKey && selectionStart !== null) setSelectionEnd(t);
    else {
      setSelectionStart(t);
      setSelectionEnd(null);
    }
    setCurrentTime(t);
    setSelectedSegmentId(null);
  };

  const beginDrag = (kind: DragKind, e: React.MouseEvent, seg: LabelSegment) => {
    e.preventDefault();
    e.stopPropagation();
    setSelectedSegmentId(seg.id);

    if (kind === "resize-start") {
      const { leftEnd } = getImmediateNeighbors(labelSegments, seg.id, seg.startTime, seg.endTime);
      const leftBound = leftEnd;
      const rightBound = seg.endTime - MIN_BLOCK_SEC;
      return setDrag({
        id: seg.id, kind, startClientX: e.clientX,
        origStart: seg.startTime, origEnd: seg.endTime,
        leftBound, rightBound,
      });
    }

    if (kind === "resize-end") {
      const { rightStart } = getImmediateNeighbors(labelSegments, seg.id, seg.startTime, seg.endTime);
      const leftBound = seg.startTime + MIN_BLOCK_SEC;
      const rightBound = rightStart;
      return setDrag({
        id: seg.id, kind, startClientX: e.clientX,
        origStart: seg.startTime, origEnd: seg.endTime,
        leftBound, rightBound,
      });
    }

    // move
    const { leftEnd, rightStart } = getImmediateNeighbors(labelSegments, seg.id, seg.startTime, seg.endTime);
    const width = seg.endTime - seg.startTime;
    const leftBound = leftEnd;
    const rightBound = rightStart - width;
    return setDrag({
      id: seg.id, kind, startClientX: e.clientX,
      origStart: seg.startTime, origEnd: seg.endTime,
      leftBound, rightBound,
    });
  };

  return (
    <div
      ref={timelineRef}
      className="relative h-24 bg-muted rounded-lg cursor-pointer border select-none"
      onClick={handleTimelineClick}
    >
      {/* blocks */}
      {labelSegments.map((seg) => {
        const leftPct  = (seg.startTime / Math.max(1, timelineDuration)) * 100;
        const widthPct = ((seg.endTime - seg.startTime) / Math.max(1, timelineDuration)) * 100;

        return (
          <div
            key={seg.id}
            className={`absolute top-0 h-full ${getLabelColor(seg.label)} opacity-60 border-l-2 border-r-2 border-primary rounded
                        ${selectedSegmentId === seg.id ? "ring-2 ring-primary" : ""}`}
            style={{ left: `${leftPct}%`, width: `${widthPct}%` }}
            onMouseDown={(e) => {
              e.stopPropagation();
              setSelectedSegmentId(seg.id);
            }}
            title={`${seg.label}: ${seg.startTime.toFixed(2)}s–${seg.endTime.toFixed(2)}s`}
          >
            {/* label */}
            <div className="text-xs p-1 text-white font-medium pointer-events-none select-none">
              {seg.label}
            </div>

            {/* left handle */}
            <div
              className="absolute left-0 top-0 h-full w-2 cursor-ew-resize bg-primary/70 hover:bg-primary"
              onMouseDown={(e) => beginDrag("resize-start", e, seg)}
              title="Drag to resize start"
            />

            {/* right handle */}
            <div
              className="absolute right-0 top-0 h-full w-2 cursor-ew-resize bg-primary/70 hover:bg-primary"
              onMouseDown={(e) => beginDrag("resize-end", e, seg)}
              title="Drag to resize end"
            />

            {/* move area */}
            <div
              className="absolute inset-y-0 left-2 right-2 cursor-grab active:cursor-grabbing"
              onMouseDown={(e) => beginDrag("move", e, seg)}
              title="Drag to move"
            />
          </div>
        );
      })}

      {/* selection preview */}
      {selectionStart !== null && selectionEnd !== null && (
        <div
          className="absolute top-0 h-full bg-primary/30 border-2 border-primary"
          style={{
            left: `${(Math.min(selectionStart, selectionEnd) / Math.max(1, timelineDuration)) * 100}%`,
            width: `${(Math.abs(selectionEnd - selectionStart) / Math.max(1, timelineDuration)) * 100}%`,
          }}
        />
      )}

      {/* playhead */}
      <div
        className="absolute top-0 h-full w-0.5 bg-destructive z-10"
        style={{ left: `${(currentTime / Math.max(1, timelineDuration)) * 100}%` }}
      />
    </div>
  );
}
