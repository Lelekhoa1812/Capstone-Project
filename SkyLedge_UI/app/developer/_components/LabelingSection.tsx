"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { 
  Play, 
  Pause, 
  RotateCcw, 
  Tag as TagIcon, 
  Trash2, 
  RefreshCw,
  Search,
  FileText,
  Clock,
  Calendar,
  Activity,
  CheckCircle,
  AlertCircle,
  Database
} from "lucide-react";
import Papa from "papaparse";
import type { DrivingLabel, LabelSegment } from "@/lib/types";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  Brush,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";

type FileType = "raw" | "processed" | "labeled";
type Row = Record<string, any> & { __sec: number; __tsISO: string };

interface LabeledFile {
  name: string;
  size: number;
  timeCreated: string;
  duration?: number;
  sessionId: string;
  date: string;
  totalSegments?: number;
  labeledSegments?: number;
  completionRate?: number;
}

const GUTTER_PX = 24;

const getLabelColor = (label: DrivingLabel) => (
  label === "idle" ? "bg-chart-3" : label === "passive" ? "bg-chart-1" : "bg-chart-4"
);

const fmtHMS = (sec: number) => {
  const s = Math.max(0, Math.floor(sec));
  const hh = String(Math.floor(s / 3600)).padStart(2, "0");
  const mm = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
  const ss = String(s % 60).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
};

const formatFileSize = (bytes: number) => {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
};

const formatDuration = (seconds?: number) => {
  if (!seconds) return 'Unknown';
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;
  
  if (hours > 0) {
    return `${hours}h ${minutes}m ${secs}s`;
  } else if (minutes > 0) {
    return `${minutes}m ${secs}s`;
  } else {
    return `${secs}s`;
  }
};

export default function LabelingSection() {
  // file selection
  const [fileType, setFileType] = useState<FileType>("processed");
  const [rawFiles, setRawFiles] = useState<string[]>([]);
  const [processedFiles, setProcessedFiles] = useState<string[]>([]);
  const [labeledFiles, setLabeledFiles] = useState<LabeledFile[]>([]);
  const [selectedFile, setSelectedFile] = useState<string>("");
  const [searchTerm, setSearchTerm] = useState("");

  // data
  const [rows, setRows] = useState<Row[]>([]);
  const [numericCols, setNumericCols] = useState<string[]>([]);
  const [selectedCols, setSelectedCols] = useState<string[]>([]);
  const [rangeStart, setRangeStart] = useState(0);
  const [rangeEnd, setRangeEnd] = useState(0);

  // playback
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const tickerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // labeling
  const [labelSegments, setLabelSegments] = useState<LabelSegment[]>([]);
  const [selectionStart, setSelectionStart] = useState<number | null>(null);
  const [selectionEnd, setSelectionEnd] = useState<number | null>(null);
  const [selectedLabel, setSelectedLabel] = useState<DrivingLabel>("passive");
  const [selectedSegmentId, setSelectedSegmentId] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [loading, setLoading] = useState(false);

  const timelineRef = useRef<HTMLDivElement>(null);
  const MIN_BLOCK_SEC = 1;

  const hasData = rows.length > 0 && rangeEnd > rangeStart;
  const rangeLen = Math.max(1e-6, rangeEnd - rangeStart);
  const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
  const clampToData = (sec: number) => clamp(sec, rangeStart, rangeEnd);
  const xToFrac = (clientX: number, el: HTMLDivElement | null) => {
    if (!el) return 0;
    const rect = el.getBoundingClientRect();
    const innerW = Math.max(1, rect.width - 2 * GUTTER_PX);
    const innerX = clientX - rect.left - GUTTER_PX;
    return clamp(innerX / innerW, 0, 1);
  };
  const fracToSec = (f: number) => rangeStart + clamp(f, 0, 1) * rangeLen;
  const pxDeltaToSec = (dx: number) => {
    const innerW = Math.max(1, (timelineRef.current?.getBoundingClientRect().width ?? 1) - 2 * GUTTER_PX);
    return (dx / innerW) * rangeLen;
  };

  const overlapsAny = (start: number, end: number, excludeId?: string) =>
    labelSegments.some((s) => s.id !== excludeId && Math.max(s.startTime, start) < Math.min(s.endTime, end));

  // fetch file lists
  useEffect(() => {
    (async () => {
      setLoading(true);
      try {
        const [rawResp, procResp, labeledResp] = await Promise.all([
          fetch("/api/raw-files").then((r) => r.json()),
          fetch("/api/processed-files").then((r) => r.json()),
          fetch("/api/labeled-files-detailed").then((r) => r.json()),
        ]);
        
        if (rawResp?.success) setRawFiles(rawResp.files || []);
        if (procResp?.success) setProcessedFiles((procResp.files || []).map((f: any) => f.name));
        if (labeledResp?.success) setLabeledFiles(labeledResp.files || []);
      } catch (e) {
        console.error("[fetch lists]", e);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  // load CSV when selected
  useEffect(() => {
    if (!selectedFile) return;
    (async () => {
      setLoading(true);
      try {
        const res = await fetch(`/api/download-file?fileName=${encodeURIComponent(selectedFile)}`);
        if (!res.ok) throw new Error("Failed to download file");
        const text = await res.text();
        const parsed = Papa.parse(text, { header: true, dynamicTyping: true, skipEmptyLines: true });
        const data = (parsed.data as any[]) || [];
        const fields = (parsed.meta.fields || []) as string[];
        const tsCol = fields.find((h) => h.toLowerCase() === "timestamp") || "timestamp";
        const labelCol = fields.find((h) => h.toLowerCase() === "driving_style") || "driving_style";
        
        const startMs = (() => {
          for (const row of data) {
            const v = row[tsCol];
            const ms = typeof v === "number" ? (v > 1e11 ? v : v * 1000) : Date.parse(String(v));
            if (!Number.isNaN(ms)) return ms;
          }
          return NaN;
        })();
        
        const normalized: Row[] = [];
        const nums = new Set<string>();
        for (const row of data) {
          const v = row[tsCol];
          const ms = typeof v === "number" ? (v > 1e11 ? v : v * 1000) : Date.parse(String(v));
          if (Number.isNaN(ms)) continue;
          const sec = (ms - startMs) / 1000;
          const nr: Row = { ...row, __sec: sec, __tsISO: new Date(ms).toISOString() };
          normalized.push(nr);
          for (const [k, val] of Object.entries(row)) {
            if (k === tsCol || k === labelCol) continue;
            if (typeof val === "number" && Number.isFinite(val)) nums.add(k);
          }
        }
        normalized.sort((a, b) => a.__sec - b.__sec);
        setRows(normalized);
        const s0 = normalized.length ? normalized[0].__sec : 0;
        const s1 = normalized.length ? normalized[normalized.length - 1].__sec : 0;
        setRangeStart(s0);
        setRangeEnd(s1);
        const numList = Array.from(nums);
        setNumericCols(numList);
        const clean = (s: string) => s.replace(/[^a-z0-9]/gi, "").toLowerCase();
        const findAlias = (aliases: string[]) => numList.find((c) => aliases.includes(clean(c)));
        const defaults: string[] = [];
        const speed = findAlias(["speed", "vehiclespeed", "speedkmh", "speedmph"]);
        if (speed) defaults.push(speed);
        const rpm = findAlias(["rpm", "enginerpm", "enginerevolutions"]);
        if (rpm) defaults.push(rpm);
        const thr = findAlias(["throttle", "throttlepos", "throttleposition"]);
        if (thr) defaults.push(thr);
        setSelectedCols((defaults.length ? defaults : numList).slice(0, 3));

        // Load existing labels from the file
        const existingSegments: LabelSegment[] = [];
        if (normalized.length && labelCol in normalized[0]) {
          // Check if there are any non-empty labels
          const hasLabels = normalized.some(row => {
            const label = row[labelCol];
            return label && label !== '' && label !== 'null' && label !== 'undefined' && label !== null;
          });

          if (hasLabels) {
            let curLabel: DrivingLabel | null = null;
            let curStart = 0;
            let segmentId = 0;

            for (let i = 0; i < normalized.length; i++) {
              const row = normalized[i];
              const lab: DrivingLabel | null = (row[labelCol] && 
                row[labelCol] !== '' && 
                row[labelCol] !== 'null' && 
                row[labelCol] !== 'undefined' && 
                row[labelCol] !== null) ? (row[labelCol] as DrivingLabel) : null;

              if (lab !== curLabel) {
                // If we had a previous label, save it
                if (curLabel !== null && i > 0) {
                  const prevSec = normalized[i - 1].__sec;
                  existingSegments.push({ 
                    id: `segment-${segmentId++}`, 
                    startTime: curStart, 
                    endTime: prevSec, 
                    label: curLabel, 
                    confidence: 1 
                  });
                }
                
                // Start new segment
                curLabel = lab;
                curStart = row.__sec;
              }
            }

            // Add the last segment if it has a label
            if (curLabel !== null) {
              existingSegments.push({
                id: `segment-${segmentId++}`,
                startTime: curStart,
                endTime: normalized[normalized.length - 1].__sec,
                label: curLabel,
                confidence: 1,
              });
            }
          }
        }
        setLabelSegments(existingSegments);

        // reset UI
        setCurrentTime(s0);
        setIsPlaying(false);
        setSelectedSegmentId(null);
        setSelectionStart(null);
        setSelectionEnd(null);
      } catch (e) {
        console.error("[load csv]", e);
        setRows([]);
        setNumericCols([]);
        setSelectedCols([]);
        setRangeStart(0);
        setRangeEnd(0);
        setLabelSegments([]);
      } finally {
        setLoading(false);
      }
    })();
  }, [selectedFile]);

  // playback
  useEffect(() => {
    if (isPlaying && hasData) {
      tickerRef.current = setInterval(() => {
        setCurrentTime((p) => {
          const n = p + playbackSpeed * 0.1;
          if (n >= rangeEnd) {
            setIsPlaying(false);
            return rangeEnd;
          }
          return n;
        });
      }, 100);
    } else if (tickerRef.current) {
      clearInterval(tickerRef.current);
      tickerRef.current = null;
    }
    return () => {
      if (tickerRef.current) {
        clearInterval(tickerRef.current);
        tickerRef.current = null;
      }
    };
  }, [isPlaying, playbackSpeed, hasData, rangeEnd]);

  // current row
  const currentRow = useMemo(() => {
    if (rows.length === 0) return null;
    let lo = 0, hi = rows.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (rows[mid].__sec < currentTime) lo = mid + 1; else hi = mid;
    }
    const a = rows[lo];
    const b = rows[Math.max(0, lo - 1)];
    if (!b) return a;
    return Math.abs(a.__sec - currentTime) < Math.abs(b.__sec - currentTime) ? a : b;
  }, [rows, currentTime]);

  // neighbors
  function getImmediateNeighbors(all: LabelSegment[], selfId: string) {
    const others = all.filter((s) => s.id !== selfId).sort((a, b) => a.startTime - b.startTime);
    const self = all.find((s) => s.id === selfId)!;
    const left = [...others].filter((o) => o.endTime <= self.startTime).slice(-1)[0] || null;
    const right = others.find((o) => o.startTime >= self.endTime) || null;
    return { leftEnd: left ? left.endTime : rangeStart, rightStart: right ? right.startTime : rangeEnd };
  }

  // save
  const saveLabels = async () => {
    try {
      if (!selectedFile) return alert("Select a file first.");
      if (labelSegments.length === 0) return alert("No labels to save.");
      setSaving(true);
      const startMs = rows.length ? Date.parse(rows[0].__tsISO) - Math.floor(rows[0].__sec * 1000) : undefined;
      if (!startMs) throw new Error("Could not infer start time");
      const ranges = labelSegments.map((seg) => {
        const s = startMs + Math.floor(seg.startTime * 1000);
        const e = startMs + Math.floor(seg.endTime * 1000);
        return {
          startTs: new Date(Math.min(s, e)).toISOString(),
          endTs: new Date(Math.max(s, e)).toISOString(),
          label: seg.label,
        };
      });
      const res = await fetch("/api/label-trip", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ 
          sourcePath: selectedFile, 
          timestampColumn: "timestamp", 
          ranges,
          sourceType: fileType // Pass the source type (raw/processed)
        }),
      });
      const json = await res.json();
      if (!res.ok || !json?.ok) throw new Error(json?.error || "Save failed");
      alert(json?.labeledPath ? `Saved:\n${json.labeledPath}` : "Labeled file created.");
    } catch (e: any) {
      console.error("[save]", e);
      alert(e?.message || "Save failed");
    } finally {
      setSaving(false);
    }
  };

  // Filter files based on search term
  const filteredFiles = useMemo(() => {
    if (fileType === "labeled") {
      return labeledFiles.filter(file => 
        file.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        file.sessionId.includes(searchTerm) ||
        file.date.includes(searchTerm)
      );
    }
    return [];
  }, [fileType, labeledFiles, searchTerm]);

  // Get current file info
  const currentFileInfo = useMemo(() => {
    if (fileType === "labeled" && selectedFile) {
      return labeledFiles.find(f => f.name === selectedFile);
    }
    return null;
  }, [fileType, selectedFile, labeledFiles]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database className="h-5 w-5" />
            Manual Labeling Tool
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Load raw/processed files, add labels, and save to labeled storage with proper naming convention
          </p>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap items-center gap-4 mb-4">
            <div className="flex items-center gap-2">
              <Label>Source:</Label>
              <Select value={fileType} onValueChange={(v: FileType) => { setFileType(v); setSelectedFile(""); }}>
                <SelectTrigger className="w-40">
                  <SelectValue placeholder="Type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="raw">Raw</SelectItem>
                  <SelectItem value="processed">Processed</SelectItem>
                  <SelectItem value="labeled">Labeled</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            {(fileType === "raw" || fileType === "processed") && (
              <div className="flex items-center gap-2 flex-1">
                <Search className="h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder={`Search ${fileType} files...`}
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="max-w-xs"
                />
              </div>
            )}
            
            {fileType === "labeled" && (
              <div className="flex items-center gap-2 flex-1">
                <Search className="h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search labeled files..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="max-w-xs"
                />
              </div>
            )}
          </div>

          <div className="flex flex-wrap items-center gap-4">
            <div className="flex items-center gap-2">
              <Label>File:</Label>
              <Select value={selectedFile} onValueChange={(v) => setSelectedFile(v)}>
                <SelectTrigger className="w-[420px]">
                  <SelectValue placeholder={`Select ${fileType} CSV...`} />
                </SelectTrigger>
                <SelectContent>
                  {fileType === "labeled" ? (
                    filteredFiles.map((f) => (
                      <SelectItem key={f.name} value={f.name}>
                        <div className="flex items-center gap-2">
                          <span>{f.name.split("/").pop()}</span>
                          <Badge variant="outline" className="text-xs">
                            {f.sessionId}
                          </Badge>
                        </div>
                      </SelectItem>
                    ))
                  ) : fileType === "raw" ? (
                    rawFiles
                      .filter(f => searchTerm === "" || f.toLowerCase().includes(searchTerm.toLowerCase()))
                      .map((f) => (
                        <SelectItem key={f} value={f}>
                          <div className="flex items-center gap-2">
                            <span>{f.split("/").pop()}</span>
                            <Badge variant="outline" className="text-xs">Raw</Badge>
                          </div>
                        </SelectItem>
                      ))
                  ) : fileType === "processed" ? (
                    processedFiles
                      .filter(f => searchTerm === "" || f.toLowerCase().includes(searchTerm.toLowerCase()))
                      .map((f) => (
                        <SelectItem key={f} value={f}>
                          <div className="flex items-center gap-2">
                            <span>{f.split("/").pop()}</span>
                            <Badge variant="outline" className="text-xs">Processed</Badge>
                          </div>
                        </SelectItem>
                      ))
                  ) : null}
                </SelectContent>
              </Select>
            </div>

            <div className="flex items-center gap-2">
              <Label>Signals:</Label>
              <Select value={selectedCols.join(",")} onValueChange={() => {}}>
                <SelectTrigger className="w-[280px]">
                  <SelectValue placeholder="Choose up to 4 signals" />
                </SelectTrigger>
                <SelectContent>
                  {numericCols.map((c) => {
                    const active = selectedCols.includes(c);
                    return (
                      <div
                        key={c}
                        className={`px-2 py-1 text-sm cursor-pointer rounded ${active ? "bg-primary/10" : ""}`}
                        onClick={() => {
                          setSelectedCols((prev) => {
                            if (active) return prev.filter((x) => x !== c);
                            if (prev.length >= 4) return prev;
                            return [...prev, c];
                          });
                        }}
                      >
                        {active ? "✓ " : ""}
                        {c}
                      </div>
                    );
                  })}
                </SelectContent>
              </Select>
            </div>

            <Button
              variant="outline"
              size="sm"
              onClick={() => window.location.reload()}
              disabled={loading}
            >
              <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            </Button>

            <Badge variant="outline">
              Duration: {hasData ? fmtHMS(rangeEnd - rangeStart) : "—:—:—"}
            </Badge>
          </div>

          {/* File Info */}
          {currentFileInfo && (
            <div className="mt-4 p-4 bg-muted rounded-lg">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="flex items-center gap-2">
                  <FileText className="h-4 w-4 text-muted-foreground" />
                  <div>
                    <p className="text-sm font-medium">Session ID</p>
                    <p className="text-sm text-muted-foreground">{currentFileInfo.sessionId}</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Clock className="h-4 w-4 text-muted-foreground" />
                  <div>
                    <p className="text-sm font-medium">Duration</p>
                    <p className="text-sm text-muted-foreground">{formatDuration(currentFileInfo.duration)}</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Activity className="h-4 w-4 text-muted-foreground" />
                  <div>
                    <p className="text-sm font-medium">Size</p>
                    <p className="text-sm text-muted-foreground">{formatFileSize(currentFileInfo.size)}</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Calendar className="h-4 w-4 text-muted-foreground" />
                  <div>
                    <p className="text-sm font-medium">Date</p>
                    <p className="text-sm text-muted-foreground">{currentFileInfo.date}</p>
                  </div>
                </div>
              </div>
              
              {currentFileInfo.totalSegments !== undefined && (
                <div className="mt-3 pt-3 border-t">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <CheckCircle className="h-4 w-4 text-green-500" />
                      <span className="text-sm font-medium">Labeling Progress</span>
                    </div>
                    <span className="text-sm text-muted-foreground">
                      {currentFileInfo.labeledSegments || 0} / {currentFileInfo.totalSegments} segments
                    </span>
                  </div>
                  <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-green-500 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${currentFileInfo.completionRate || 0}%` }}
                    />
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {currentFileInfo.completionRate || 0}% complete
                  </p>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Signals (scrub to navigate)</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={rows}
                onClick={(e: any) => {
                  const sec = (typeof e?.activeLabel === "number" && e.activeLabel) || e?.activePayload?.[0]?.payload?.__sec || null;
                  if (typeof sec === "number") setCurrentTime(clampToData(sec));
                }}
                onMouseMove={(e: any) => {
                  const sec = (typeof e?.activeLabel === "number" && e.activeLabel) || e?.activePayload?.[0]?.payload?.__sec || null;
                  if (typeof sec === "number") setCurrentTime(clampToData(sec));
                }}
                margin={{ top: 10, right: GUTTER_PX, left: GUTTER_PX, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="__sec" tickFormatter={fmtHMS} type="number" domain={[rangeStart, Math.max(rangeStart + 1, rangeEnd)]} />
                <YAxis />
                <Tooltip labelFormatter={(v) => fmtHMS(Number(v))} />
                <Legend />
                {selectedCols.map((c) => (
                  <Line key={c} type="monotone" dataKey={c} dot={false} strokeWidth={2} isAnimationActive={false} />
                ))}
                <ReferenceLine x={currentTime} stroke="red" strokeDasharray="4 4" />
                <Brush dataKey="__sec" tickFormatter={fmtHMS} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Timeline + transport */}
      <Card>
        <CardHeader>
          <CardTitle>Interactive Timeline</CardTitle>
          <p className="text-sm text-muted-foreground">
            <strong>Step 1:</strong> Click on timeline to set start position → <strong>Step 2:</strong> Shift+Click to set end position → <strong>Step 3:</strong> Click "Add/Overwrite Label" button (automatically overwrites overlapping labels)
          </p>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center gap-4">
              <Button variant="outline" size="sm" onClick={() => setIsPlaying((p) => !p)} disabled={!hasData}>
                {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  setCurrentTime(rangeStart);
                  setIsPlaying(false);
                }}
                disabled={!hasData}
              >
                <RotateCcw className="h-4 w-4" />
              </Button>
              <div className="flex items-center gap-2">
                <Label className="text-sm">Speed:</Label>
                <Select value={String(playbackSpeed)} onValueChange={(v) => setPlaybackSpeed(Number(v))}>
                  <SelectTrigger className="w-20">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="0.5">0.5x</SelectItem>
                    <SelectItem value="1">1x</SelectItem>
                    <SelectItem value="2">2x</SelectItem>
                    <SelectItem value="5">5x</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="text-sm text-muted-foreground">{fmtHMS(currentTime - rangeStart)} / {fmtHMS(rangeEnd - rangeStart)}</div>
            </div>

            <div
              ref={timelineRef}
              className="relative h-24 bg-muted rounded-lg cursor-pointer border select-none"
              onClick={(e) => {
                if (!hasData) return;
                const f = xToFrac(e.clientX, timelineRef.current);
                const t = fracToSec(f);
                if (e.shiftKey && selectionStart !== null) setSelectionEnd(t);
                else {
                  setSelectionStart(t);
                  setSelectionEnd(null);
                }
                setCurrentTime(t);
                setSelectedSegmentId(null);
              }}
            >
              <div className="absolute inset-y-0" style={{ left: GUTTER_PX, right: GUTTER_PX }}>
                {labelSegments.map((seg) => {
                  const leftPct = ((seg.startTime - rangeStart) / rangeLen) * 100;
                  const widthPct = ((seg.endTime - seg.startTime) / rangeLen) * 100;
                  return (
                    <div
                      key={seg.id}
                      className={`absolute top-0 h-full ${getLabelColor(seg.label)} opacity-60 border-l-2 border-r-2 border-primary rounded ${selectedSegmentId === seg.id ? "ring-2 ring-primary" : ""}`}
                      style={{ left: `${leftPct}%`, width: `${widthPct}%` }}
                      onMouseDown={(e) => { e.stopPropagation(); setSelectedSegmentId(seg.id); }}
                    >
                      <div className="text-xs p-1 text-white font-medium pointer-events-none select-none">{seg.label}</div>
                      <div
                        className="absolute left-0 top-0 h-full w-2 cursor-ew-resize bg-primary/70 hover:bg-primary"
                        onMouseDown={(e) => {
                          e.preventDefault(); e.stopPropagation(); setSelectedSegmentId(seg.id);
                          const { leftEnd } = getImmediateNeighbors(labelSegments, seg.id);
                          const leftBound = Math.max(leftEnd, rangeStart);
                          const rightBound = seg.endTime - MIN_BLOCK_SEC;
                          const startClientX = e.clientX; const origStart = seg.startTime; const origEnd = seg.endTime;
                          const onMove = (me: MouseEvent) => {
                            let newStart = origStart + pxDeltaToSec(me.clientX - startClientX);
                            newStart = clamp(newStart, leftBound, rightBound);
                            newStart = clampToData(newStart);
                            setLabelSegments((prev) => prev.map((s) => (s.id === seg.id ? { ...s, startTime: newStart } : s)));
                          };
                          const onUp = () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
                          window.addEventListener("mousemove", onMove);
                          window.addEventListener("mouseup", onUp);
                        }}
                      />
                      <div
                        className="absolute right-0 top-0 h-full w-2 cursor-ew-resize bg-primary/70 hover:bg-primary"
                        onMouseDown={(e) => {
                          e.preventDefault(); e.stopPropagation(); setSelectedSegmentId(seg.id);
                          const { rightStart } = getImmediateNeighbors(labelSegments, seg.id);
                          const leftBound = seg.startTime + MIN_BLOCK_SEC;
                          const rightBound = Math.min(rightStart, rangeEnd);
                          const startClientX = e.clientX; const origStart = seg.startTime; const origEnd = seg.endTime;
                          const onMove = (me: MouseEvent) => {
                            let newEnd = origEnd + pxDeltaToSec(me.clientX - startClientX);
                            newEnd = clamp(newEnd, leftBound, rightBound);
                            newEnd = clampToData(newEnd);
                            setLabelSegments((prev) => prev.map((s) => (s.id === seg.id ? { ...s, endTime: newEnd } : s)));
                          };
                          const onUp = () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
                          window.addEventListener("mousemove", onMove);
                          window.addEventListener("mouseup", onUp);
                        }}
                      />
                      <div
                        className="absolute inset-y-0 left-2 right-2 cursor-grab active:cursor-grabbing"
                        onMouseDown={(e) => {
                          e.preventDefault(); e.stopPropagation(); setSelectedSegmentId(seg.id);
                          const { leftEnd, rightStart } = getImmediateNeighbors(labelSegments, seg.id);
                          const width = seg.endTime - seg.startTime;
                          const leftBound = Math.max(leftEnd, rangeStart);
                          const rightBound = Math.min(rightStart - width, rangeEnd - width);
                          const startClientX = e.clientX; const origStart = seg.startTime; const origEnd = seg.endTime;
                          const onMove = (me: MouseEvent) => {
                            let newStart = origStart + pxDeltaToSec(me.clientX - startClientX);
                            newStart = clamp(newStart, leftBound, rightBound);
                            newStart = clampToData(newStart);
                            let newEnd = newStart + width; newEnd = clampToData(newEnd);
                            setLabelSegments((prev) => prev.map((s) => (s.id === seg.id ? { ...s, startTime: newStart, endTime: newEnd } : s)));
                          };
                          const onUp = () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
                          window.addEventListener("mousemove", onMove);
                          window.addEventListener("mouseup", onUp);
                        }}
                      />
                    </div>
                  );
                })}

                {/* Selection indicators */}
                {selectionStart !== null && (
                  <div
                    className="absolute top-0 h-full bg-primary/20 border-l-2 border-primary"
                    style={{
                      left: `${((selectionStart - rangeStart) / rangeLen) * 100}%`,
                      width: selectionEnd !== null 
                        ? `${(Math.abs(selectionEnd - selectionStart) / rangeLen) * 100}%`
                        : '2px',
                    }}
                  />
                )}
                
                {selectionStart !== null && selectionEnd === null && (
                  <div
                    className="absolute top-0 h-full w-1 bg-primary border border-primary"
                    style={{
                      left: `${((selectionStart - rangeStart) / rangeLen) * 100}%`,
                    }}
                  />
                )}

                <div
                  className="absolute top-0 h-full w-0.5 bg-destructive z-10"
                  style={{ left: `${((currentTime - rangeStart) / rangeLen) * 100}%` }}
                />
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Current Sample + labeling panel */}
      <Card>
        <CardHeader>
          <CardTitle>Current Sample</CardTitle>
        </CardHeader>
        <CardContent>
          {currentRow ? (
            <div className={`grid grid-cols-${Math.max(1, Math.min(5, selectedCols.length || 1))} gap-4 p-4 bg-card rounded-lg`}>
              {selectedCols.length === 0 && <div className="text-sm text-muted-foreground">Pick signals above to display.</div>}
              {selectedCols.map((key) => (
                <div key={key} className="text-center">
                  <div className="text-2xl font-bold">{Number.isFinite(currentRow[key]) ? currentRow[key] : "—"}</div>
                  <div className="text-sm text-muted-foreground">{key}</div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-sm text-muted-foreground">No data.</div>
          )}
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Assign Label</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label>Driving Style</Label>
              <Select value={selectedLabel} onValueChange={(v: DrivingLabel) => setSelectedLabel(v)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="idle">Idle</SelectItem>
                  <SelectItem value="passive">Passive</SelectItem>
                  <SelectItem value="moderate">Moderate</SelectItem>
                  <SelectItem value="aggressive">Aggressive</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Selection Status */}
            <div className="p-3 bg-muted rounded-lg text-sm">
              <div className="font-medium">Selection Status</div>
              <div className="text-muted-foreground">
                {selectionStart === null ? (
                  "Click on timeline to start selection"
                ) : selectionEnd === null ? (
                  `Start: ${fmtHMS(selectionStart - rangeStart)} - Shift+Click to set end`
                ) : (
                  `Selected: ${fmtHMS(Math.min(selectionStart, selectionEnd) - rangeStart)} — ${fmtHMS(Math.max(selectionStart, selectionEnd) - rangeStart)}`
                )}
              </div>
            </div>

            <div className="flex gap-2">
              <Button
                onClick={() => {
                  if (!hasData) return;
                  if (selectionStart === null || selectionEnd === null) return;
                  const start = clampToData(Math.min(selectionStart, selectionEnd));
                  const end = clampToData(Math.max(selectionStart, selectionEnd));
                  
                  // Remove any overlapping segments and add the new one
                  setLabelSegments((prev) => {
                    // Filter out segments that overlap with the new range
                    const filtered = prev.filter(seg => {
                      const segStart = seg.startTime;
                      const segEnd = seg.endTime;
                      // Check if segments don't overlap
                      return !(Math.max(segStart, start) < Math.min(segEnd, end));
                    });
                    
                    // Add the new segment
                    return [
                      ...filtered,
                      { id: crypto.randomUUID(), startTime: start, endTime: end, label: selectedLabel, confidence: 1 },
                    ];
                  });
                  
                  setSelectionStart(null);
                  setSelectionEnd(null);
                }}
                disabled={!hasData || selectionStart === null || selectionEnd === null}
                className="flex-1"
              >
                <TagIcon className="h-4 w-4 mr-2" />
                Add/Overwrite Label
              </Button>
              
              <Button
                variant="outline"
                onClick={() => {
                  setSelectionStart(null);
                  setSelectionEnd(null);
                }}
                disabled={selectionStart === null && selectionEnd === null}
              >
                Clear
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Current Labels</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {labelSegments.map((seg) => (
                <div key={seg.id} className={`flex items-center justify-between p-2 rounded border ${selectedSegmentId === seg.id ? "bg-primary/10 border-primary" : "bg-muted"}`}>
                  <div className="flex items-center gap-2">
                    <div className={`w-3 h-3 rounded ${getLabelColor(seg.label)}`} />
                    <button className="font-medium capitalize text-left" onClick={() => setSelectedSegmentId(seg.id)}>{seg.label}</button>
                    <span className="text-sm text-muted-foreground">{fmtHMS(seg.startTime - rangeStart)} - {fmtHMS(seg.endTime - rangeStart)}</span>
                  </div>
                  <Button variant="ghost" size="sm" onClick={() => setLabelSegments((p) => p.filter((x) => x.id !== seg.id))}>
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              ))}
              {labelSegments.length === 0 && (
                <div className="text-sm text-muted-foreground p-2">No labels yet. Select a range and click "Add Label".</div>
              )}
            </div>
            <Separator className="my-4" />
            <Button className="w-full" onClick={saveLabels} disabled={saving || !selectedFile}>
              {saving ? "Saving…" : "Save Labels to Database"}
            </Button>
            <p className="text-xs text-muted-foreground mt-2">
              Files will be saved as: 001_{fileType}-002_2025-09-19-labelled.csv
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}