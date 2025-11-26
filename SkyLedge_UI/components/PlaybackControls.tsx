// components/PlaybackControls.tsx
"use client";

import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Play, Pause, RotateCcw } from "lucide-react";

type Props = {
  isPlaying: boolean;
  setIsPlaying: (v: boolean) => void;
  currentTimeLabel: string;
  durationLabel: string;
  playbackSpeed: number;
  setPlaybackSpeed: (n: number) => void;
  onReset: () => void;
  disabled?: boolean;
};

export default function PlaybackControls({
  isPlaying, setIsPlaying,
  currentTimeLabel, durationLabel,
  playbackSpeed, setPlaybackSpeed,
  onReset, disabled
}: Props) {
  return (
    <div className="flex items-center gap-4">
      <Button variant="outline" size="sm" onClick={() => setIsPlaying(!isPlaying)} disabled={disabled}>
        {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
      </Button>

      <Button variant="outline" size="sm" onClick={onReset} disabled={disabled}>
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

      <div className="text-sm text-muted-foreground">
        {currentTimeLabel} / {durationLabel}
      </div>
    </div>
  );
}
