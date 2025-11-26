// components/AssignLabel.tsx
"use client";

import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import type { DrivingLabel } from "@/lib/types";
import { Tag as TagIcon } from "lucide-react";

type Props = {
  timelineDuration: number;
  selectionStart: number | null;
  selectionEnd: number | null;
  selectedLabel: DrivingLabel;
  setSelectedLabel: (l: DrivingLabel) => void;
  addSegment: (start: number, end: number, label: DrivingLabel) => boolean; // return false if overlap blocked
};

export default function AssignLabel({
  timelineDuration,
  selectionStart, selectionEnd,
  selectedLabel, setSelectedLabel,
  addSegment
}: Props) {
  const handleAdd = () => {
    if (timelineDuration <= 0) return;
    if (selectionStart == null || selectionEnd == null) return;
    const start = Math.max(0, Math.min(selectionStart, selectionEnd));
    const end = Math.min(timelineDuration, Math.max(selectionStart, selectionEnd));
    const ok = addSegment(start, end, selectedLabel);
    if (!ok) alert("That range overlaps an existing label. Move/resize or delete existing labels first.");
  };

  return (
    <div className="space-y-4">
      <div>
        <Label>Driving Style</Label>
        <Select value={selectedLabel} onValueChange={(v: DrivingLabel) => setSelectedLabel(v)}>
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="idle">Idle</SelectItem>
            <SelectItem value="passive">Passive</SelectItem>
            <SelectItem value="aggressive">Aggressive</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <Button
        onClick={handleAdd}
        disabled={timelineDuration <= 0 || selectionStart == null || selectionEnd == null}
        className="w-full"
      >
        <TagIcon className="h-4 w-4 mr-2" />
        Add Label
      </Button>
    </div>
  );
}
