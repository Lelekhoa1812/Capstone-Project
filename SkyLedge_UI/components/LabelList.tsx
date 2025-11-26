// components/LabelList.tsx
"use client";

import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import type { LabelSegment } from "@/lib/types";
import { Trash2 } from "lucide-react";

type Props = {
  labels: LabelSegment[];
  selectedId: string | null;
  setSelectedId: (id: string | null) => void;
  removeLabel: (id: string) => void;
  formatTime: (sec: number) => string;
  getLabelColor: (label: LabelSegment["label"]) => string;
  onSave: () => Promise<void>;
  saving: boolean;
  canSave: boolean;
};

export default function LabelList({
  labels, selectedId, setSelectedId, removeLabel, formatTime, getLabelColor, onSave, saving, canSave
}: Props) {
  return (
    <div>
      <div className="space-y-2 max-h-64 overflow-y-auto">
        {labels.map((seg) => (
          <div
            key={seg.id}
            className={`flex items-center justify-between p-2 rounded border ${selectedId === seg.id ? "bg-primary/10 border-primary" : "bg-muted"}`}
          >
            <div className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded ${getLabelColor(seg.label)}`} />
              <button className="font-medium capitalize text-left" onClick={() => setSelectedId(seg.id)} title="Select segment">
                {seg.label}
              </button>
              <span className="text-sm text-muted-foreground">
                {formatTime(seg.startTime)} - {formatTime(seg.endTime)}
              </span>
            </div>
            <Button variant="ghost" size="sm" onClick={() => removeLabel(seg.id)}>
              <Trash2 className="h-4 w-4" />
            </Button>
          </div>
        ))}
        {labels.length === 0 && (
          <div className="text-sm text-muted-foreground p-2">No labels yet. Select a range and click “Add Label”.</div>
        )}
      </div>

      <Separator className="my-4" />
      <Button className="w-full" onClick={onSave} disabled={saving || !canSave}>
        {saving ? "Saving…" : "Save Labels to Database"}
      </Button>
    </div>
  );
}
