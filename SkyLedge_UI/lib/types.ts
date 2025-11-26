export type DrivingLabel = "idle" | "passive" | "moderate" | "aggressive";

export type LabelSegment = {
  id: string;
  startTime: number;
  endTime: number;
  label: DrivingLabel;
  confidence?: number;
};
