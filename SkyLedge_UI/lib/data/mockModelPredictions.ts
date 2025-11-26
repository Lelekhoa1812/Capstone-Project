export const mockModelPredictions = [
  {
    id: "pred_001",
    tripId: "trip_001",
    filename: "trip_data_2024_01_15_morning.csv",
    segments: [
      { id: "seg_1", startTime: 0,   endTime: 120, predictedLabel: "idle",       confidence: 0.95, actualLabel: "idle",       isCorrect: true },
      { id: "seg_2", startTime: 120, endTime: 300, predictedLabel: "aggressive", confidence: 0.72, actualLabel: "passive",    isCorrect: false },
      { id: "seg_3", startTime: 300, endTime: 480, predictedLabel: "passive",    confidence: 0.88, actualLabel: "passive",    isCorrect: true },
      { id: "seg_4", startTime: 600, endTime: 720, predictedLabel: "aggressive", confidence: 0.91, actualLabel: "aggressive", isCorrect: true },
    ],
    overallAccuracy: 0.75,
    modelVersion: "v2.1.3",
    processedAt: "2024-01-16 09:30:22",
  },
] as const;
