// Make sure this file exists and the export is named:
export const mockTripData = {
  tripId: "trip_001",
  deviceId: "RPI_001",
  startTime: "2024-01-15T12:36:27Z",
  duration: 900,
  dataPoints: Array.from({ length: 900 }, (_, i) => ({
    timestamp: i,
    speed: Math.floor(Math.random() * 60) + 20,
    rpm: Math.floor(Math.random() * 3000) + 1000,
    throttle: Math.floor(Math.random() * 100),
    brake: Math.random() > 0.8 ? Math.floor(Math.random() * 50) : 0,
    fuel: Math.floor(Math.random() * 20) + 80,
  })),
} as const;
