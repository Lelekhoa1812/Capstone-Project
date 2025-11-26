export const mockUploadedFiles = [
  {
    id: 1,
    filename: "trip_data_2024_01_15_morning.csv",
    uploadTime: "2024-01-15 08:30:22",
    size: "2.4 MB",
    status: "processed",
    deviceId: "RPI_001",
    tripDuration: "45 min",
    dataPoints: 2700,
    preview: `timestamp,speed,rpm,throttle,brake,fuel_rate
2024-01-15T08:30:00,0,800,0,0,0.8
2024-01-15T08:30:01,5,850,15,0,1.2
2024-01-15T08:30:02,12,900,25,0,1.8
2024-01-15T08:30:03,18,950,35,0,2.1
2024-01-15T08:30:04,25,1000,45,0,2.5`,
  },
  {
    id: 2,
    filename: "highway_drive_2024_01_15.json",
    uploadTime: "2024-01-15 14:22:11",
    size: "1.8 MB",
    status: "pending",
    deviceId: "RPI_002",
    tripDuration: "32 min",
    dataPoints: 1920,
    preview: `{
  "trip_id": "trip_20240115_142211",
  "device_id": "RPI_002",
  "start_time": "2024-01-15T14:22:11",
  "data": [
    {"timestamp":"2024-01-15T14:22:11","obd_data":{"speed":65,"rpm":2200,"throttle_position":45,"engine_load":35,"fuel_rate":8.2}}
  ]
}`,
  },
  {
    id: 3,
    filename: "city_commute_2024_01_16.csv",
    uploadTime: "2024-01-16 07:45:33",
    size: "3.1 MB",
    status: "error",
    deviceId: "RPI_001",
    tripDuration: "52 min",
    dataPoints: 3120,
    preview: `timestamp,speed,rpm,throttle,brake,fuel_rate,error
2024-01-16T07:45:33,0,750,0,0,0.7,
2024-01-16T07:45:34,3,800,10,0,1.0,
2024-01-16T07:45:35,8,850,20,0,1.4,
2024-01-16T07:45:36,ERROR: Missing RPM data
2024-01-16T07:45:37,15,900,30,0,1.8,`,
  },
] as const;
