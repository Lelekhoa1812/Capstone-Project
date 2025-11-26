"use client";

import { useState, useEffect } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer 
} from "recharts";
import { 
  Loader2, 
  AlertCircle, 
  Activity,
  Gauge,
  Wind,
  Clock
} from "lucide-react";

interface SensorData {
  index: number;
  rpm: number;
  engineLoad: number;
  intakePressure: number;
}

interface SensorPreviewDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  fileName: string | null;
}

export function SensorPreviewDialog({ open, onOpenChange, fileName }: SensorPreviewDialogProps) {
  const [data, setData] = useState<SensorData[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fileInfo, setFileInfo] = useState<{
    totalSamples: number;
    duration: number;
    sampleRate: number;
  } | null>(null);

  useEffect(() => {
    if (open && fileName) {
      fetchSensorData();
    } else {
      setData([]);
      setError(null);
      setFileInfo(null);
    }
  }, [open, fileName]);

  const fetchSensorData = async () => {
    if (!fileName) return;

    setLoading(true);
    setError(null);

    try {
      console.log(`📊 Fetching sensor data for: ${fileName}`);
      
      const response = await fetch(`/api/sensor-data?fileName=${encodeURIComponent(fileName)}`);
      const result = await response.json();

      if (!result.success) {
        throw new Error(result.error || 'Failed to fetch sensor data');
      }

      console.log(`✅ Sensor data fetched: ${result.data.length} samples`);
      
      setData(result.data);
      setFileInfo(result.fileInfo);
    } catch (error) {
      console.error('❌ Error fetching sensor data:', error);
      setError(String(error));
    } finally {
      setLoading(false);
    }
  };

  const normalizeValue = (value: number, min: number, max: number) => {
    if (max === min) return 0.5;
    return Math.max(0, Math.min(1, (value - min) / (max - min)));
  };

  const processedData = data.map((item, index) => {
    // Normalize values to 0-1 range for better visualization
    const rpmNormalized = normalizeValue(item.rpm, 0, 8000);
    const engineLoadNormalized = normalizeValue(item.engineLoad, 0, 100);
    const intakePressureNormalized = normalizeValue(item.intakePressure, 0, 100);

    return {
      index,
      sample: index,
      rpm: rpmNormalized,
      engineLoad: engineLoadNormalized,
      intakePressure: intakePressureNormalized,
      // Keep original values for tooltip
      rpmOriginal: item.rpm,
      engineLoadOriginal: item.engineLoad,
      intakePressureOriginal: item.intakePressure,
    };
  });

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-background border rounded-lg p-3 shadow-lg">
          <p className="font-medium">Sample {label}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} style={{ color: entry.color }} className="text-sm">
              {entry.dataKey}: {entry.payload[`${entry.dataKey}Original`]} 
              {entry.dataKey === 'rpm' ? ' RPM' : entry.dataKey === 'engineLoad' ? '%' : ' kPa'}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-6xl max-h-[90vh] overflow-hidden">
        <DialogHeader>
          <DialogTitle className="flex items-center space-x-2">
            <Activity className="h-5 w-5" />
            <span>Sensor Data Preview</span>
            {fileName && (
              <Badge variant="outline" className="ml-2">
                {fileName.split('/').pop()}
              </Badge>
            )}
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-4">
          {fileInfo && (
            <div className="grid grid-cols-3 gap-4">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium flex items-center">
                    <Clock className="h-4 w-4 mr-1" />
                    Duration
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-2xl font-bold">
                    {Math.floor(fileInfo.duration / 60)}m {fileInfo.duration % 60}s
                  </p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium flex items-center">
                    <Activity className="h-4 w-4 mr-1" />
                    Samples
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-2xl font-bold">{fileInfo.totalSamples.toLocaleString()}</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium flex items-center">
                    <Gauge className="h-4 w-4 mr-1" />
                    Sample Rate
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-2xl font-bold">{fileInfo.sampleRate.toFixed(1)} Hz</p>
                </CardContent>
              </Card>
            </div>
          )}

          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Sensor Trends (Normalized)</CardTitle>
              <p className="text-sm text-muted-foreground">
                RPM, Engine Load, and Intake Pressure over time (values normalized to 0-1 range)
              </p>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="flex items-center justify-center h-64">
                  <Loader2 className="h-8 w-8 animate-spin mr-2" />
                  <span>Loading sensor data...</span>
                </div>
              ) : error ? (
                <div className="flex items-center justify-center h-64 text-destructive">
                  <AlertCircle className="h-8 w-8 mr-2" />
                  <div>
                    <p className="font-medium">Failed to load sensor data</p>
                    <p className="text-sm text-muted-foreground">{error}</p>
                    <Button 
                      onClick={fetchSensorData} 
                      variant="outline" 
                      size="sm" 
                      className="mt-2"
                    >
                      Retry
                    </Button>
                  </div>
                </div>
              ) : processedData.length === 0 ? (
                <div className="flex items-center justify-center h-64 text-muted-foreground">
                  <AlertCircle className="h-8 w-8 mr-2" />
                  <span>No sensor data available</span>
                </div>
              ) : (
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={processedData}>
                      <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                      <XAxis 
                        dataKey="sample" 
                        type="number"
                        scale="linear"
                        domain={['dataMin', 'dataMax']}
                        tick={{ fontSize: 12 }}
                      />
                      <YAxis 
                        domain={[0, 1]}
                        tick={{ fontSize: 12 }}
                        tickFormatter={(value) => value.toFixed(1)}
                      />
                      <Tooltip content={<CustomTooltip />} />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="rpm"
                        stroke="#3b82f6"
                        strokeWidth={2}
                        dot={false}
                        name="RPM"
                      />
                      <Line
                        type="monotone"
                        dataKey="engineLoad"
                        stroke="#f97316"
                        strokeWidth={2}
                        dot={false}
                        name="Engine Load"
                      />
                      <Line
                        type="monotone"
                        dataKey="intakePressure"
                        stroke="#10b981"
                        strokeWidth={2}
                        dot={false}
                        name="Intake Pressure"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </CardContent>
          </Card>

          <div className="flex justify-end space-x-2">
            <Button variant="outline" onClick={() => onOpenChange(false)}>
              Close
            </Button>
            {!loading && !error && processedData.length > 0 && (
              <Button onClick={fetchSensorData} variant="outline">
                Refresh Data
              </Button>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
