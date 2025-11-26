"use client";

import { useEffect, useState } from "react";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Calendar, Clock, TrendingUp, RefreshCw, Loader2, AlertCircle, BarChart3 } from "lucide-react";

interface ProcessedTrip {
  filename: string;
  date: string;
  sessionId: string;
  startTime?: string;
  endTime?: string;
  duration?: string;
  drivingStyles: {
    idle: number;
    passive: number;
    moderate: number;
    aggressive: number;
  };
  efficiencyScore?: number;
  efficiencyStatus?: string;
  loading: boolean;
  error?: string;
}

export default function ProcessedTripsSection() {
  const [trips, setTrips] = useState<ProcessedTrip[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const getEfficiencyColor = (score?: number) => {
    if (!score) return "text-gray-600";
    if (score >= 85) return "text-green-600";
    if (score >= 70) return "text-blue-600";
    return "text-orange-600";
  };

  const getEfficiencyBadge = (score?: number) => {
    if (!score) return { variant: "outline" as const, label: "N/A" };
    if (score >= 85) return { variant: "default" as const, label: "Excellent" };
    if (score >= 70) return { variant: "secondary" as const, label: "Good" };
    return { variant: "destructive" as const, label: "Needs Work" };
  };

  const parseCSV = (text: string): { headers: string[], rows: string[][] } => {
    const lines = text.split(/\r?\n/).filter(Boolean);
    if (lines.length === 0) return { headers: [], rows: [] };
    
    const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
    const rows = lines.slice(1).map(line => line.split(',').map(cell => cell.trim()));
    
    return { headers, rows };
  };

  const analyzeDrivingStyles = (csvData: { headers: string[], rows: string[][] }) => {
    const { headers, rows } = csvData;
    const styleIdx = headers.findIndex(h => h === 'driving_style' || h === 'drivingstyle');
    const timestampIdx = headers.findIndex(h => h === 'timestamp' || h === 'time' || h === 'ts');

    if (styleIdx === -1) {
      return {
        styles: { idle: 0, passive: 0, moderate: 0, aggressive: 0 },
        startTime: undefined,
        endTime: undefined
      };
    }

    const styleCounts = { idle: 0, passive: 0, moderate: 0, aggressive: 0 };
    let startTime: string | undefined;
    let endTime: string | undefined;

    rows.forEach((row, idx) => {
      const style = (row[styleIdx] || '').toLowerCase();
      if (style in styleCounts) {
        styleCounts[style as keyof typeof styleCounts]++;
      }

      if (timestampIdx !== -1 && row[timestampIdx]) {
        const timestamp = row[timestampIdx];
        if (idx === 0) startTime = timestamp;
        if (idx === rows.length - 1) endTime = timestamp;
      }
    });

    const total = Object.values(styleCounts).reduce((sum, count) => sum + count, 0);
    const styles = {
      idle: total > 0 ? Math.round((styleCounts.idle / total) * 100) : 0,
      passive: total > 0 ? Math.round((styleCounts.passive / total) * 100) : 0,
      moderate: total > 0 ? Math.round((styleCounts.moderate / total) * 100) : 0,
      aggressive: total > 0 ? Math.round((styleCounts.aggressive / total) * 100) : 0,
    };

    return { styles, startTime, endTime };
  };

  const formatTime = (timestamp?: string) => {
    if (!timestamp) return undefined;
    
    const num = Number(timestamp);
    let date: Date;
    
    if (!isNaN(num)) {
      date = new Date(num >= 1e11 ? num : num * 1000);
    } else {
      date = new Date(timestamp);
    }
    
    if (isNaN(date.getTime())) return undefined;
    
    return date.toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit',
      hour12: true 
    });
  };

  const calculateDuration = (start?: string, end?: string) => {
    if (!start || !end) return undefined;
    
    const parseTs = (ts: string) => {
      const num = Number(ts);
      if (!isNaN(num)) return num >= 1e11 ? num : num * 1000;
      return Date.parse(ts);
    };

    const startMs = parseTs(start);
    const endMs = parseTs(end);
    
    if (isNaN(startMs) || isNaN(endMs)) return undefined;
    
    const durationMin = Math.round((endMs - startMs) / 60000);
    if (durationMin < 60) return `${durationMin}m`;
    
    const hours = Math.floor(durationMin / 60);
    const minutes = durationMin % 60;
    return `${hours}h ${minutes}m`;
  };

  const fetchEfficiencyScore = async (filename: string, tripIndex: number) => {
    try {
      const response = await fetch(`https://binkhoale1812-obd-logger.hf.space/efficiency/${filename}`);
      const data = await response.json();
      
      setTrips(prev => prev.map((trip, idx) => 
        idx === tripIndex 
          ? { 
              ...trip, 
              efficiencyScore: data.efficiency_score,
              efficiencyStatus: data.status,
              loading: false 
            }
          : trip
      ));
    } catch (error) {
      console.error(`Error fetching efficiency for ${filename}:`, error);
      setTrips(prev => prev.map((trip, idx) => 
        idx === tripIndex 
          ? { 
              ...trip, 
              loading: false,
              error: 'Failed to load efficiency score'
            }
          : trip
      ));
    }
  };

  const loadProcessedTrips = async () => {
    setLoading(true);
    setError(null);

    try {
      // Fetch list of files from Firebase Storage
      const response = await fetch('/api/list-processed-files');
      const data = await response.json();

      if (!data.success || !data.files) {
        throw new Error('Failed to fetch processed files');
      }

      const fileList = data.files as string[];
      const processedTrips: ProcessedTrip[] = [];

      // Sort files by date (newest first) - extract date from filename
      const sortedFiles = fileList.sort((a, b) => {
        const dateA = a.match(/\d{4}-\d{2}-\d{2}/)?.[0] || '';
        const dateB = b.match(/\d{4}-\d{2}-\d{2}/)?.[0] || '';
        return dateB.localeCompare(dateA); // Descending order (newest first)
      });

      // Initialize trips array with placeholders
      const initialTrips: ProcessedTrip[] = sortedFiles
        .map(filename => {
          const match = filename.match(/(\d{3})_(\d{4}-\d{2}-\d{2})_processed\.csv/);
          if (!match) return null;
          const [, sessionId, date] = match;
          return {
            filename,
            date,
            sessionId,
            drivingStyles: { idle: 0, passive: 0, moderate: 0, aggressive: 0 },
            loading: true
          };
        })
        .filter((t): t is ProcessedTrip => t !== null);

      setTrips(initialTrips);

      // Process each file
      for (let i = 0; i < sortedFiles.length; i++) {
        const filename = sortedFiles[i];
        
        // Parse filename: 001_2025-10-01_processed.csv
        const match = filename.match(/(\d{3})_(\d{4}-\d{2}-\d{2})_processed\.csv/);
        if (!match) continue;

        const [, sessionId, date] = match;

        // Fetch efficiency score first
        fetchEfficiencyScore(filename, i);

        // Then fetch and parse CSV data
        try {
          const csvResponse = await fetch(`/api/get-processed-file?filename=${encodeURIComponent(filename)}`);
          const csvText = await csvResponse.text();

          // Parse CSV and analyze
          const csvData = parseCSV(csvText);
          const { styles, startTime, endTime } = analyzeDrivingStyles(csvData);

          // Update trip with CSV data while preserving efficiency score
          setTrips(prev => prev.map((trip, idx) => 
            idx === i 
              ? {
                  ...trip,
                  startTime: formatTime(startTime),
                  endTime: formatTime(endTime),
                  duration: calculateDuration(startTime, endTime),
                  drivingStyles: styles,
                }
              : trip
          ));
        } catch (err) {
          console.error(`Error processing ${filename}:`, err);
        }
      }

      setLoading(false);
    } catch (err) {
      console.error('Error loading processed trips:', err);
      setError(err instanceof Error ? err.message : 'Failed to load processed trips');
      setLoading(false);
    }
  };

  useEffect(() => {
    loadProcessedTrips();
  }, []);

  const getDrivingStyleColor = (style: string) => {
    switch (style) {
      case 'idle': return 'bg-gray-500';
      case 'passive': return 'bg-green-500';
      case 'moderate': return 'bg-blue-500';
      case 'aggressive': return 'bg-red-500';
      default: return 'bg-gray-400';
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Processed Trips Analysis
              </CardTitle>
              <CardDescription className="mt-1">
                View driving style distribution and efficiency scores from processed OBD-II data
              </CardDescription>
            </div>
            <Button onClick={loadProcessedTrips} disabled={loading} variant="outline">
              <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {error && (
            <div className="flex items-center gap-2 p-4 mb-4 bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-900 rounded-lg">
              <AlertCircle className="h-5 w-5 text-red-600" />
              <span className="text-sm text-red-900 dark:text-red-100">{error}</span>
            </div>
          )}

          {loading && trips.length === 0 ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
              <span className="ml-3 text-muted-foreground">Loading processed trips...</span>
            </div>
          ) : trips.length === 0 ? (
            <div className="text-center py-12 text-muted-foreground">
              No processed trips found
            </div>
          ) : (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {trips.map((trip, idx) => {
                const effBadge = getEfficiencyBadge(trip.efficiencyScore);
                
                return (
                  <Card key={idx} className="hover:shadow-md transition-shadow">
                    <CardHeader className="pb-3">
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex-1">
                          <div className="flex items-center gap-2">
                            <Badge variant="outline">Session {trip.sessionId}</Badge>
                            <span className="text-sm text-muted-foreground">•</span>
                            <div className="flex items-center gap-1 text-sm text-muted-foreground">
                              <Calendar className="h-3 w-3" />
                              {new Date(trip.date).toLocaleDateString('en-US', { 
                                month: 'short', 
                                day: 'numeric', 
                                year: 'numeric' 
                              })}
                            </div>
                          </div>
                          {trip.startTime && trip.endTime && (
                            <div className="flex items-center gap-2 mt-1 text-xs text-muted-foreground">
                              <Clock className="h-3 w-3" />
                              {trip.startTime} - {trip.endTime}
                              {trip.duration && <span>• {trip.duration}</span>}
                            </div>
                          )}
                        </div>
                        {trip.loading ? (
                          <div className="flex items-center gap-2 text-sm text-muted-foreground">
                            <Loader2 className="h-4 w-4 animate-spin" />
                            Loading...
                          </div>
                        ) : trip.error ? (
                          <Badge variant="destructive">Error</Badge>
                        ) : (
                          <Badge variant={effBadge.variant} className="shrink-0">
                            {trip.efficiencyScore ? Math.round(trip.efficiencyScore) : 'N/A'}
                          </Badge>
                        )}
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      {/* Efficiency Score */}
                      {!trip.loading && trip.efficiencyScore && (
                        <div className="space-y-1">
                          <div className="flex items-center justify-between text-sm">
                            <span className="text-muted-foreground">Driving Efficiency</span>
                            <span className={`font-semibold ${getEfficiencyColor(trip.efficiencyScore)}`}>
                              {effBadge.label} ({Math.round(trip.efficiencyScore)}/100)
                            </span>
                          </div>
                          <Progress value={trip.efficiencyScore} className="h-2" />
                        </div>
                      )}

                      {/* Driving Styles Distribution */}
                      <div className="space-y-2">
                        <h4 className="text-sm font-medium">Driving Style Distribution</h4>
                        <div className="space-y-2">
                          {Object.entries(trip.drivingStyles).map(([style, percentage]) => (
                            <div key={style} className="space-y-1">
                              <div className="flex items-center justify-between text-sm">
                                <span className="capitalize text-muted-foreground">{style}</span>
                                <span className="font-medium">{percentage}%</span>
                              </div>
                              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                                <div
                                  className={`h-2 rounded-full ${getDrivingStyleColor(style)}`}
                                  style={{ width: `${percentage}%` }}
                                />
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* File Info */}
                      <div className="pt-2 border-t">
                        <p className="text-xs text-muted-foreground">
                          File: {trip.filename}
                        </p>
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Summary Statistics */}
      {trips.length > 0 && (
        <div className="grid gap-4 md:grid-cols-3">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Trips</CardTitle>
              <BarChart3 className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{trips.length}</div>
              <p className="text-xs text-muted-foreground mt-1">Processed sessions</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Avg Efficiency</CardTitle>
              <TrendingUp className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {trips.filter(t => t.efficiencyScore).length > 0
                  ? Math.round(
                      trips.reduce((sum, t) => sum + (t.efficiencyScore || 0), 0) /
                      trips.filter(t => t.efficiencyScore).length
                    )
                  : 'N/A'}
              </div>
              <p className="text-xs text-muted-foreground mt-1">Across all trips</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Latest Session</CardTitle>
              <Calendar className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{trips[0]?.sessionId || 'N/A'}</div>
              <p className="text-xs text-muted-foreground mt-1">
                {trips[0] ? new Date(trips[0].date).toLocaleDateString() : 'No data'}
              </p>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}