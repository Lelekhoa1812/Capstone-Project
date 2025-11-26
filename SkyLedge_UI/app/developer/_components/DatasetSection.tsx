"use client";

import { useState, useEffect } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { 
  Search, 
  Filter, 
  Database, 
  Eye, 
  Download, 
  RefreshCw,
  Clock,
  Calendar,
  Activity,
  FileText,
  AlertCircle
} from "lucide-react";

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

const getAccuracyColor = (acc: number) =>
  acc >= 0.9 ? "text-chart-1" : acc >= 0.7 ? "text-chart-3" : "text-chart-4";

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

const getStatusFromCompletion = (completionRate?: number) => {
  if (!completionRate) return "unknown";
  if (completionRate >= 100) return "complete";
  if (completionRate > 0) return "partial";
  return "empty";
};

const getStatusColor = (status: string) => {
  switch (status) {
    case "complete":
      return "bg-green-500";
    case "partial":
      return "bg-yellow-500";
    case "empty":
      return "bg-gray-500";
    default:
      return "bg-gray-500";
  }
};

export default function DatasetSection() {
  const [searchTerm, setSearchTerm] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [labeledFiles, setLabeledFiles] = useState<LabeledFile[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchLabeledFiles = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch("/api/labeled-files-detailed");
      const data = await response.json();
      
      if (data.success) {
        setLabeledFiles(data.files || []);
      } else {
        setError(data.error || "Failed to fetch labeled files");
      }
    } catch (err) {
      console.error("Error fetching labeled files:", err);
      setError("Failed to fetch labeled files");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchLabeledFiles();
  }, []);

  const filtered = labeledFiles.filter((file) => {
    const q = searchTerm.toLowerCase();
    const matches = file.name.toLowerCase().includes(q) || 
                   file.sessionId.toLowerCase().includes(q) ||
                   file.date.includes(q);
    
    const status = getStatusFromCompletion(file.completionRate);
    const matchesStatus = statusFilter === "all" || status === statusFilter;
    
    return matches && matchesStatus;
  });

  const totalTrips = labeledFiles.length;
  const totalSegments = labeledFiles.reduce((sum, file) => sum + (file.totalSegments || 0), 0);
  const labeledSegments = labeledFiles.reduce((sum, file) => sum + (file.labeledSegments || 0), 0);
  const overallCompletionRate = totalSegments > 0 ? Math.round((labeledSegments / totalSegments) * 100) : 0;

  const handleDownload = async (fileName: string) => {
    try {
      const response = await fetch(`/api/download-file?fileName=${encodeURIComponent(fileName)}`);
      
      if (!response.ok) {
        throw new Error('Download failed');
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = fileName.split('/').pop() || 'download.csv';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Download error:', error);
      alert('Failed to download file');
    }
  };

  const handlePreview = (fileName: string) => {
    // For now, just open the file in a new tab
    const downloadUrl = `/api/download-file?fileName=${encodeURIComponent(fileName)}`;
    window.open(downloadUrl, '_blank');
  };

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database className="h-5 w-5" />
            Labeled Dataset
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <RefreshCw className="h-6 w-6 animate-spin mr-2" />
            <span>Loading labeled datasets...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database className="h-5 w-5" />
            Labeled Dataset
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8 text-destructive">
            <AlertCircle className="h-6 w-6 mr-2" />
            <div>
              <p className="font-medium">Failed to load labeled datasets</p>
              <p className="text-sm text-muted-foreground">{error}</p>
              <Button 
                onClick={fetchLabeledFiles} 
                variant="outline" 
                size="sm" 
                className="mt-2"
              >
                <RefreshCw className="h-4 w-4 mr-2" />
                Retry
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Database className="h-5 w-5" />
            Labeled Dataset
          </CardTitle>
          <Button onClick={fetchLabeledFiles} disabled={loading} variant="outline" size="sm">
            <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-4 gap-4 mb-6">
          <div className="text-center p-4 bg-muted rounded-lg">
            <div className="text-2xl font-bold text-primary">{totalTrips}</div>
            <div className="text-sm text-muted-foreground">Total Trips</div>
          </div>
          <div className="text-center p-4 bg-muted rounded-lg">
            <div className="text-2xl font-bold text-chart-1">{totalSegments.toLocaleString()}</div>
            <div className="text-sm text-muted-foreground">Total Segments</div>
          </div>
          <div className="text-center p-4 bg-muted rounded-lg">
            <div className="text-2xl font-bold text-chart-2">{labeledSegments.toLocaleString()}</div>
            <div className="text-sm text-muted-foreground">Labeled Segments</div>
          </div>
          <div className="text-center p-4 bg-muted rounded-lg">
            <div className="text-2xl font-bold text-chart-3">{overallCompletionRate}%</div>
            <div className="text-sm text-muted-foreground">Completion Rate</div>
          </div>
        </div>

        <div className="flex gap-4 mb-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input 
              placeholder="Search by filename, session ID, or date..." 
              value={searchTerm} 
              onChange={(e) => setSearchTerm(e.target.value)} 
              className="pl-10" 
            />
          </div>
          <Select value={statusFilter} onValueChange={setStatusFilter}>
            <SelectTrigger className="w-48">
              <Filter className="h-4 w-4 mr-2" />
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Status</SelectItem>
              <SelectItem value="complete">Complete</SelectItem>
              <SelectItem value="partial">Partial</SelectItem>
              <SelectItem value="empty">Empty</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {filtered.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
            <FileText className="h-12 w-12 mb-4" />
            <h3 className="text-lg font-semibold mb-2">No labeled datasets found</h3>
            <p className="text-center">
              {searchTerm ? 'No files match your search criteria.' : 'No labeled datasets available.'}
            </p>
          </div>
        ) : (
          <div className="rounded-md border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Trip</TableHead>
                  <TableHead>Session ID</TableHead>
                  <TableHead>Date</TableHead>
                  <TableHead>Duration</TableHead>
                  <TableHead>Progress</TableHead>
                  <TableHead>Size</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filtered.map((file) => {
                  const status = getStatusFromCompletion(file.completionRate);
                  return (
                    <TableRow key={file.name}>
                      <TableCell className="font-medium">
                        <div className="flex items-center space-x-2">
                          <FileText className="h-4 w-4 text-muted-foreground" />
                          <span className="truncate max-w-[200px]" title={file.name}>
                            {file.name.split("/").pop()}
                          </span>
                        </div>
                      </TableCell>
                      <TableCell>
                        <Badge variant="outline" className="font-mono">
                          {file.sessionId}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center space-x-1">
                          <Calendar className="h-3 w-3 text-muted-foreground" />
                          <span className="text-sm">
                            {file.date}
                          </span>
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center space-x-1">
                          <Clock className="h-3 w-3 text-muted-foreground" />
                          <span className="text-sm">
                            {formatDuration(file.duration)}
                          </span>
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <div className="flex-1 bg-muted rounded-full h-2">
                            <div 
                              className="bg-primary h-2 rounded-full transition-all duration-300" 
                              style={{ width: `${file.completionRate || 0}%` }} 
                            />
                          </div>
                          <span className="text-sm text-muted-foreground">
                            {file.labeledSegments || 0}/{file.totalSegments || 0}
                          </span>
                        </div>
                      </TableCell>
                      <TableCell className="text-sm text-muted-foreground">
                        {formatFileSize(file.size)}
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <div className={`w-2 h-2 rounded-full ${getStatusColor(status)}`} />
                          <Badge variant={status === "complete" ? "default" : "secondary"}>
                            {status}
                          </Badge>
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => handlePreview(file.name)}
                            className="h-8 w-8 p-0"
                          >
                            <Eye className="h-4 w-4" />
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => handleDownload(file.name)}
                            className="h-8 w-8 p-0"
                          >
                            <Download className="h-4 w-4" />
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}