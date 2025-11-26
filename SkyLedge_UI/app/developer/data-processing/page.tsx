"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { 
  Table, 
  TableBody, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from "@/components/ui/table";
import { 
  Download, 
  Eye, 
  Search, 
  Clock, 
  Calendar, 
  FileText,
  RefreshCw,
  AlertCircle,
  Play,
  CheckCircle,
  XCircle
} from "lucide-react";
import { SensorPreviewDialog } from "../raw-logs/_components/SensorPreviewDialog";

interface ProcessedFile {
  name: string;
  size: number;
  timeCreated: string;
  duration?: number;
  sessionId: string;
  date: string;
}

interface ProcessingJob {
  id: string;
  rawFile: string;
  labeledFile: string;
  status: 'pending' | 'processing' | 'completed' | 'error';
  progress: number;
  error?: string;
  startTime: number;
}

export default function DataProcessingPage() {
  const [files, setFiles] = useState<ProcessedFile[]>([]);
  const [rawFiles, setRawFiles] = useState<string[]>([]);
  const [labeledFiles, setLabeledFiles] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const [previewFile, setPreviewFile] = useState<string | null>(null);
  const [previewOpen, setPreviewOpen] = useState(false);
  
  // Processing state
  const [selectedFile, setSelectedFile] = useState<string>("");
  const [selectedFileType, setSelectedFileType] = useState<'raw' | 'labeled'>('raw');
  const [processingJobs, setProcessingJobs] = useState<ProcessingJob[]>([]);
  const [nextFileId, setNextFileId] = useState<string>("");

  const fetchProcessedFiles = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/processed-files');
      const data = await response.json();
      
      if (data.success) {
        setFiles(data.files);
      } else {
        console.error('Failed to fetch processed files:', data.error);
      }
    } catch (error) {
      console.error('Error fetching processed files:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchRawFiles = async () => {
    try {
      const response = await fetch('/api/raw-files');
      const data = await response.json();
      
      if (data.success) {
        setRawFiles(data.files);
      }
    } catch (error) {
      console.error('Error fetching raw files:', error);
    }
  };

  const fetchLabeledFiles = async () => {
    try {
      const response = await fetch('/api/labeled-files');
      const data = await response.json();
      
      if (data.success) {
        setLabeledFiles(data.files);
      }
    } catch (error) {
      console.error('Error fetching labeled files:', error);
    }
  };

  const fetchNextFileId = async () => {
    try {
      const response = await fetch('/api/session-id');
      const data = await response.json();
      setNextFileId(data.sessionId || '001');
    } catch (error) {
      console.error('Error fetching next file ID:', error);
      setNextFileId('001');
    }
  };

  useEffect(() => {
    fetchProcessedFiles();
    fetchRawFiles();
    fetchLabeledFiles();
    fetchNextFileId();
  }, []);

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
    setPreviewFile(fileName);
    setPreviewOpen(true);
  };

  const handleProcessFiles = async () => {
    if (!selectedFile) {
      alert('Please select a file to process');
      return;
    }

    const jobId = `job_${Date.now()}`;
    const newJob: ProcessingJob = {
      id: jobId,
      rawFile: selectedFileType === 'raw' ? selectedFile : '',
      labeledFile: selectedFileType === 'labeled' ? selectedFile : '',
      status: 'pending',
      progress: 0,
      startTime: Date.now()
    };

    setProcessingJobs(prev => [...prev, newJob]);
    setSelectedFile("");

    try {
      console.log(`🚀 Starting processing job: ${jobId}`);
      console.log(`📁 ${selectedFileType} file: ${selectedFile}`);
      console.log(`🆔 Next file ID: ${nextFileId}`);

      // Update job status to processing
      setProcessingJobs(prev => prev.map(job => 
        job.id === jobId ? { ...job, status: 'processing', progress: 10 } : job
      ));

      // Create form data
      const formData = new FormData();
      
      // Fetch selected file
      const fileResponse = await fetch(`/api/download-file?fileName=${encodeURIComponent(selectedFile)}`);
      const fileBlob = await fileResponse.blob();
      formData.append('file', fileBlob, selectedFile.split('/').pop());

      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setProcessingJobs(prev => prev.map(job => {
          if (job.id === jobId && job.status === 'processing') {
            const newProgress = Math.min(90, job.progress + Math.random() * 10);
            return { ...job, progress: newProgress };
          }
          return job;
        }));
      }, 1000);

      // Send to processing backend
      const response = await fetch('https://binkhoale1812-obd-logger.hf.space/upload-csv/', {
        method: 'POST',
        body: formData,
      });

      clearInterval(progressInterval);

      if (response.ok) {
        console.log(`✅ Processing job completed: ${jobId}`);
        setProcessingJobs(prev => prev.map(job => 
          job.id === jobId ? { ...job, status: 'completed', progress: 100 } : job
        ));
        
        // Refresh processed files list
        setTimeout(() => {
          fetchProcessedFiles();
        }, 2000);
      } else {
        throw new Error(`Processing failed: ${response.statusText}`);
      }
    } catch (error) {
      console.error(`❌ Processing job failed: ${jobId}`, error);
      setProcessingJobs(prev => prev.map(job => 
        job.id === jobId ? { 
          ...job, 
          status: 'error', 
          error: String(error),
          progress: 0 
        } : job
      ));
    }
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

  const formatFileSize = (bytes: number) => {
    const sizes = ['B', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 B';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${sizes[i]}`;
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'error':
        return <XCircle className="h-4 w-4 text-red-500" />;
      case 'processing':
        return <RefreshCw className="h-4 w-4 text-blue-500 animate-spin" />;
      default:
        return <Clock className="h-4 w-4 text-yellow-500" />;
    }
  };

  const filteredFiles = files.filter(file => 
    file.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    file.sessionId.includes(searchTerm) ||
    file.date.includes(searchTerm)
  );

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Data Processing</h1>
          <p className="text-muted-foreground">
            Process raw and labeled OBD-II data files using machine learning backend
          </p>
        </div>
        <Button onClick={fetchProcessedFiles} disabled={loading} variant="outline">
          <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>

      {/* Processing Section */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Play className="h-5 w-5" />
            <span>Process Files</span>
            <Badge variant="outline" className="ml-2">
              Next ID: {nextFileId}
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">File Type</label>
              <Select value={selectedFileType} onValueChange={(value: 'raw' | 'labeled') => {
                setSelectedFileType(value);
                setSelectedFile(""); // Clear selected file when changing type
              }}>
                <SelectTrigger>
                  <SelectValue placeholder="Select file type..." />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="raw">Raw Data</SelectItem>
                  <SelectItem value="labeled">Labeled Data</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div className="space-y-2">
              <label className="text-sm font-medium">
                {selectedFileType === 'raw' ? 'Raw Data File' : 'Labeled Data File'}
              </label>
              <Select value={selectedFile} onValueChange={setSelectedFile}>
                <SelectTrigger>
                  <SelectValue placeholder={`Select ${selectedFileType} data file...`} />
                </SelectTrigger>
                <SelectContent>
                  {(selectedFileType === 'raw' ? rawFiles : labeledFiles).map((file) => (
                    <SelectItem key={file} value={file}>
                      {file.split('/').pop()}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
          
          <Button 
            onClick={handleProcessFiles}
            disabled={!selectedFile}
            className="w-full"
          >
            <Play className="h-4 w-4 mr-2" />
            Process File
          </Button>
        </CardContent>
      </Card>

      {/* Processing Jobs */}
      {processingJobs.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Processing Jobs</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {processingJobs.map((job) => (
                <div key={job.id} className="border rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      {getStatusIcon(job.status)}
                      <span className="font-medium">
                        {(job.rawFile || job.labeledFile).split('/').pop()}
                      </span>
                      <Badge variant="outline" className="text-xs">
                        {job.rawFile ? 'Raw' : 'Labeled'}
                      </Badge>
                    </div>
                    <Badge variant={job.status === 'completed' ? 'default' : job.status === 'error' ? 'destructive' : 'secondary'}>
                      {job.status}
                    </Badge>
                  </div>
                  
                  {job.status === 'processing' && (
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm text-muted-foreground">
                        <span>Processing...</span>
                        <span>{Math.round(job.progress)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${job.progress}%` }}
                        />
                      </div>
                    </div>
                  )}
                  
                  {job.status === 'error' && job.error && (
                    <div className="text-sm text-red-600 bg-red-50 p-2 rounded">
                      {job.error}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Processed Files List */}
      <Card>
        <CardHeader>
          <div className="flex items-center space-x-4">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
              <Input
                placeholder="Search processed files..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10"
              />
            </div>
            <Badge variant="secondary" className="px-3 py-1">
              {filteredFiles.length} files
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="flex items-center justify-center py-8">
              <RefreshCw className="h-6 w-6 animate-spin mr-2" />
              <span>Loading processed files...</span>
            </div>
          ) : filteredFiles.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
              <FileText className="h-12 w-12 mb-4" />
              <h3 className="text-lg font-semibold mb-2">No processed files found</h3>
              <p className="text-center">
                {searchTerm ? 'No files match your search criteria.' : 'No processed files available. Process some files to get started.'}
              </p>
            </div>
          ) : (
            <div className="rounded-md border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[200px]">Filename</TableHead>
                    <TableHead className="w-[100px]">Session ID</TableHead>
                    <TableHead className="w-[120px]">Duration</TableHead>
                    <TableHead className="w-[120px]">Size</TableHead>
                    <TableHead className="w-[140px]">Processed Date</TableHead>
                    <TableHead className="w-[100px] text-center">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredFiles.map((file) => (
                    <TableRow key={file.name}>
                      <TableCell className="font-medium">
                        <div className="flex items-center space-x-2">
                          <FileText className="h-4 w-4 text-muted-foreground" />
                          <span className="truncate max-w-[180px]" title={file.name}>
                            {file.name.split('/').pop()}
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
                          <Clock className="h-3 w-3 text-muted-foreground" />
                          <span className="text-sm">
                            {formatDuration(file.duration)}
                          </span>
                        </div>
                      </TableCell>
                      <TableCell className="text-sm text-muted-foreground">
                        {formatFileSize(file.size)}
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center space-x-1">
                          <Calendar className="h-3 w-3 text-muted-foreground" />
                          <span className="text-sm">
                            {new Date(file.timeCreated).toLocaleDateString()}
                          </span>
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center justify-center space-x-1">
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
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>

      <SensorPreviewDialog
        open={previewOpen}
        onOpenChange={setPreviewOpen}
        fileName={previewFile}
      />
    </div>
  );
}
