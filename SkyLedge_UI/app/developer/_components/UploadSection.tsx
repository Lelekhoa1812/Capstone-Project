"use client";

import { useEffect, useState, useRef } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Upload, FileText, Eye, Download, Calendar, Clock, X, CheckCircle, AlertCircle, Loader2 } from "lucide-react";
// Firebase removed: use direct POST endpoint instead

interface UploadingFile {
  file: File;
  progress: number;
  status: 'uploading' | 'completed' | 'error';
  error?: string;
  downloadURL?: string;
}

function getStatusBadge(status: string) {
  switch (status) {
    case "processed":
      return <Badge variant="default">Processed</Badge>;
    case "pending":
      return <Badge variant="secondary">Pending</Badge>;
    case "error":
      return <Badge variant="destructive">Error</Badge>;
    default:
      return <Badge variant="outline">Unknown</Badge>;
  }
}

export default function UploadSection() {
  const [selectedFile, setSelectedFile] = useState<null | { filename: string }>(null);
  const [uploadingFiles, setUploadingFiles] = useState<UploadingFile[]>([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const getNextSessionId = async (): Promise<string> => {
    try {
      console.log("🔍 Getting next session ID from Firebase Storage...");
      const response = await fetch('/api/session-id');
      const data = await response.json();
      
      console.log("📊 Session ID API Response:", data);
      console.log(`📈 Current last session ID: ${data.lastSessionId || 0}`);
      console.log(`🔢 Existing session IDs: [${data.existingSessionIds?.join(', ') || 'none'}]`);
      console.log(`✅ Next session ID: ${data.sessionId}`);
      
      return data.sessionId || '001';
    } catch (error) {
      console.error('❌ Error getting session ID:', error);
      // Fallback to time-based seed
      const seed = Date.now() % 1000;
      console.log(`⚠️ Using fallback session ID: ${String(seed).padStart(3, '0')}`);
      return String(seed).padStart(3, '0');
    }
  };

  const handleFileSelect = (files: FileList | null) => {
    if (!files) return;

    const csvFiles = Array.from(files).filter(file => 
      file.type === 'text/csv' || 
      file.name.toLowerCase().endsWith('.csv') ||
      file.type === 'application/json' ||
      file.name.toLowerCase().endsWith('.json')
    );

    if (csvFiles.length === 0) {
      alert('Please select only CSV or JSON files.');
      return;
    }

    // Process files sequentially to ensure proper session ID incrementing
    csvFiles.forEach((file, index) => {
      // Add a small delay between uploads to prevent race conditions
      setTimeout(() => uploadFile(file), index * 200);
    });
  };

  const uploadFile = async (file: File) => {
    console.log("🚀 Starting upload for file:", file.name);
    
    const uploadingFile: UploadingFile = {
      file,
      progress: 0,
      status: 'uploading'
    };

    setUploadingFiles(prev => [...prev, uploadingFile]);

    try {
      // Get the next session ID by checking existing files
      const sessionId = await getNextSessionId();
      const date = new Date().toISOString().split('T')[0]; // YYYY-MM-DD format
      
      // Create filename: sessionid_date_raw.csv (NO VEHICLE ID)
      const fileName = `skyledge/raw/${sessionId}_${date}_raw.csv`;
      console.log("📁 Uploading file as:", fileName);
      
      // Upload to Firebase Storage via server-side API
      console.log("☁️ Uploading to Firebase Storage via API...");
      const formData = new FormData();
      formData.append('file', file);
      formData.append('fileName', fileName);
      
      const firebaseResponse = await fetch('/api/upload-firebase', {
        method: 'POST',
        body: formData,
      });
      
      const firebaseResult = await firebaseResponse.json();
      
      if (!firebaseResult.success) {
        console.error("❌ Firebase Storage upload failed:", firebaseResult.error);
        setUploadingFiles(prev => prev.map(f => f.file === file ? { ...f, status: 'error', error: `Firebase upload failed: ${firebaseResult.error}` } : f));
        return;
      }
      
      console.log("✅ Firebase Storage upload successful!");
      
      // Prepare form data for external endpoint
      const form = new FormData();
      form.append('file', file);
      form.append('filename', fileName);

      // Fire-and-forget send to HF; optimistic UI with loader animation
      const controller = new AbortController();
      const progressInterval = setInterval(() => {
        setUploadingFiles(prev => prev.map(f => f.file === file ? { ...f, progress: Math.min(95, (f.progress || 0) + 6) } : f));
      }, 300);

      console.log("🌐 Sending to external endpoint...");
      fetch('https://binkhoale1812-obd-logger.hf.space/upload-csv/', { method: 'POST', body: form, signal: controller.signal })
        .then(() => {
          console.log("✅ External endpoint upload successful!");
          clearInterval(progressInterval);
          setUploadingFiles(prev => prev.map(f => f.file === file ? { ...f, status: 'completed', progress: 100 } : f));
        })
        .catch((error: any) => {
          console.error("❌ External endpoint upload failed:", error);
          clearInterval(progressInterval);
          setUploadingFiles(prev => prev.map(f => f.file === file ? { ...f, status: 'error', error: String(error?.message || error) } : f));
        });

      // Immediately record in Mongo (server API) without waiting for HF response, computing duration from CSV
      try {
        const text = await file.text();
        const lines = text.split(/\r?\n/).filter(Boolean);
        let durationSec: number | undefined = undefined;
        if (lines.length > 1) {
          const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
          const tsIdx = headers.findIndex(h => h === 'timestamp' || h === 'time' || h === 'ts');
          if (tsIdx >= 0) {
            const firstRow = lines[1].split(',');
            const lastRow = lines[lines.length - 1].split(',');
            const parseTs = (v: string) => {
              const num = Number(v);
              if (!Number.isNaN(num)) return num >= 1e11 ? num : num * 1000;
              const d = Date.parse(v);
              return isNaN(d) ? undefined : d;
            };
            const t0 = parseTs(firstRow[tsIdx] || '');
            const t1 = parseTs(lastRow[tsIdx] || '');
            if (typeof t0 === 'number' && typeof t1 === 'number' && t1 >= t0) {
              durationSec = Math.round((t1 - t0) / 1000);
            }
          }
        }
        fetch('/api/uploads', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ filename: fileName, size: file.size, durationSec, uploadedAt: Date.now(), status: 'pending' })
        }).then(() => {
          refreshRecentUploads();
        }).catch(() => {});
      } catch {}
    } catch (error) {
      console.error('Error in uploadFile:', error);
      setUploadingFiles(prev =>
        prev.map(f =>
          f.file === file
            ? { ...f, status: 'error', error: 'Failed to initialize upload' }
            : f
        )
      );
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    handleFileSelect(e.dataTransfer.files);
  };

  const removeUploadingFile = (file: File) => {
    setUploadingFiles(prev => prev.filter(f => f.file !== file));
  };

  const openFileDialog = () => {
    fileInputRef.current?.click();
  };

  // Load recent uploads from Mongo API
  const [recent, setRecent] = useState<Array<{ filename: string; uploadedAt: number; size?: number; durationSec?: number; status?: string }>>([]);
  const [recentLoading, setRecentLoading] = useState(false);
  const refreshRecentUploads = () => {
    setRecentLoading(true);
    fetch('/api/uploads?limit=20')
      .then(r => r.json())
      .then((rows: any[]) => {
        // auto-purge invalid names client-side from view
        const pattern = /^skyledge\/raw\/\d{3}_\d{4}-\d{2}-\d{2}_raw\.csv$/;
        setRecent(rows.filter(r => pattern.test(r.filename || '')));
      })
      .finally(() => setRecentLoading(false));
  };

  // Initialize recent list on mount
  useEffect(() => {
    refreshRecentUploads();
  }, []);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Upload className="h-5 w-5" />
            File Upload & Buffer
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div 
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer ${
              isDragOver 
                ? 'border-primary bg-primary/5' 
                : 'border-border hover:border-primary/50'
            }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <Upload className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-lg font-medium mb-2">Upload OBD-II Data Files</h3>
            <p className="text-muted-foreground mb-4">
              Drag & drop CSV file, or click to browse
            </p>
            <Button onClick={(e) => { e.stopPropagation(); openFileDialog(); }}>
              <Upload className="h-4 w-4 mr-2" />
              Choose Files
            </Button>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".csv,.json"
              onChange={(e) => {
                console.log("File input onChange triggered");
                handleFileSelect(e.target.files);
              }}
              style={{ display: 'none' }}
            />
          </div>

          {/* Upload Progress Section */}
          {uploadingFiles.length > 0 && (
            <div className="mt-6 space-y-3">
              <h4 className="font-medium text-sm">Uploading Files</h4>
              {uploadingFiles.map((uploadingFile, index) => (
                <div key={index} className="border rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <FileText className="h-4 w-4" />
                      <span className="text-sm font-medium">
                        {uploadingFile.file.name}
                      </span>
                      {uploadingFile.status === 'uploading' && (
                        <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
                      )}
                      {uploadingFile.status === 'completed' && (
                        <CheckCircle className="h-4 w-4 text-green-500" />
                      )}
                      {uploadingFile.status === 'error' && (
                        <AlertCircle className="h-4 w-4 text-red-500" />
                      )}
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => removeUploadingFile(uploadingFile.file)}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                  
                  {uploadingFile.status === 'uploading' && (
                    <div className="space-y-2">
                      <Progress value={uploadingFile.progress} className="h-2" />
                      <p className="text-xs text-muted-foreground">
                        {Math.round(uploadingFile.progress)}% uploaded
                      </p>
                    </div>
                  )}
                  
                  {uploadingFile.status === 'completed' && (
                    <div className="space-y-2">
                      <Progress value={100} className="h-2" />
                      <p className="text-xs text-green-600">
                        Upload completed successfully!
                      </p>
                      {uploadingFile.downloadURL && (
                        <p className="text-xs text-muted-foreground break-all">
                          File available at: {uploadingFile.downloadURL}
                        </p>
                      )}
                    </div>
                  )}
                  
                  {uploadingFile.status === 'error' && (
                    <p className="text-xs text-red-600">
                      Error: {uploadingFile.error}
                    </p>
                  )}
                  
                  <p className="text-xs text-muted-foreground mt-1">
                    Size: {(uploadingFile.file.size / 1024).toFixed(1)} KB
                  </p>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileText className="h-5 w-5" />
            Recently Uploaded Files
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Filename</TableHead>
                <TableHead>Upload Time</TableHead>
                <TableHead>Size</TableHead>
                <TableHead>Duration</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {recent.map((file, idx) => (
                <TableRow key={idx}>
                  <TableCell className="font-medium">{file.filename}</TableCell>
                  <TableCell>
                    <div className="flex items-center gap-1 text-sm text-muted-foreground">
                      <Calendar className="h-3 w-3" />
                      {new Date(file.uploadedAt).toLocaleDateString()}
                      <Clock className="h-3 w-3 ml-2" />
                      {new Date(file.uploadedAt).toLocaleTimeString()}
                    </div>
                  </TableCell>
                  <TableCell>{typeof file.size === 'number' ? `${(file.size / 1024).toFixed(1)} KB` : '-'}</TableCell>
                  <TableCell>{typeof file.durationSec === 'number' ? `${file.durationSec}s` : '-'}</TableCell>
                  <TableCell>{getStatusBadge(file.status || 'pending')}</TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <Dialog>
                        <DialogTrigger asChild>
                          <Button variant="ghost" size="sm">
                            <Eye className="h-4 w-4" />
                          </Button>
                        </DialogTrigger>
                        <DialogContent className="max-w-2xl">
                          <DialogHeader>
                            <DialogTitle>File: {file.filename}</DialogTitle>
                          </DialogHeader>
                          <div className="space-y-2 text-sm text-muted-foreground">
                            <div>Uploaded: {new Date(file.uploadedAt).toLocaleString()}</div>
                            <div>Size: {typeof file.size === 'number' ? `${(file.size / 1024).toFixed(1)} KB` : '-'}</div>
                            <div>Duration: {typeof file.durationSec === 'number' ? `${file.durationSec}s` : '-'}</div>
                          </div>
                        </DialogContent>
                      </Dialog>
                      <Button variant="ghost" size="sm">
                        <Download className="h-4 w-4" />
                      </Button>
                      {(file.status || 'pending') === 'pending' && (
                        <Button variant="outline" size="sm">Process</Button>
                      )}
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}