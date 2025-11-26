"use client";

import { useEffect, useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { 
  Brain, 
  Play, 
  RefreshCw, 
  CheckCircle, 
  Clock, 
  AlertTriangle, 
  ExternalLink,
  Database,
  TrendingUp,
  Activity
} from "lucide-react";


interface TrainingStatus {
  status: string;
  labeled_datasets_count: number;
  datasets: Array<{
    name: string;
    size: number;
    created: string;
  }>;
  firebase_bucket: string;
  labeled_path: string;
  timestamp: string;
}

interface TrainedDatasets {
  trained_datasets_count: number;
  trained_datasets: string[];
  timestamp: string;
}

interface PendingDatasets {
  pending_datasets_count: number;
  pending_datasets: Array<{
    name: string;
    size: number;
    created: string;
  }>;
  total_available: number;
  already_trained: number;
  timestamp: string;
}

interface LatestModel {
  status: string;
  latest_version: string;
  model_repository: string;
  version_format: string;
  timestamp: string;
}

export default function ReinforcementSection() {
  // State management
  const [maxDatasets, setMaxDatasets] = useState(5);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);
  const [trainedDatasets, setTrainedDatasets] = useState<TrainedDatasets | null>(null);
  const [pendingDatasets, setPendingDatasets] = useState<PendingDatasets | null>(null);
  const [latestModel, setLatestModel] = useState<LatestModel | null>(null);
  const [showTrainingStatus, setShowTrainingStatus] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  // Fetch all data on component mount
  useEffect(() => {
    fetchAllData();
  }, []);

  const fetchAllData = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const [statusRes, trainedRes, pendingRes, modelRes] = await Promise.all([
        fetch("/api/rlhf/status"),
        fetch("/api/rlhf/trained-datasets"),
        fetch("/api/rlhf/pending-datasets"),
        fetch("/api/rlhf/latest-model")
      ]);

      const [statusData, trainedData, pendingData, modelData] = await Promise.all([
        statusRes.json(),
        trainedRes.json(),
        pendingRes.json(),
        modelRes.json()
      ]);

      if (statusData.ok) setTrainingStatus(statusData);
      if (trainedData.ok) setTrainedDatasets(trainedData);
      if (pendingData.ok) setPendingDatasets(pendingData);
      if (modelData.ok) setLatestModel(modelData);

    } catch (err) {
      setError("Failed to fetch data. Please try again.");
      console.error("Error fetching data:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const triggerTraining = async () => {
    if (!pendingDatasets || pendingDatasets.pending_datasets_count === 0) {
      setError("No pending datasets available for training. Please upload some labeled data first.");
      return;
    }

    setIsTraining(true);
    setError(null);
    setShowTrainingStatus(true);

    try {
      const response = await fetch("/api/rlhf/train", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          max_datasets: maxDatasets,
          force_retrain: false
        })
      });

      const data = await response.json();

      if (!data.ok) {
        throw new Error(data.error || "Training failed");
      }

      // Start polling for status updates
      startStatusPolling();

    } catch (err) {
      setError(err instanceof Error ? err.message : "Training failed");
      setIsTraining(false);
    }
  };

  const startStatusPolling = () => {
    const interval = setInterval(async () => {
      try {
        const response = await fetch("/api/rlhf/status");
        const data = await response.json();
        
        if (data.ok) {
          setTrainingStatus(data);
          
          // Check if training is complete
          if (data.status === "completed" || data.status === "available") {
            setIsTraining(false);
            clearInterval(interval);
            fetchAllData(); // Refresh all data
          }
        }
      } catch (err) {
        console.error("Status polling error:", err);
      }
    }, 2000); // Poll every 2 seconds

    // Stop polling after 5 minutes
    setTimeout(() => {
      clearInterval(interval);
      setIsTraining(false);
    }, 300000);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  const generateModelVersions = (latestVersion: string) => {
    const versions = [];
    const [major, minor] = latestVersion.replace('v', '').split('.').map(Number);
    
    // Generate versions from v1.0 to current version
    for (let m = 1; m <= major; m++) {
      const maxMinor = m === major ? minor : 9;
      for (let mi = 0; mi <= maxMinor; mi++) {
        versions.push(`v${m}.${mi}`);
      }
    }
    
    return versions.reverse(); // Show latest first
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case "available":
      case "completed":
        return "bg-green-500";
      case "training":
      case "in_progress":
        return "bg-blue-500";
      case "error":
      case "failed":
        return "bg-red-500";
      default:
        return "bg-gray-500";
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status.toLowerCase()) {
      case "available":
      case "completed":
        return <CheckCircle className="h-4 w-4" />;
      case "training":
      case "in_progress":
        return <Activity className="h-4 w-4" />;
      case "error":
      case "failed":
        return <AlertTriangle className="h-4 w-4" />;
      default:
        return <Clock className="h-4 w-4" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Reinforcement Learning from Human Feedback (RLHF)
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Train and manage your driver behavior model using labeled datasets
          </p>
        </CardHeader>
      </Card>

      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {/* Trained Datasets */}
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Trained Datasets</p>
                <p className="text-2xl font-bold">
                  {trainedDatasets?.trained_datasets_count || 0}
                </p>
              </div>
              <Database className="h-8 w-8 text-green-500" />
            </div>
          </CardContent>
        </Card>

        {/* Pending Datasets */}
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Pending Datasets</p>
                <p className="text-2xl font-bold">
                  {pendingDatasets?.pending_datasets_count || 0}
                </p>
              </div>
              <Clock className="h-8 w-8 text-amber-500" />
            </div>
          </CardContent>
        </Card>

        {/* Total Available */}
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Total Available</p>
                <p className="text-2xl font-bold">
                  {pendingDatasets?.total_available || 0}
                </p>
              </div>
              <TrendingUp className="h-8 w-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>

        {/* Latest Model Version */}
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Latest Model</p>
                <p className="text-lg font-bold">
                  {latestModel?.latest_version || "N/A"}
                </p>
              </div>
              <Brain className="h-8 w-8 text-purple-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Training Control */}
      <Card>
        <CardHeader>
          <CardTitle>Training Control</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Label htmlFor="maxDatasets">Max Datasets:</Label>
              <Input
                id="maxDatasets"
                type="number"
                min="1"
                max="100"
                value={maxDatasets}
                onChange={(e) => setMaxDatasets(parseInt(e.target.value) || 1)}
                className="w-20"
              />
            </div>
            
            <Button
              onClick={triggerTraining}
              disabled={isTraining || !pendingDatasets || pendingDatasets.pending_datasets_count === 0}
              className="flex items-center gap-2"
            >
              {isTraining ? (
                <RefreshCw className="h-4 w-4 animate-spin" />
              ) : (
                <Play className="h-4 w-4" />
              )}
              {isTraining ? "Training..." : "Start Training"}
            </Button>

            <Button
              variant="outline"
              onClick={fetchAllData}
              disabled={isLoading}
              className="flex items-center gap-2"
            >
              <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </div>

          {error && (
            <div className="flex items-center gap-2 p-3 bg-red-50 border border-red-200 rounded-md">
              <AlertTriangle className="h-4 w-4 text-red-500" />
              <span className="text-sm text-red-700">{error}</span>
            </div>
          )}

          {pendingDatasets && pendingDatasets.pending_datasets_count === 0 && (
            <div className="flex items-center gap-2 p-3 bg-amber-50 border border-amber-200 rounded-md">
              <AlertTriangle className="h-4 w-4 text-amber-500" />
              <span className="text-sm text-amber-700">
                No pending datasets available for training. Please upload some labeled data first.
              </span>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Training Status (Hidden by default, shown when training) */}
      {showTrainingStatus && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Training Status
            </CardTitle>
          </CardHeader>
          <CardContent>
            {trainingStatus ? (
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <div className={`w-3 h-3 rounded-full ${getStatusColor(trainingStatus.status)}`} />
                  <span className="font-medium capitalize">{trainingStatus.status}</span>
                  {getStatusIcon(trainingStatus.status)}
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-muted-foreground">Labeled Datasets Count</p>
                    <p className="text-lg font-semibold">{trainingStatus.labeled_datasets_count}</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Firebase Bucket</p>
                    <p className="text-sm font-mono">{trainingStatus.firebase_bucket}</p>
                  </div>
                </div>

                {trainingStatus.datasets && trainingStatus.datasets.length > 0 && (
                  <div>
                    <p className="text-sm font-medium mb-2">Available Datasets</p>
                    <div className="space-y-2 max-h-40 overflow-y-auto">
                      {trainingStatus.datasets.map((dataset, index) => (
                        <div key={index} className="flex items-center justify-between p-2 bg-muted rounded">
                          <span className="text-sm font-mono">{dataset.name}</span>
                          <div className="flex items-center gap-2">
                            <Badge variant="secondary">{formatFileSize(dataset.size)}</Badge>
                            <span className="text-xs text-muted-foreground">
                              {formatDate(dataset.created)}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex items-center justify-center p-8">
                <RefreshCw className="h-6 w-6 animate-spin" />
                <span className="ml-2">Loading training status...</span>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Trained Datasets */}
      <Card>
        <CardHeader>
          <CardTitle>Trained Datasets</CardTitle>
          <p className="text-sm text-muted-foreground">
            Datasets that have been successfully trained
          </p>
        </CardHeader>
        <CardContent>
          {trainedDatasets && trainedDatasets.trained_datasets.length > 0 ? (
            <div className="space-y-2">
              {trainedDatasets.trained_datasets.map((dataset, index) => (
                <div key={index} className="flex items-center justify-between p-3 bg-green-50 border border-green-200 rounded-md">
                  <div className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    <span className="font-mono text-sm">{dataset}</span>
                  </div>
                  <Badge variant="outline" className="bg-green-100 text-green-700">
                    Trained
                  </Badge>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-muted-foreground">
              No trained datasets available
            </div>
          )}
        </CardContent>
      </Card>

      {/* Pending Datasets */}
      <Card>
        <CardHeader>
          <CardTitle>Pending Datasets</CardTitle>
          <p className="text-sm text-muted-foreground">
            Datasets waiting to be trained
          </p>
        </CardHeader>
        <CardContent>
          {pendingDatasets && pendingDatasets.pending_datasets.length > 0 ? (
            <div className="space-y-2">
              {pendingDatasets.pending_datasets.map((dataset, index) => (
                <div key={index} className="flex items-center justify-between p-3 bg-amber-50 border border-amber-200 rounded-md">
                  <div className="flex items-center gap-2">
                    <Clock className="h-4 w-4 text-amber-500" />
                    <span className="font-mono text-sm">{dataset.name}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="secondary">{formatFileSize(dataset.size)}</Badge>
                    <Badge variant="outline" className="bg-amber-100 text-amber-700">
                      Pending
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-muted-foreground">
              No pending datasets available
            </div>
          )}
        </CardContent>
      </Card>

      {/* Model Versions */}
      {latestModel && (
        <Card>
          <CardHeader>
            <CardTitle>Model Versions</CardTitle>
            <p className="text-sm text-muted-foreground">
              Available model versions from v1.0 to {latestModel.latest_version}
            </p>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-2">
              {generateModelVersions(latestModel.latest_version).map((version) => (
                <a
                  key={version}
                  href={`https://huggingface.co/BinKhoaLe1812/Driver_Behavior_OBD/tree/main/${version}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center justify-center gap-1 p-2 border rounded-md hover:bg-muted transition-colors"
                >
                  <span className="text-sm font-mono">{version}</span>
                  <ExternalLink className="h-3 w-3" />
                </a>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}