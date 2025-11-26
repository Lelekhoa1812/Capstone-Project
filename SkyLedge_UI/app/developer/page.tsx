"use client";

import { useEffect, useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import UploadSection from "./_components/UploadSection";
import LabelingSection from "./_components/LabelingSection";
import DatasetSection from "./_components/DatasetSection";
import ReinforcementSection from "./_components/ReinforcementSection";
import RawLogsPage from "./raw-logs/page";
import DataProcessingPage from "./data-processing/page";
import TripsSection from "./_components/TripSection";
export default function DeveloperPage() {
  const [activeTab, setActiveTab] = useState("upload");

  useEffect(() => {
    const applyFromHash = () => {
      const h = window.location.hash.replace("#", "");
      const allowed = ["upload", "labeling", "dataset", "reinforcement", "raw-logs", "data-processing", "processed-trips"];      if (allowed.includes(h)) setActiveTab(h);
    };
    applyFromHash();
    window.addEventListener("hashchange", applyFromHash);
    return () => window.removeEventListener("hashchange", applyFromHash);
  }, []);

  // Tab → hash
  const onTabChange = (val: string) => {
    setActiveTab(val);
    history.replaceState(null, "", `#${val}`);
  };

  return (
    <div className="space-y-6">
      <div className="border-b border-border pb-4">
        <h2 className="text-2xl font-bold">Developer Dashboard</h2>
        <p className="text-muted-foreground mt-1">Manage OBD-II data processing, labeling, and model training</p>
      </div>

      <Tabs value={activeTab} onValueChange={onTabChange} className="space-y-4">
        <TabsList>
          <TabsTrigger value="processed-trips">Processed Trips</TabsTrigger>
          <TabsTrigger value="upload">Upload/Buffer</TabsTrigger>
          <TabsTrigger value="raw-logs">Raw Logs</TabsTrigger>
          <TabsTrigger value="data-processing">Data Processing</TabsTrigger>
          <TabsTrigger value="labeling">Manual Labeling</TabsTrigger>
          <TabsTrigger value="dataset">Labeled Dataset</TabsTrigger>
          <TabsTrigger value="reinforcement">Reinforcement</TabsTrigger>
        </TabsList>
      <TabsContent value="processed-trips"><TripsSection /></TabsContent>
        <TabsContent value="upload"><UploadSection /></TabsContent>
        <TabsContent value="raw-logs"><RawLogsPage /></TabsContent>
        <TabsContent value="data-processing"><DataProcessingPage /></TabsContent>
        <TabsContent value="labeling"><LabelingSection /></TabsContent>
        <TabsContent value="dataset"><DatasetSection /></TabsContent>
        <TabsContent value="reinforcement"><ReinforcementSection /></TabsContent>
      </Tabs>
    </div>
  );
}
