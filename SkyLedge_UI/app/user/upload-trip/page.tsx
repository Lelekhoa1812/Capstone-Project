"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Loader2, UploadCloud, CalendarIcon } from "lucide-react";

type SentItem = {
  id: string;
  filename: string;
  sentAt: number;
};

// DB-backed recent uploads via API

export default function UploadTripPage() {
  const [file, setFile] = useState<File | null>(null);
  const [isSending, setIsSending] = useState(false);
  const [sent, setSent] = useState<SentItem[]>([]);
  const [filterDate, setFilterDate] = useState<string>("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    // Load recent from DB
    fetch('/api/uploads?limit=200')
      .then(r => r.json())
      .then((items: Array<{ filename: string; uploadedAt: number }>) => {
        const mapped: SentItem[] = items.map((i, idx) => ({ id: `${i.uploadedAt}-${idx}-${i.filename}`, filename: i.filename, sentAt: i.uploadedAt }));
        setSent(mapped);
      })
      .catch(() => {});
  }, []);

  const onSelect = (f: File | null) => setFile(f);

  const filtered = useMemo(() => {
    if (!filterDate) return sent;
    return sent.filter((s) => {
      const d = new Date(s.sentAt);
      const yyyy = d.getFullYear();
      const mm = String(d.getMonth() + 1).padStart(2, "0");
      const dd = String(d.getDate()).padStart(2, "0");
      const key = `${yyyy}-${mm}-${dd}`;
      return key === filterDate;
    });
  }, [sent, filterDate]);

  const onSend = async () => {
    if (!file) return;
    setIsSending(true);
    try {
      // Generate consistent filename like developer: skyledge/raw/{sessionId}_{yyyy-mm-dd}_raw.csv
      let sessionId = '001';
      try {
        console.log("🔍 Getting next session ID from Firebase Storage...");
        const response = await fetch('/api/session-id');
        const data = await response.json();
        
        console.log("📊 Session ID API Response:", data);
        console.log(`📈 Current last session ID: ${data.lastSessionId || 0}`);
        console.log(`🔢 Existing session IDs: [${data.existingSessionIds?.join(', ') || 'none'}]`);
        console.log(`✅ Next session ID: ${data.sessionId}`);
        
        sessionId = data.sessionId || '001';
      } catch (error) {
        console.error('❌ Error getting session ID:', error);
        // Fallback to time-based seed
        const seed = Date.now() % 1000;
        sessionId = String(seed).padStart(3, '0');
        console.log(`⚠️ Using fallback session ID: ${sessionId}`);
      }
      
      const date = new Date().toISOString().split('T')[0];
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
        alert(`Upload failed: ${firebaseResult.error}`);
        setIsSending(false);
        return;
      }
      
      console.log("✅ Firebase Storage upload successful!");
      
      const form = new FormData();
      // We cannot rename the File object, but backend can use this filename hint
      form.append("file", file);
      form.append("filename", fileName);
      
      // Fire-and-forget to backend; we won't block UI on response
      console.log("🌐 Sending to external endpoint...");
      fetch("https://binkhoale1812-obd-logger.hf.space/upload-csv/", {
        method: "POST",
        body: form,
      }).then(() => {
        console.log("✅ External endpoint upload successful!");
      }).catch((error) => {
        console.error("❌ External endpoint upload failed:", error);
      }).finally(() => setIsSending(false));

      // Immediately record in Mongo via API and refresh list (store canonical filename)
      fetch('/api/uploads', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ filename: fileName, size: file.size, uploadedAt: Date.now(), status: 'pending' })
      }).then(() => {
        return fetch('/api/uploads?limit=200').then(r => r.json()).then((items: Array<{ filename: string; uploadedAt: number }>) => {
          const mapped: SentItem[] = items.map((i, idx) => ({ id: `${i.uploadedAt}-${idx}-${i.filename}`, filename: i.filename, sentAt: i.uploadedAt }));
          setSent(mapped);
        });
      }).catch(() => {});
      setFile(null);
      if (inputRef.current) inputRef.current.value = "";
    } finally {
      // If the HF request hasn't finished yet, the finally of that promise will clear isSending
    }
  };

  return (
    <div className="space-y-6">
      <div className="border-b border-border pb-4">
        <h2 className="text-2xl font-bold">Upload Trip</h2>
        <p className="text-muted-foreground mt-1">Send your driving CSV to processing backend</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Upload CSV</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 sm:grid-cols-3 items-end">
            <div className="sm:col-span-2 space-y-2">
              <Label htmlFor="tripFile">CSV file</Label>
              <Input
                ref={inputRef}
                id="tripFile"
                type="file"
                accept=".csv,text/csv"
                onChange={(e) => onSelect(e.target.files?.[0] ?? null)}
              />
            </div>
            <div>
              <Button className="w-full" onClick={onSend} disabled={!file || isSending}>
                {isSending ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Sending...
                  </>
                ) : (
                  <>
                    <UploadCloud className="mr-2 h-4 w-4" />
                    Send
                  </>
                )}
              </Button>
            </div>
          </div>
          {file ? <p className="text-xs text-muted-foreground">Ready: {file.name}</p> : null}
          <Separator />
          <div className="grid gap-4 sm:grid-cols-3 items-end">
            <div className="sm:col-span-1 space-y-2">
              <Label>Date filter</Label>
              <div className="flex items-center gap-2">
                <CalendarIcon className="h-4 w-4 text-muted-foreground" />
                <Input type="date" value={filterDate} onChange={(e) => setFilterDate(e.target.value)} />
                {filterDate ? (
                  <Button variant="ghost" onClick={() => setFilterDate("")}>Clear</Button>
                ) : null}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Sent History</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Filename</TableHead>
                <TableHead>Sent At</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filtered.map((s) => (
                <TableRow key={s.id}>
                  <TableCell>{s.filename}</TableCell>
                  <TableCell>
                    {new Date(s.sentAt).toLocaleString()}
                  </TableCell>
                </TableRow>
              ))}
              {filtered.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={2} className="text-center text-sm text-muted-foreground">
                    No items
                  </TableCell>
                </TableRow>
              ) : null}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}


