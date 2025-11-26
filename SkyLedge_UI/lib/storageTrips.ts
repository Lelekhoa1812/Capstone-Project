// storageTrips.ts
import { storage } from "@/lib/firebase";
import Papa from "papaparse";

// List all raw trips and filter out those that already have a labeled twin
export async function listEligibleTrips(): Promise<string[]> {
  if (!storage) {
    throw new Error('Firebase Storage not initialized. This function should only be called server-side.');
  }
  
  const bucket = storage.bucket();
  
  // List raw files
  const [rawFiles] = await bucket.getFiles({ prefix: 'skyledge/raw/' });
  const [labeledFiles] = await bucket.getFiles({ prefix: 'skyledge/labeled/' });

  const labeledSet = new Set(
    labeledFiles
      .map((file) => file.name)
      .filter((n) => n.endsWith("_labeled.csv"))
  );

  // a raw name "skyledge/raw/001_2025-09-05_raw.csv" → labeled "skyledge/labeled/001_2025-09-05_labeled.csv"
  const toLabeledName = (raw: string) => raw.replace("skyledge/raw/", "skyledge/labeled/").replace("_raw.csv", "_labeled.csv");

  return rawFiles
    .map((file) => file.name)
    .filter((n) => n.endsWith("_raw.csv"))
    .filter((raw) => !labeledSet.has(toLabeledName(raw)));
}

// Load a CSV file from Storage → { headers, rows }  (CORS-safe via SDK)
export async function loadCsvFromStorage(path: string): Promise<{ headers: string[]; rows: any[] }> {
  if (!storage) {
    throw new Error('Firebase Storage not initialized. This function should only be called server-side.');
  }
  
  const bucket = storage.bucket();
  const file = bucket.file(path);
  
  // Download the file as a buffer
  const [buffer] = await file.download();
  const text = buffer.toString('utf-8');

  const parsed = Papa.parse(text, { header: true, skipEmptyLines: true });
  if (parsed.errors?.length) {
    console.warn("[csv] parse warnings/errors:", parsed.errors);
  }
  const headers = (parsed.meta.fields || []) as string[];
  const rows = (parsed.data as any[]) || [];
  return { headers, rows };
}

// Ensure 'driving_style' column exists and return updated {headers,rows}
export function ensureDrivingStyle(headers: string[], rows: any[]) {
  if (!headers.includes("driving_style")) headers = [...headers, "driving_style"];
  // do not touch existing labels if present
  return { headers, rows };
}

// Apply labels by timestamp window (seconds since trip start)
export function applyLabelByRange(
  headers: string[],
  rows: any[],
  rangeStartSec: number,
  rangeEndSec: number,
  drivingStyle: "idle" | "passive" | "aggressive"
) {
  // detect timestamp column
  const tsCol = headers.find((h) => h.toLowerCase() === "timestamp");
  if (!tsCol) throw new Error("CSV must have a 'timestamp' column");

  // infer start epoch (in seconds) from first row; handle ISO or numeric
  const first = rows[0]?.[tsCol];
  const parseTs = (v: any): number => {
    if (v == null || v === "") return NaN;
    if (/^\d+(\.\d+)?$/.test(String(v))) {
      const num = Number(v);
      return num > 10_000_000_000 ? Math.floor(num / 1000) : Math.floor(num);
    }
    const ms = Date.parse(String(v));
    return isNaN(ms) ? NaN : Math.floor(ms / 1000);
  };

  const startEpoch = parseTs(first);
  if (!isFinite(startEpoch)) throw new Error("Unable to parse first timestamp");

  const lo = Math.min(rangeStartSec, rangeEndSec);
  const hi = Math.max(rangeStartSec, rangeEndSec);

  rows.forEach((r) => {
    const ts = parseTs(r[tsCol]);
    if (!isFinite(ts)) return;
    const tFromStart = ts - startEpoch; // seconds since trip start
    if (tFromStart >= lo && tFromStart <= hi) {
      r["driving_style"] = drivingStyle;
    }
  });

  return { headers, rows };
}

// Save updated CSV to labeled path
export async function saveLabeledCsv(rawFilename: string, headers: string[], rows: any[]) {
  if (!storage) {
    throw new Error('Firebase Storage not initialized. This function should only be called server-side.');
  }
  
  const labeledName = rawFilename.replace("_raw.csv", "_labeled.csv");
  const labeledPath = `skyledge/labeled/${labeledName}`;

  const csv = Papa.unparse(rows, { columns: headers });
  const bucket = storage.bucket();
  const file = bucket.file(labeledPath);
  
  await file.save(csv, {
    metadata: {
      contentType: "text/csv; charset=utf-8",
    },
  });

  return labeledPath;
}
