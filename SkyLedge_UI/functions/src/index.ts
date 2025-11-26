// functions/src/index.ts
import { setGlobalOptions } from "firebase-functions/v2";
import { onCall, HttpsError } from "firebase-functions/v2/https";
import { parse } from "csv-parse";
import { stringify } from "csv-stringify/sync";

// ✅ use shared admin exports
import { db, storage, FieldValue } from "./admin.js";

// Global runtime opts
setGlobalOptions({
  region: "us-central1",
  memory: "1GiB",
  timeoutSeconds: 300,
});

type DrivingLabel = "idle" | "passive" | "aggressive";
interface LabelRange { startTs: string | number; endTs: string | number; label: DrivingLabel; }

function toEpochSeconds(v: string | number): number {
  if (typeof v === "number") return v > 10_000_000_000 ? Math.floor(v / 1000) : Math.floor(v);
  if (/^\d+(\.\d+)?$/.test(v)) {
    const num = Number(v);
    return num > 10_000_000_000 ? Math.floor(num / 1000) : Math.floor(num);
  }
  const ms = Date.parse(v);
  if (isNaN(ms)) throw new HttpsError("invalid-argument", `Invalid timestamp: ${v}`);
  return Math.floor(ms / 1000);
}

// Re-export helpers so Firebase discovers them
export { listEligibleTrips } from "./listEligibleTrips.js";
export { getTripMeta } from "./getTripMeta.js";
export { getTripSeries } from "./getTripSeries.js";

export const labelTrip = onCall(async (request) => {
  if (!request.auth) throw new HttpsError("unauthenticated", "Sign in to label trips.");

  const data = request.data as { rawPath?: string; ranges?: LabelRange[]; timestampColumn?: string; };
  const rawPath = String(data?.rawPath ?? "");
  const ranges = (data?.ranges ?? []) as LabelRange[];
  const timestampColumn = data?.timestampColumn || "timestamp";

  if (!rawPath || !rawPath.endsWith("_raw.csv")) {
    throw new HttpsError("invalid-argument", "rawPath must end with _raw.csv and be a Storage path");
  }
  if (!Array.isArray(ranges) || ranges.length === 0) {
    throw new HttpsError("invalid-argument", "ranges is required");
  }

  const windows = ranges.map((r) => {
    const a = toEpochSeconds(r.startTs);
    const b = toEpochSeconds(r.endTs);
    const lo = Math.min(a, b);
    const hi = Math.max(a, b);
    const label = r.label as DrivingLabel;
    if (!["idle", "passive", "aggressive"].includes(label)) {
      throw new HttpsError("invalid-argument", `Invalid label: ${label}`);
    }
    return { lo, hi, label };
  });

  const labeledName = rawPath.replace("_raw.csv", "_labeled.csv");
  const labeledPath = rawPath.replace("skyledge/raw/", "skyledge/labeled/").replace("_raw.csv", "_labeled.csv");

  const bucket = storage.bucket();
  const file = bucket.file(rawPath);
  const [exists] = await file.exists();
  if (!exists) throw new HttpsError("not-found", `File not found: ${rawPath}`);

  const rows: any[] = [];
  const headersSet = new Set<string>();
  let headers: string[] = [];

  await new Promise<void>((resolve, reject) => {
    file.createReadStream()
      .on("error", reject)
      .pipe(parse({ columns: true, trim: true }))
      .on("data", (row: any) => {
        if (headers.length === 0) {
          headers = Object.keys(row);
          headers.forEach((h) => headersSet.add(h));
        }
        rows.push(row);
      })
      .on("end", resolve);
  });

  if (headers.length === 0) throw new HttpsError("failed-precondition", "CSV appears empty or has no header row.");
  if (!headersSet.has(timestampColumn)) throw new HttpsError("failed-precondition", `CSV missing '${timestampColumn}' column.`);

  if (!headersSet.has("driving_style")) {
    headers = [...headers, "driving_style"];
    headersSet.add("driving_style");
  }

  for (const row of rows) {
    const v = row[timestampColumn];
    let tsSec: number | null = null;

    if (v == null || v === "") tsSec = null;
    else if (typeof v === "number") tsSec = toEpochSeconds(v);
    else if (/^\d+(\.\d+)?$/.test(String(v))) tsSec = toEpochSeconds(Number(v));
    else {
      const ms = Date.parse(String(v));
      tsSec = isNaN(ms) ? null : Math.floor(ms / 1000);
    }

    if (tsSec == null) continue;

    let applied: DrivingLabel | null = null;
    for (const w of windows) if (tsSec >= w.lo && tsSec <= w.hi) applied = w.label;
    if (applied) row["driving_style"] = applied;
  }

  const csv = stringify(rows, { header: true, columns: headers });

  await bucket.file(labeledPath).save(csv, { contentType: "text/csv; charset=utf-8", resumable: false });

  const uid = request.auth.uid!;
  await db.collection("labelJobs").add({
    uid, rawPath, labeledPath, timestampColumn, windows, rowCount: rows.length,
    createdAt: FieldValue.serverTimestamp(),
  });

  return { labeledPath, labeledName };
});
