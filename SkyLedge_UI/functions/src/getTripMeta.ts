import { onCall, HttpsError } from "firebase-functions/v2/https";
import { storage } from "./admin.js";   // ✅ use shared admin
import { parse } from "csv-parse";

function toEpochMs(v: any): number | null {
  if (v == null || v === "") return null;
  if (typeof v === "number") return v > 10_000_000_000 ? v : v * 1000;
  if (/^\d+(\.\d+)?$/.test(String(v))) {
    const num = Number(v);
    return num > 10_000_000_000 ? num : num * 1000;
  }
  const ms = Date.parse(String(v));
  return isNaN(ms) ? null : ms;
}

export const getTripMeta = onCall(async (req) => {
  if (!req.auth) throw new HttpsError("unauthenticated", "Sign in.");
  const rawPath: string = req.data?.rawPath || "";
  const timestampColumn: string = req.data?.timestampColumn || "timestamp";
  if (!rawPath || !rawPath.endsWith("_raw.csv")) throw new HttpsError("invalid-argument", "rawPath must end with _raw.csv");

  const bucket = storage.bucket();
  const file = bucket.file(rawPath);
  const [exists] = await file.exists();
  if (!exists) throw new HttpsError("not-found", `File not found: ${rawPath}`);

  let startMs: number | null = null;
  let endMs: number | null = null;

  await new Promise<void>((resolve, reject) => {
    file.createReadStream()
      .on("error", reject)
      .pipe(parse({ columns: true, trim: true }))
      .on("data", (row: any) => {
        const ms = toEpochMs(row[timestampColumn]);
        if (ms == null) return;
        if (startMs == null) startMs = ms;
        endMs = ms;
      })
      .on("end", resolve);
  });

  if (startMs == null || endMs == null) throw new HttpsError("failed-precondition", "Could not infer start/end timestamps.");

  return {
    timestampColumn,
    startTsISO: new Date(startMs).toISOString(),
    endTsISO: new Date(endMs).toISOString(),
  };
});
