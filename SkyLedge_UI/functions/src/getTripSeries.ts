import { getStorage } from "firebase-admin/storage";
import { onCall, HttpsError } from "firebase-functions/v2/https";
import { parse } from "csv-parse";

const storage = getStorage();

type SeriesReq = {
  rawPath: string;
  timestampColumn?: string;     // default "timestamp"
  columns?: string[];           // optional: which numeric columns to include
  stride?: number;              // take every Nth row (default 10)
  maxPoints?: number;           // hard cap (default 5000)
};

export const getTripSeries = onCall(async (req) => {
  if (!req.auth) throw new HttpsError("unauthenticated", "Sign in.");

  const {
    rawPath,
    timestampColumn = "timestamp",
    columns,
    stride = 10,
    maxPoints = 5000,
  } = (req.data || {}) as SeriesReq;

  if (!rawPath || !rawPath.endsWith("_raw.csv")) {
    throw new HttpsError("invalid-argument", "rawPath must end with _raw.csv");
  }
  if (stride < 1) throw new HttpsError("invalid-argument", "stride must be >= 1");

  const bucket = storage.bucket();
  const file = bucket.file(rawPath);
  const [exists] = await file.exists();
  if (!exists) throw new HttpsError("not-found", `File not found: ${rawPath}`);

  let header: string[] = [];
  let includeCols: string[] = [];
  const rows: Array<{ t: number; [k: string]: number }> = [];
  let i = 0;

 const read = file.createReadStream();
const parser = parse({ columns: true, trim: true });

await new Promise<void>((resolve, reject) => {
  read
    .on("error", reject)
    .pipe(parser)
    .on("headers", (h: string[]) => {
      header = h;
      const wanted = new Set(columns && columns.length ? columns : header);
      includeCols = header.filter(
        (c) => wanted.has(c) && c !== timestampColumn && c !== "driving_style"
      );
    })
    .on("data", (row: any) => {
      i++;
      if (i % stride !== 0) return;

      const tsMs = Date.parse(String(row[timestampColumn]));
      if (Number.isNaN(tsMs)) return;

      const out: any = { t: Math.floor(tsMs / 1000) };
      let hasAny = false;
      for (const c of includeCols) {
        const vRaw = row[c];
        const vNum =
          typeof vRaw === "number"
            ? vRaw
            : /^\s*-?\d+(\.\d+)?\s*$/.test(String(vRaw))
            ? Number(vRaw)
            : NaN;
        if (Number.isFinite(vNum)) {
          out[c] = vNum;
          hasAny = true;
        }
      }
      if (hasAny) rows.push(out);

      // ✅ stop parsing once we hit maxPoints
      if (rows.length >= maxPoints) {
        parser.pause();           // stop receiving more 'data'
        read.unpipe(parser);      // detach the stream
        parser.removeAllListeners();
        read.destroy();           // close the file stream
        resolve();                // resolve the promise early
      }
    })
    .on("end", resolve)
    .on("error", reject);
});


  if (rows.length === 0) {
    return { columns: includeCols, points: [] as any[] };
  }

  // Build per-column min/max for UI autoscale
  const stats: Record<string, { min: number; max: number }> = {};
  for (const c of includeCols) {
    let mn = Infinity, mx = -Infinity;
    for (const r of rows) {
      const v = r[c];
      if (typeof v === "number" && Number.isFinite(v)) {
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
    }
    if (mn !== Infinity && mx !== -Infinity) stats[c] = { min: mn, max: mx };
  }

  // convert to {tsISO, ...}
  const points = rows.map(r => {
    const { t, ...rest } = r;
    return { tsISO: new Date(t * 1000).toISOString(), ...rest };
  });

  return {
    columns: includeCols,
    points,
    stats,
  };
});
