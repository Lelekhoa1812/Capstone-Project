import { NextRequest } from "next/server";
import { storage } from "@/lib/firebase";
import Papa from "papaparse";

type DrivingStyle = "idle" | "passive" | "moderate" | "aggressive";

function toEpochSeconds(v: any): number | null {
  if (v == null || v === "") return null;
  if (typeof v === "number") return v > 10_000_000_000 ? Math.floor(v / 1000) : Math.floor(v);
  if (/^\d+(\.\d+)?$/.test(String(v))) {
    const num = Number(v);
    return num > 10_000_000_000 ? Math.floor(num / 1000) : Math.floor(num);
  }
  const ms = Date.parse(String(v));
  return isNaN(ms) ? null : Math.floor(ms / 1000);
}

export async function POST(req: NextRequest) {
  if (!storage) {
    return new Response(JSON.stringify({ ok: false, error: "Storage not initialized" }), {
      status: 500,
      headers: { "content-type": "application/json" },
    });
  }

  try {
    const body = await req.json();
    const sourcePath = String(body?.rawPath || body?.sourcePath || "");
    const timestampColumn = String(body?.timestampColumn || "timestamp");
    const sourceType = String(body?.sourceType || "processed"); // raw or processed
    const ranges = (body?.ranges || []) as Array<{
      startTs: string;
      endTs: string;
      label: DrivingStyle;
    }>;

    if (!sourcePath) {
      return new Response(JSON.stringify({ ok: false, error: "rawPath/sourcePath is required" }), {
        status: 400,
        headers: { "content-type": "application/json" },
      });
    }
    if (!Array.isArray(ranges) || ranges.length === 0) {
      return new Response(JSON.stringify({ ok: false, error: "ranges is required" }), {
        status: 400,
        headers: { "content-type": "application/json" },
      });
    }

    const bucket = storage.bucket();
    const file = bucket.file(sourcePath);
    const [exists] = await file.exists();
    if (!exists) {
      return new Response(JSON.stringify({ ok: false, error: `File not found: ${sourcePath}` }), {
        status: 404,
        headers: { "content-type": "application/json" },
      });
    }

    const [buffer] = await file.download();
    const text = buffer.toString("utf-8");
    const parsed = Papa.parse(text, { header: true, skipEmptyLines: true });
    if (parsed.errors?.length) {
      console.warn("[label-trip] CSV parse warnings:", parsed.errors);
    }
    let headers = (parsed.meta.fields || []) as string[];
    const rows = (parsed.data as any[]) || [];

    if (!headers.includes("driving_style")) headers = [...headers, "driving_style"]; 

    const firstTs = rows[0]?.[timestampColumn];
    const startEpoch = toEpochSeconds(firstTs);
    if (!Number.isFinite(startEpoch)) {
      return new Response(JSON.stringify({ ok: false, error: `Unable to parse first ${timestampColumn}` }), {
        status: 400,
        headers: { "content-type": "application/json" },
      });
    }

    const labelRanges = ranges.map((r) => ({
      lo: Math.min(Date.parse(r.startTs), Date.parse(r.endTs)) / 1000,
      hi: Math.max(Date.parse(r.startTs), Date.parse(r.endTs)) / 1000,
      label: r.label as DrivingStyle,
    }));

    rows.forEach((row) => {
      const tsSec = toEpochSeconds(row[timestampColumn]);
      if (tsSec == null || !Number.isFinite(tsSec)) return;
      const t = tsSec as number; // absolute seconds
      const hit = labelRanges.find((rg) => t >= rg.lo && t <= rg.hi);
      if (hit) row["driving_style"] = hit.label;
    });

    const csvOut = Papa.unparse(rows, { columns: headers });

    // Determine labeled output path
    // Extract source trip parts from sourcePath
    let base = sourcePath;
    if (base.startsWith("skyledge/raw/")) base = base.replace("skyledge/raw/", "");
    if (base.startsWith("skyledge/processed/")) base = base.replace("skyledge/processed/", "");
    
    // Parse the source filename to extract ID and date
    const srcMatch = base.match(/^(\d{3})_(\d{4}-\d{2}-\d{2})_(raw|processed)\.csv$/);
    if (!srcMatch) {
      return new Response(JSON.stringify({ ok: false, error: `Unexpected source filename format: ${base}. Expected format: 001_YYYY-MM-DD_raw.csv or 001_YYYY-MM-DD_processed.csv` }), {
        status: 400,
        headers: { "content-type": "application/json" },
      });
    }
    const [, srcId, srcDate, detectedType] = srcMatch;
    
    // Use the provided sourceType or fall back to detected type
    const finalSourceType = sourceType === "raw" || sourceType === "processed" ? sourceType : detectedType;

    // Scan existing labeled to compute next leading 3-digit id
    const [labeledFiles] = await bucket.getFiles({ prefix: "skyledge/labeled/" });
    let maxLeading = 0;
    const leadingRe = /^skyledge\/labeled\/(\d{3})_(raw|processed)-\d{3}_\d{4}-\d{2}-\d{2}-labelled(?:_v\d+)?\.csv$/;
    for (const f of labeledFiles) {
      const m = f.name.match(leadingRe);
      if (m) {
        const n = parseInt(m[1], 10);
        if (!Number.isNaN(n)) maxLeading = Math.max(maxLeading, n);
      }
    }
    const nextLeading = String(Math.min(maxLeading + 1, 999)).padStart(3, "0");

    // Compose base labelled name (without version) - Format: 001_processed-002_2025-09-19-labelled.csv
    let labeledName = `${nextLeading}_${finalSourceType}-${srcId}_${srcDate}-labelled.csv`;
    let labeledPath = `skyledge/labeled/${labeledName}`;

    // Versioning if duplicate somehow exists
    let outFile = bucket.file(labeledPath);
    let version = 2;
    while (true) {
      const [exists] = await outFile.exists();
      if (!exists) break;
      labeledName = `${nextLeading}_${finalSourceType}-${srcId}_${srcDate}-labelled_v${version}.csv`;
      labeledPath = `skyledge/labeled/${labeledName}`;
      outFile = bucket.file(labeledPath);
      version += 1;
      if (version > 99) {
        return new Response(JSON.stringify({ ok: false, error: "Too many labelled versions exist for this session" }), {
          status: 409,
          headers: { "content-type": "application/json" },
        });
      }
    }

    await outFile.save(csvOut, { metadata: { contentType: "text/csv; charset=utf-8" } });

    return new Response(JSON.stringify({ ok: true, labeledPath }), {
      headers: { "content-type": "application/json" },
    });
  } catch (error: any) {
    console.error("[label-trip] error", error);
    return new Response(JSON.stringify({ ok: false, error: String(error?.message || error) }), {
      status: 500,
      headers: { "content-type": "application/json" },
    });
  }
}


