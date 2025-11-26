import { NextRequest } from "next/server";
import { storage } from "@/lib/firebase";

export async function GET(req: NextRequest) {
  if (!storage) {
    return new Response(JSON.stringify({ success: false, error: "Storage not initialized" }), {
      status: 500,
      headers: { "content-type": "application/json" },
    });
  }

  try {
    const bucket = storage.bucket();
    const [rawFiles] = await bucket.getFiles({ prefix: "skyledge/raw/" });
    const [labeledFiles] = await bucket.getFiles({ prefix: "skyledge/labeled/" });

    const labeledSet = new Set(
      labeledFiles
        .map((f) => f.name)
        .filter((n) => n.endsWith("_labeled.csv"))
        .map((n) => n.replace("skyledge/labeled/", "").replace("_labeled.csv", ""))
    );

    const eligible = rawFiles
      .map((f) => f.name)
      .filter((n) => n.endsWith("_raw.csv"))
      .map((n) => n.replace("skyledge/raw/", "").replace("_raw.csv", ""))
      .filter((base) => !labeledSet.has(base));

    return new Response(JSON.stringify({ success: true, trips: eligible }), {
      headers: { "content-type": "application/json" },
    });
  } catch (error) {
    console.error("[eligible-trips] error", error);
    return new Response(JSON.stringify({ success: false, error: String(error) }), {
      status: 500,
      headers: { "content-type": "application/json" },
    });
  }
}


