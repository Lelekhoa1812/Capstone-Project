export const runtime = "edge";

function makeDemoCSV(): string {
  // 60 seconds, label changes every 15s
  const start = Date.now();
  const labels = ["idle", "passive", "moderate", "aggressive"] as const;
  const rows: string[] = ["timestamp,driving_style"];
  for (let i = 0; i < 60; i++) {
    const ts = new Date(start + i * 1000).toISOString();
    const lab = labels[Math.floor(i / 15) % labels.length];
    rows.push(`${ts},${lab}`);
  }
  return rows.join("\n");
}

export async function GET(req: Request) {
  const url = new URL(req.url);
  const fileName = url.searchParams.get("fileName");
  if (!fileName) return new Response("fileName required", { status: 400 });

  // For demo, always return generated CSV regardless of name
  const csv = makeDemoCSV();
  return new Response(csv, {
    status: 200,
    headers: {
      "content-type": "text/csv; charset=utf-8",
      "cache-control": "no-store",
    },
  });
}
