export const runtime = "edge";

export async function POST(req: Request) {
  const body = await req.json().catch(() => null);
  if (!body?.sourcePath || !Array.isArray(body?.ranges)) {
    return Response.json({ ok: false, error: "Invalid payload" }, { status: 400 });
  }

  // In real impl: write to storage at body.targetFolder.
  // For demo, just echo back.
  const outName = (body.sourcePath as string).split("/").pop()?.replace(/\.csv$/i, "") || "demo";
  const correctedPath = `${body.targetFolder || "skyledge/fixed-labels"}/${outName}.json`;

  return Response.json({
    ok: true,
    correctedPath,
    received: {
      sourcePath: body.sourcePath,
      ranges: body.ranges,
      count: body.ranges.length,
    },
  });
}
