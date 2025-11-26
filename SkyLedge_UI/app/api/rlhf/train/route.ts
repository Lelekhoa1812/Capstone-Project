export const runtime = "edge";

export async function POST(req: Request) {
  try {
    const body = await req.json().catch(() => null);
    if (!body) {
      return Response.json({ ok: false, error: "Invalid JSON payload" }, { status: 400 });
    }

    const { max_datasets = 5, force_retrain = false } = body;

    // Validate max_datasets
    if (typeof max_datasets !== 'number' || max_datasets < 1 || max_datasets > 100) {
      return Response.json({ 
        ok: false, 
        error: "max_datasets must be a number between 1 and 100" 
      }, { status: 400 });
    }

    // Forward request to the actual RLHF training service
    const response = await fetch("https://binkhoale1812-obd-logger.hf.space/rlhf/train", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        max_datasets,
        force_retrain
      })
    });

    const data = await response.json();

    if (!response.ok) {
      return Response.json({ 
        ok: false, 
        error: data.error || "Training request failed" 
      }, { status: response.status });
    }

    return Response.json({
      ok: true,
      message: "RLHF training started successfully",
      max_datasets,
      force_retrain,
      ...data
    });

  } catch (error) {
    console.error("RLHF training error:", error);
    return Response.json({ 
      ok: false, 
      error: "Internal server error" 
    }, { status: 500 });
  }
}
