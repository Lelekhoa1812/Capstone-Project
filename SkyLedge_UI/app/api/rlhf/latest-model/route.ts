export const runtime = "edge";

export async function GET() {
  try {
    // Forward request to the actual RLHF latest model service
    const response = await fetch("https://binkhoale1812-obd-logger.hf.space/rlhf/latest-model", {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      }
    });

    const data = await response.json();

    if (!response.ok) {
      return Response.json({ 
        ok: false, 
        error: data.error || "Failed to fetch latest model" 
      }, { status: response.status });
    }

    return Response.json({
      ok: true,
      ...data
    });

  } catch (error) {
    console.error("RLHF latest model error:", error);
    return Response.json({ 
      ok: false, 
      error: "Internal server error" 
    }, { status: 500 });
  }
}
