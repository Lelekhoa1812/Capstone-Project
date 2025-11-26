export const runtime = "edge"; // or "nodejs"

export async function GET() {
  // Return a single demo CSV 
  return Response.json({
    files: ["skyledge/prediction/demo.csv"],
  });
}
