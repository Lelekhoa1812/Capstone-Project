import { NextRequest } from "next/server";
import { storage } from "@/lib/firebase";
import Papa from "papaparse";

interface SensorData {
  index: number;
  rpm: number;
  engineLoad: number;
  intakePressure: number;
}

export async function GET(req: NextRequest) {
  if (!storage) {
    return new Response(JSON.stringify({ 
      success: false, 
      error: "Storage not initialized" 
    }), { 
      status: 500,
      headers: { "content-type": "application/json" } 
    });
  }

  try {
    const url = new URL(req.url);
    const fileName = url.searchParams.get('fileName');

    if (!fileName) {
      return new Response(JSON.stringify({ 
        success: false, 
        error: "fileName parameter is required" 
      }), { 
        status: 400,
        headers: { "content-type": "application/json" } 
      });
    }

    console.log(`📊 Fetching sensor data for: ${fileName}`);

    const bucket = storage.bucket();
    const file = bucket.file(fileName);

    // Check if file exists
    const [exists] = await file.exists();
    if (!exists) {
      return new Response(JSON.stringify({ 
        success: false, 
        error: "File not found" 
      }), { 
        status: 404,
        headers: { "content-type": "application/json" } 
      });
    }

    // Download file content
    const [buffer] = await file.download();
    const text = buffer.toString('utf-8');

    // Parse CSV
    const parsed = Papa.parse(text, { 
      header: true, 
      skipEmptyLines: true,
      dynamicTyping: true
    });

    if (parsed.errors?.length) {
      console.warn("CSV parse warnings:", parsed.errors);
    }

    const rows = parsed.data as any[];
    if (rows.length === 0) {
      return new Response(JSON.stringify({ 
        success: false, 
        error: "No data found in file" 
      }), { 
        status: 400,
        headers: { "content-type": "application/json" } 
      });
    }

    // Find sensor columns (case-insensitive)
    const headers = Object.keys(rows[0]);
    const findColumn = (patterns: string[]) => {
      for (const pattern of patterns) {
        const found = headers.find(h => 
          h.toLowerCase().includes(pattern.toLowerCase())
        );
        if (found) return found;
      }
      return null;
    };

    const rpmColumn = findColumn(['rpm', 'engine_rpm', 'engine_speed']);
    const engineLoadColumn = findColumn(['engine_load', 'load', 'engine_load_percent']);
    const intakePressureColumn = findColumn(['intake_pressure', 'manifold_pressure', 'map']);

    if (!rpmColumn || !engineLoadColumn || !intakePressureColumn) {
      return new Response(JSON.stringify({ 
        success: false, 
        error: `Required sensor columns not found. Available columns: ${headers.join(', ')}` 
      }), { 
        status: 400,
        headers: { "content-type": "application/json" } 
      });
    }

    console.log(`📈 Found sensor columns: RPM=${rpmColumn}, Engine Load=${engineLoadColumn}, Intake Pressure=${intakePressureColumn}`);

    // Process data
    const sensorData: SensorData[] = rows.map((row, index) => ({
      index,
      rpm: Number(row[rpmColumn]) || 0,
      engineLoad: Number(row[engineLoadColumn]) || 0,
      intakePressure: Number(row[intakePressureColumn]) || 0,
    }));

    // Calculate file info
    const totalSamples = sensorData.length;
    const duration = calculateDuration(rows);
    const sampleRate = totalSamples / Math.max(duration, 1);

    console.log(`✅ Processed ${totalSamples} sensor samples`);

    return new Response(JSON.stringify({ 
      success: true, 
      data: sensorData,
      fileInfo: {
        totalSamples,
        duration,
        sampleRate
      }
    }), { 
      headers: { "content-type": "application/json" } 
    });
  } catch (error) {
    console.error('❌ Error fetching sensor data:', error);
    return new Response(JSON.stringify({ 
      success: false, 
      error: String(error) 
    }), { 
      status: 500,
      headers: { "content-type": "application/json" } 
    });
  }
}

function calculateDuration(rows: any[]): number {
  if (rows.length < 2) return 0;

  // Try to find timestamp column
  const headers = Object.keys(rows[0]);
  const timestampColumn = headers.find(h => 
    h.toLowerCase().includes('timestamp') || 
    h.toLowerCase().includes('time') || 
    h.toLowerCase().includes('ts')
  );

  if (!timestampColumn) return 0;

  const firstRow = rows[0];
  const lastRow = rows[rows.length - 1];

  const parseTs = (v: any) => {
    if (v == null || v === "") return null;
    if (typeof v === "number") return v >= 1e11 ? v : v * 1000;
    if (/^\d+(\.\d+)?$/.test(String(v))) {
      const num = Number(v);
      return num >= 1e11 ? num : num * 1000;
    }
    const ms = Date.parse(String(v));
    return isNaN(ms) ? null : ms;
  };

  const t0 = parseTs(firstRow[timestampColumn]);
  const t1 = parseTs(lastRow[timestampColumn]);

  if (t0 && t1 && t1 >= t0) {
    return Math.round((t1 - t0) / 1000);
  }

  return 0;
}
