import { NextRequest } from "next/server";
import { storage } from "@/lib/firebase";

interface RawLogFile {
  name: string;
  size: number;
  timeCreated: string;
  duration?: number;
  sessionId: string;
  date: string;
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
    console.log("🔍 Fetching raw logs from Firebase Storage...");
    
    const bucket = storage.bucket();
    const [files] = await bucket.getFiles({
      prefix: 'skyledge/raw/',
      delimiter: '/'
    });

    console.log(`📁 Found ${files.length} raw log files`);

    const rawLogs: RawLogFile[] = [];
    const filenamePattern = /^skyledge\/raw\/(\d{3})_(\d{4}-\d{2}-\d{2})_raw\.csv$/;

    for (const file of files) {
      const match = file.name.match(filenamePattern);
      if (match) {
        const [, sessionId, date] = match;
        
        // Get file metadata
        const [metadata] = await file.getMetadata();
        
        // Calculate duration by reading the file
        let duration: number | undefined;
        try {
          const [buffer] = await file.download();
          const text = buffer.toString('utf-8');
          const lines = text.split(/\r?\n/).filter(Boolean);
          
          if (lines.length > 1) {
            const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
            const tsIdx = headers.findIndex(h => h === 'timestamp' || h === 'time' || h === 'ts');
            
            if (tsIdx >= 0) {
              const firstRow = lines[1].split(',');
              const lastRow = lines[lines.length - 1].split(',');
              
              const parseTs = (v: string) => {
                const num = Number(v);
                if (!Number.isNaN(num)) return num >= 1e11 ? num : num * 1000;
                const d = Date.parse(v);
                return isNaN(d) ? undefined : d;
              };
              
              const t0 = parseTs(firstRow[tsIdx] || '');
              const t1 = parseTs(lastRow[tsIdx] || '');
              
              if (typeof t0 === 'number' && typeof t1 === 'number' && t1 >= t0) {
                duration = Math.round((t1 - t0) / 1000);
              }
            }
          }
        } catch (error) {
          console.warn(`⚠️ Could not calculate duration for ${file.name}:`, error);
        }

        rawLogs.push({
          name: file.name,
          size: parseInt(metadata.size || '0'),
          timeCreated: metadata.timeCreated || new Date().toISOString(),
          duration,
          sessionId,
          date
        });
      }
    }

    // Sort by upload date (newest first)
    rawLogs.sort((a, b) => new Date(b.timeCreated).getTime() - new Date(a.timeCreated).getTime());

    console.log(`✅ Processed ${rawLogs.length} raw log files`);

    return new Response(JSON.stringify({ 
      success: true, 
      files: rawLogs 
    }), { 
      headers: { "content-type": "application/json" } 
    });
  } catch (error) {
    console.error('❌ Error fetching raw logs:', error);
    return new Response(JSON.stringify({ 
      success: false, 
      error: String(error) 
    }), { 
      status: 500,
      headers: { "content-type": "application/json" } 
    });
  }
}
