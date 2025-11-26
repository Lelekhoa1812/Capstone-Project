import { NextRequest } from "next/server";
import { storage } from "@/lib/firebase";

interface LabeledFile {
  name: string;
  size: number;
  timeCreated: string;
  duration?: number;
  sessionId: string;
  date: string;
  totalSegments?: number;
  labeledSegments?: number;
  completionRate?: number;
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
    console.log("🔍 Fetching labeled files from Firebase Storage...");
    
    const bucket = storage.bucket();
    const [files] = await bucket.getFiles({
      prefix: 'skyledge/labeled/',
      delimiter: '/'
    });

    console.log(`📁 Found ${files.length} labeled files`);

    const labeledFiles: LabeledFile[] = [];
    const filenamePattern = /^skyledge\/labeled\/(\d{3})_(\d{4}-\d{2}-\d{2})_labeled\.csv$/;

    for (const file of files) {
      const match = file.name.match(filenamePattern);
      if (match) {
        const [, sessionId, date] = match;
        
        // Get file metadata
        const [metadata] = await file.getMetadata();
        
        // Calculate duration and analyze labels by reading the file
        let duration: number | undefined;
        let totalSegments = 0;
        let labeledSegments = 0;
        let completionRate = 0;

        try {
          const [buffer] = await file.download();
          const text = buffer.toString('utf-8');
          const lines = text.split(/\r?\n/).filter(Boolean);
          
          if (lines.length > 1) {
            const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
            const tsIdx = headers.findIndex(h => h === 'timestamp' || h === 'time' || h === 'ts');
            const labelIdx = headers.findIndex(h => h === 'driving_style' || h === 'label' || h === 'driving_label');
            
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

            // Analyze labeling completion if label column exists
            if (labelIdx >= 0) {
              const dataRows = lines.slice(1);
              totalSegments = dataRows.length;
              
              // Count non-empty labels
              labeledSegments = dataRows.filter(row => {
                const cols = row.split(',');
                const label = cols[labelIdx]?.trim();
                return label && label !== '' && label !== 'null' && label !== 'undefined';
              }).length;
              
              completionRate = totalSegments > 0 ? Math.round((labeledSegments / totalSegments) * 100) : 0;
            }
          }
        } catch (error) {
          console.warn(`⚠️ Could not analyze file ${file.name}:`, error);
        }

        labeledFiles.push({
          name: file.name,
          size: parseInt(metadata.size || '0'),
          timeCreated: metadata.timeCreated || new Date().toISOString(),
          duration,
          sessionId,
          date,
          totalSegments,
          labeledSegments,
          completionRate
        });
      }
    }

    // Sort by upload date (newest first)
    labeledFiles.sort((a, b) => new Date(b.timeCreated).getTime() - new Date(a.timeCreated).getTime());

    console.log(`✅ Processed ${labeledFiles.length} labeled files`);

    return new Response(JSON.stringify({ 
      success: true, 
      files: labeledFiles 
    }), { 
      headers: { "content-type": "application/json" } 
    });
  } catch (error) {
    console.error('❌ Error fetching labeled files:', error);
    return new Response(JSON.stringify({ 
      success: false, 
      error: String(error) 
    }), { 
      status: 500,
      headers: { "content-type": "application/json" } 
    });
  }
}
