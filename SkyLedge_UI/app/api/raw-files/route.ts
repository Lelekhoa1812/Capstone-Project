import { NextRequest } from "next/server";
import { storage } from "@/lib/firebase";

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
    console.log("🔍 Fetching raw files from Firebase Storage...");
    
    const bucket = storage.bucket();
    const [files] = await bucket.getFiles({
      prefix: 'skyledge/raw/',
      delimiter: '/'
    });

    console.log(`📁 Found ${files.length} raw files`);

    const rawFiles = files
      .map(file => file.name)
      .filter(name => name.endsWith('_raw.csv'))
      .sort();

    console.log(`✅ Processed ${rawFiles.length} raw files`);

    return new Response(JSON.stringify({ 
      success: true, 
      files: rawFiles 
    }), { 
      headers: { "content-type": "application/json" } 
    });
  } catch (error) {
    console.error('❌ Error fetching raw files:', error);
    return new Response(JSON.stringify({ 
      success: false, 
      error: String(error) 
    }), { 
      status: 500,
      headers: { "content-type": "application/json" } 
    });
  }
}
