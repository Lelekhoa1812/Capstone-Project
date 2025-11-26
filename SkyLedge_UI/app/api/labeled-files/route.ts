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
    console.log("🔍 Fetching labeled files from Firebase Storage...");
    
    const bucket = storage.bucket();
    const [files] = await bucket.getFiles({
      prefix: 'skyledge/labeled/',
      delimiter: '/'
    });

    console.log(`📁 Found ${files.length} labeled files`);

    const labeledFiles = files
      .map(file => file.name)
      .filter(name => name.endsWith('_labeled.csv'))
      .sort();

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
