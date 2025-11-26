import { NextRequest } from "next/server";
import { storage } from "@/lib/firebase";

export async function GET(req: NextRequest) {
  console.log("🔍 Session ID API called");
  console.log("🔧 Storage initialized:", !!storage);
  
  if (!storage) {
    console.log("❌ Firebase Storage not initialized");
    console.log("🔍 Checking Firebase configuration...");
    console.log("🔧 Environment variables:", {
      FIREBASE_PROJECT_ID: !!process.env.FIREBASE_PROJECT_ID,
      FIREBASE_PRIVATE_KEY: !!process.env.FIREBASE_PRIVATE_KEY,
      FIREBASE_CLIENT_EMAIL: !!process.env.FIREBASE_CLIENT_EMAIL,
    });
    return new Response(JSON.stringify({ 
      error: "Storage not initialized",
      sessionId: "001" // Fallback
    }), { 
      status: 500,
      headers: { "content-type": "application/json" } 
    });
  }
  
  try {
    console.log("🔍 Checking Firebase Storage for existing files...");
    
    // List all files in the raw directory
    const bucket = storage.bucket();
    const [files] = await bucket.getFiles({
      prefix: 'skyledge/raw/',
      delimiter: '/'
    });

    console.log(`📁 Found ${files.length} files in Firebase Storage`);

    // Extract session numbers from filenames
    const sessionNumbers: number[] = [];
    const filenamePattern = /^skyledge\/raw\/(\d{3})_\d{4}-\d{2}-\d{2}_raw\.csv$/;
    
    files.forEach(file => {
      const match = file.name.match(filenamePattern);
      if (match) {
        const sessionNum = parseInt(match[1], 10);
        sessionNumbers.push(sessionNum);
        console.log(`📄 Found file: ${file.name} -> Session ID: ${match[1]}`);
      }
    });

    console.log(`🔢 Extracted session numbers: [${sessionNumbers.sort((a, b) => a - b).join(', ')}]`);

    // Get the next session ID
    let nextSessionId = 1;
    if (sessionNumbers.length > 0) {
      const maxSessionId = Math.max(...sessionNumbers);
      console.log(`📊 Current highest session ID: ${maxSessionId}`);
      
      nextSessionId = maxSessionId + 1;
      if (nextSessionId > 999) {
        // Find the first available number starting from 1
        for (let i = 1; i <= 999; i++) {
          if (!sessionNumbers.includes(i)) {
            nextSessionId = i;
            break;
          }
        }
        console.log(`🔄 Wrapped around, found next available: ${nextSessionId}`);
      }
    } else {
      console.log("🆕 No existing files found, starting with session ID: 001");
    }

    const paddedSessionId = String(nextSessionId).padStart(3, '0');
    
    console.log(`✅ Next session ID: ${paddedSessionId}`);
    
    return new Response(JSON.stringify({ 
      sessionId: paddedSessionId,
      totalFiles: files.length,
      existingSessionIds: sessionNumbers.sort((a, b) => a - b),
      lastSessionId: sessionNumbers.length > 0 ? Math.max(...sessionNumbers) : 0
    }), { 
      headers: { "content-type": "application/json" } 
    });
  } catch (error) {
    console.error('❌ Error getting session ID:', error);
    return new Response(JSON.stringify({ 
      error: "Failed to get session ID",
      sessionId: "001" // Fallback
    }), { 
      status: 500,
      headers: { "content-type": "application/json" } 
    });
  }
}
