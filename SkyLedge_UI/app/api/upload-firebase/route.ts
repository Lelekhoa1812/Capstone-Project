import { NextRequest } from "next/server";
import { uploadFileToFirebaseStorage } from "@/lib/firebaseUpload";

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get('file') as File;
    const fileName = formData.get('fileName') as string;

    if (!file || !fileName) {
      return new Response(JSON.stringify({ 
        success: false, 
        error: "File and fileName are required" 
      }), { 
        status: 400,
        headers: { "content-type": "application/json" } 
      });
    }

    console.log(`🚀 Server-side Firebase upload: ${fileName}`);
    const result = await uploadFileToFirebaseStorage(file, fileName);
    
    return new Response(JSON.stringify(result), { 
      headers: { "content-type": "application/json" } 
    });
  } catch (error) {
    console.error('❌ Server-side Firebase upload error:', error);
    return new Response(JSON.stringify({ 
      success: false, 
      error: String(error) 
    }), { 
      status: 500,
      headers: { "content-type": "application/json" } 
    });
  }
}
