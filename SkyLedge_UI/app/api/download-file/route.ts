import { NextRequest } from "next/server";
import { storage } from "@/lib/firebase";

export async function GET(req: NextRequest) {
  if (!storage) {
    return new Response("Storage not initialized", { status: 500 });
  }

  try {
    const url = new URL(req.url);
    const fileName = url.searchParams.get('fileName');

    if (!fileName) {
      return new Response("fileName parameter is required", { status: 400 });
    }

    console.log(`📥 Downloading file: ${fileName}`);

    const bucket = storage.bucket();
    const file = bucket.file(fileName);

    // Check if file exists
    const [exists] = await file.exists();
    if (!exists) {
      return new Response("File not found", { status: 404 });
    }

    // Download file content
    const [buffer] = await file.download();
    
    // Get file metadata for content type
    const [metadata] = await file.getMetadata();
    const contentType = metadata.contentType || 'text/csv';

    console.log(`✅ File downloaded successfully: ${fileName}`);

    return new Response(buffer, {
      headers: {
        'Content-Type': contentType,
        'Content-Disposition': `attachment; filename="${fileName.split('/').pop()}"`,
        'Content-Length': buffer.length.toString(),
      },
    });
  } catch (error) {
    console.error('❌ Download error:', error);
    return new Response("Download failed", { status: 500 });
  }
}
