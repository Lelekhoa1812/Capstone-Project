// lib/firebaseUpload.ts
import { storage } from "@/lib/firebase";

export async function uploadFileToFirebaseStorage(
  file: File, 
  fileName: string
): Promise<{ success: boolean; downloadURL?: string; error?: string }> {
  if (!storage) {
    return { success: false, error: "Firebase Storage not initialized" };
  }

  try {
    const bucket = storage.bucket();
    const fileRef = bucket.file(fileName);
    
    // Convert File to Buffer
    const arrayBuffer = await file.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);
    
    // Upload to Firebase Storage
    await fileRef.save(buffer, {
      metadata: {
        contentType: file.type || 'text/csv',
        cacheControl: 'public, max-age=31536000',
      },
    });

    // Get download URL
    const [downloadURL] = await fileRef.getSignedUrl({
      action: 'read',
      expires: Date.now() + 1000 * 60 * 60 * 24 * 7, // 7 days
    });

    console.log(`✅ Firebase Storage upload successful: ${fileName}`);
    console.log(`📁 Download URL: ${downloadURL}`);

    return { success: true, downloadURL };
  } catch (error) {
    console.error(`❌ Firebase Storage upload failed for ${fileName}:`, error);
    return { success: false, error: String(error) };
  }
}
