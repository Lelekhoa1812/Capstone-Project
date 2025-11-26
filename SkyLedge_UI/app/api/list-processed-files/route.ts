// app/api/list-processed-files/route.ts
import { NextResponse } from 'next/server';
import admin from 'firebase-admin';

// Initialize Firebase Admin (if not already done)
if (!admin.apps.length) {
  admin.initializeApp({
    credential: admin.credential.cert({
      projectId: process.env.FIREBASE_PROJECT_ID,
      clientEmail: process.env.FIREBASE_CLIENT_EMAIL,
      privateKey: process.env.FIREBASE_PRIVATE_KEY?.replace(/\\n/g, '\n'),
    }),
    storageBucket: process.env.FIREBASE_STORAGE_BUCKET,
  });
}

export async function GET() {
  try {
    const bucket = admin.storage().bucket();
    const [files] = await bucket.getFiles({ prefix: 'skyledge/processed/' });
    
    const fileNames = files
      .map(file => file.name.replace('skyledge/processed/', ''))
      .filter(name => name.match(/^\d{3}_\d{4}-\d{2}-\d{2}_processed\.csv$/));
    
    return NextResponse.json({ success: true, files: fileNames });
  } catch (error) {
    console.error('Error listing processed files:', error);
    return NextResponse.json({ success: false, error: String(error) }, { status: 500 });
  }
}