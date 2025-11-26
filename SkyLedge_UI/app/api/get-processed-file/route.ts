// app/api/get-processed-file/route.ts
import { NextResponse } from 'next/server';
import admin from 'firebase-admin';

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const filename = searchParams.get('filename');
    
    if (!filename) {
      return new Response('Filename required', { status: 400 });
    }
    
    const bucket = admin.storage().bucket();
    const file = bucket.file(`skyledge/processed/${filename}`);
    
    const [content] = await file.download();
    
    return new Response(content.toString('utf-8'), {
      headers: { 'Content-Type': 'text/csv' }
    });
  } catch (error) {
    console.error('Error downloading file:', error);
    return new Response('Failed to download file', { status: 500 });
  }
}