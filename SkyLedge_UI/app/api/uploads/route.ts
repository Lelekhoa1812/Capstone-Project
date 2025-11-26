import { NextRequest } from "next/server";
import { db } from "@/lib/firebase";

type UploadRecord = {
  id?: string;
  filename: string;
  size?: number;
  durationSec?: number;
  deviceId?: string;
  uploadedAt: number; // epoch ms
  status?: "pending" | "uploaded" | "processed" | "error";
};

export async function GET(req: NextRequest) {
  if (!db) {
    return new Response(JSON.stringify({ error: "Database not initialized" }), { 
      status: 500, 
      headers: { "content-type": "application/json" } 
    });
  }
  
  const url = new URL(req.url);
  const limitParam = url.searchParams.get("limit");
  const limit = Math.min(Number(limitParam || 20) || 20, 200);
  
  const snapshot = await db
    .collection("uploads")
    .orderBy("uploadedAt", "desc")
    .limit(limit)
    .get();
  
  const items = snapshot.docs.map(doc => ({
    id: doc.id,
    ...doc.data()
  })) as UploadRecord[];
  
  return new Response(JSON.stringify(items), { headers: { "content-type": "application/json" } });
}

export async function POST(req: NextRequest) {
  if (!db) {
    return new Response(JSON.stringify({ error: "Database not initialized" }), { 
      status: 500, 
      headers: { "content-type": "application/json" } 
    });
  }
  
  const body = (await req.json()) as Partial<UploadRecord>;
  if (!body || !body.filename) {
    return new Response(JSON.stringify({ error: "filename is required" }), { status: 400 });
  }
  
  const rec: Omit<UploadRecord, 'id'> = {
    filename: body.filename,
    size: body.size,
    durationSec: body.durationSec,
    deviceId: body.deviceId,
    uploadedAt: body.uploadedAt || Date.now(),
    status: body.status || "pending",
  };
  
  const docRef = await db.collection("uploads").add(rec);
  return new Response(JSON.stringify({ ok: true, id: docRef.id }), { headers: { "content-type": "application/json" } });
}

export async function DELETE(req: NextRequest) {
  if (!db) {
    return new Response(JSON.stringify({ error: "Database not initialized" }), { 
      status: 500, 
      headers: { "content-type": "application/json" } 
    });
  }
  
  const url = new URL(req.url);
  const filename = url.searchParams.get("filename");
  const purgeInvalid = url.searchParams.get("purgeInvalid");
  
  if (filename) {
    const snapshot = await db.collection("uploads").where("filename", "==", filename).get();
    const batch = db.batch();
    snapshot.docs.forEach(doc => batch.delete(doc.ref));
    await batch.commit();
    return new Response(JSON.stringify({ ok: true, deleted: filename }), { headers: { "content-type": "application/json" } });
  }
  
  if (purgeInvalid) {
    const pattern = /^skyledge\/raw\/\d{3}_\d{4}-\d{2}-\d{2}_raw\.csv$/;
    const snapshot = await db.collection("uploads").get();
    const invalid = snapshot.docs.filter(doc => {
      const data = doc.data();
      return !pattern.test(data.filename || "");
    });
    
    if (invalid.length > 0) {
      const batch = db.batch();
      invalid.forEach(doc => batch.delete(doc.ref));
      await batch.commit();
      return new Response(JSON.stringify({ ok: true, purged: invalid.length }), { headers: { "content-type": "application/json" } });
    }
    return new Response(JSON.stringify({ ok: true, purged: 0 }), { headers: { "content-type": "application/json" } });
  }
  
  return new Response(JSON.stringify({ ok: false, error: "missing filename or purgeInvalid" }), { status: 400 });
}


