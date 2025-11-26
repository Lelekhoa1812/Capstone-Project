import { storage } from "./firebase";
import { ref, uploadBytesResumable, getDownloadURL } from "firebase/storage";

function tsAEST(d = new Date()) {
  const p = (n: number) => String(n).padStart(2, "0");
  const yyyy = d.getFullYear();
  const MM = p(d.getMonth() + 1);
  const dd = p(d.getDate());
  const HH = p(d.getHours());
  const mm = p(d.getMinutes());
  const ss = p(d.getSeconds());
  return `${yyyy}${MM}${dd}-${HH}${mm}${ss}`;
}

export async function uploadCsvToFirebase(opts: {
  file: File;
  vehicleId: string;
  sessionId: string;
}) {
  const { file, vehicleId, sessionId } = opts;

  if (!file) throw new Error("uploadCsvToFirebase: file missing");
  if (!vehicleId) throw new Error("uploadCsvToFirebase: vehicleId missing");
  if (!sessionId) throw new Error("uploadCsvToFirebase: sessionId missing");

  const name = `${vehicleId}_${sessionId}_${tsAEST()}_raw.csv`;
  const path = `skyledge/raw/${name}`;
  console.log("[upload] begin", { path, name, size: file.size, type: file.type });

  const task = uploadBytesResumable(ref(storage, path), file, { contentType: "text/csv" });

  return await new Promise<{ path: string; name: string; downloadURL: string }>((resolve, reject) => {
    task.on(
      "state_changed",
      (snap) => {
        const pct = (snap.bytesTransferred / snap.totalBytes) * 100;
        console.log(`[upload] progress ${pct.toFixed(1)}% (${snap.bytesTransferred}/${snap.totalBytes})`);
      },
      (err) => {
        console.error("[upload] error event", err);
        reject(err);
      },
      async () => {
        const downloadURL = await getDownloadURL(task.snapshot.ref);
        console.log("[upload] success", { path, name, downloadURL });
        resolve({ path, name, downloadURL });
      }
    );
  });
}
