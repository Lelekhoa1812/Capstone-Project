// lib/listRecentFiles.ts
import { storage } from "./firebase";
import { ref, list, getDownloadURL } from "firebase/storage";

export async function listRecentFiles(limit = 20) {
  const folderRef = ref(storage, "skyledge/raw");
  // 'list' paginates; 'listAll' returns everything (not recommended for huge folders)
  const first = await list(folderRef, { maxResults: limit });
  const items = await Promise.all(
    first.items.map(async (itemRef) => ({
      name: itemRef.name,
      fullPath: itemRef.fullPath,
      url: await getDownloadURL(itemRef),
    }))
  );
  // Note: Storage doesn't preserve upload time in listing; if you need timestamps, encode in filename (we do)
  // You can sort by name (since we used yyyyMMdd-HHmmss) to approximate recency:
  return items.sort((a, b) => (a.name > b.name ? -1 : 1));
}
