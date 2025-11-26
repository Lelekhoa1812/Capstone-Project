// functions/src/admin.ts
import { initializeApp, getApp } from "firebase-admin/app";
import { getFirestore, FieldValue } from "firebase-admin/firestore";
import { getStorage } from "firebase-admin/storage";

// Ensure a default app exists even if modules are loaded in any order
const app = (() => {
  try {
    return getApp();
  } catch {
    return initializeApp();
  }
})();

export const db = getFirestore(app);
export const storage = getStorage(app);
export { FieldValue };
