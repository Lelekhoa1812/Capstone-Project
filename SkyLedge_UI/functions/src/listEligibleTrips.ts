import { onCall, HttpsError } from "firebase-functions/v2/https";
import { storage } from "./admin.js"; // ✅ use shared admin

export const listEligibleTrips = onCall(async (req) => {
  if (!req.auth) throw new HttpsError("unauthenticated", "Sign in.");

  const bucket = storage.bucket();
  const [rawFiles] = await bucket.getFiles({ prefix: "skyledge/raw/", autoPaginate: true });
  const [labeledFiles] = await bucket.getFiles({ prefix: "skyledge/labeled/", autoPaginate: true });

  const labeledSet = new Set(
    labeledFiles
      .filter((f) => f.name.endsWith("_labeled.csv"))
      .map((f) => f.name.split("/").pop()!)
  );

  const toLabeledName = (rawName: string) => rawName.replace("_raw.csv", "_labeled.csv");

  const eligible = rawFiles
    .map((f) => f.name.split("/").pop()!)
    .filter((n) => n.endsWith("_raw.csv"))
    .filter((n) => !labeledSet.has(toLabeledName(n)));

  return { trips: eligible };
});
