// lib/firebase.ts
import { initializeApp, getApps, cert } from 'firebase-admin/app';
import { getStorage } from 'firebase-admin/storage';
import { getFirestore } from 'firebase-admin/firestore';
import { getAuth } from 'firebase-admin/auth';
import { getFunctions } from 'firebase-admin/functions';

// Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyC5FRDIr7AJo7nprJdlAqRcoFjh-bOjXlc",
  authDomain: "skyledge-36b56.firebaseapp.com",
  projectId: "skyledge-36b56",
  storageBucket: "skyledge-36b56.firebasestorage.app",
  messagingSenderId: "412586225510",
  appId: "1:412586225510:web:7f074db14e973d9eb5e2a7",
  measurementId: "G-G5QMJ16KHQ"
};

// Initialize Firebase Admin SDK (server-side only)
let firebaseApp: any = null;
let storage: any = null;
let db: any = null;
let auth: any = null;
let functions: any = null;

// Only initialize on server-side
if (typeof window === 'undefined') {
  try {
    if (getApps().length === 0) {
      console.log("🔧 Initializing Firebase Admin SDK...");
      
      // Use environment variables for production, fallback to local file for development
      let serviceAccount;
      
      if (process.env.FIREBASE_PRIVATE_KEY) {
        console.log("🌐 Using environment variables for Firebase configuration");
        serviceAccount = {
          type: "service_account",
          project_id: process.env.FIREBASE_PROJECT_ID || firebaseConfig.projectId,
          private_key_id: process.env.FIREBASE_PRIVATE_KEY_ID,
          private_key: process.env.FIREBASE_PRIVATE_KEY?.replace(/\\n/g, '\n'),
          client_email: process.env.FIREBASE_CLIENT_EMAIL || "firebase-adminsdk-fbsvc@skyledge-36b56.iam.gserviceaccount.com",
          client_id: process.env.FIREBASE_CLIENT_ID,
          auth_uri: process.env.FIREBASE_AUTH_URI || "https://accounts.google.com/o/oauth2/auth",
          token_uri: process.env.FIREBASE_TOKEN_URI || "https://oauth2.googleapis.com/token",
          auth_provider_x509_cert_url: process.env.FIREBASE_AUTH_PROVIDER_X509_CERT_URL || "https://www.googleapis.com/oauth2/v1/certs",
          client_x509_cert_url: process.env.FIREBASE_CLIENT_X509_CERT_URL || `https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-fbsvc%40skyledge-36b56.iam.gserviceaccount.com`,
          universe_domain: process.env.FIREBASE_UNIVERSE_DOMAIN || "googleapis.com"
        };
      } else if (process.env.GOOGLE_APPLICATION_CREDENTIALS) {
        console.log("🔑 Using Application Default Credentials");
        // Use Application Default Credentials - no service account object needed
        serviceAccount = undefined;
      } else {
        console.log("📁 Using local firebase-adminsdk-sa.json file");
        try {
          serviceAccount = require('../firebase-adminsdk-sa.json');
        } catch (error) {
          console.log("⚠️ firebase-adminsdk-sa.json not found, falling back to firebase-sa.json");
          serviceAccount = require('../firebase-sa.json');
        }
      }

      console.log("🔧 Firebase service account loaded:", {
        project_id: serviceAccount.project_id,
        client_email: serviceAccount.client_email,
        has_private_key: !!serviceAccount.private_key
      });

      firebaseApp = initializeApp({
        credential: serviceAccount ? cert(serviceAccount as any) : undefined,
        storageBucket: firebaseConfig.storageBucket,
        projectId: firebaseConfig.projectId
      });
      
      console.log("✅ Firebase Admin SDK initialized successfully");
    } else {
      firebaseApp = getApps()[0];
      console.log("♻️ Using existing Firebase app");
    }

    // Initialize services
    storage = getStorage(firebaseApp);
    db = getFirestore(firebaseApp);
    auth = getAuth(firebaseApp);
    functions = getFunctions(firebaseApp);
    
    console.log("✅ Firebase services initialized:", {
      storage: !!storage,
      db: !!db,
      auth: !!auth,
      functions: !!functions
    });
  } catch (error) {
    console.error("❌ Firebase initialization failed:", error);
    console.error("🔍 Error details:", {
      message: error.message,
      stack: error.stack
    });
  }
}

// Export Firebase services (undefined on client-side)
export { storage, db, auth, functions };

// Helper function to ensure anonymous authentication (if needed)
export async function ensureAnon() {
  // Firebase Admin SDK doesn't require client-side auth for server operations
  return;
}

// Cloud Functions calls (placeholder for now)
export const callLabelTrip = undefined as unknown as any;
export const callListEligible = undefined as unknown as any;
export const callGetTripMeta = undefined as unknown as any;