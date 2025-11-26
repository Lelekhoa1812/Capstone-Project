# SkyLedge

A comprehensive **Predictive Maintenance Dashboard** for automotive OBD-II data analysis, featuring real-time data processing, machine learning labeling tools, and driver behavior insights.

## 🚗 Overview

SkyLedge is a Next.js-based web application designed for automotive data analysis and predictive maintenance. It provides tools for uploading OBD-II data, manually labeling driving behaviors, training machine learning models, and generating driver insights for fleet management and individual drivers.

### Key Features

- **📊 Dual Dashboard System**: Separate interfaces for developers and end users
- **📁 Upload Trip (Driver)**: Send CSV to backend endpoint with loader and local history + date filter
- **🏷️ Manual Labeling Tool**: Interactive timeline-based labeling system for driving behaviors
- **🤖 Machine Learning Pipeline**: Dataset management and reinforcement learning capabilities
- **📈 Driver Analytics**: Comprehensive trip analysis, fuel efficiency tracking, and behavior scoring
- **🧠 Auth Persistence**: Client-side session persists via `localStorage` (no re-login until logout)
- **🗄️ Database**: Firebase Firestore for metadata and upload records
- **☁️ Storage**: Firebase Storage for file uploads and data management
- **🎨 Modern UI**: Built with Radix UI components and Tailwind CSS

## 🏗️ Architecture

### Frontend Stack
- **Framework**: Next.js 15.2.4 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS with custom design system
- **UI Components**: Radix UI primitives
- **State Management**: React hooks and context
- **Charts**: Recharts for data visualization

### Backend Stack
- **Database**: Firebase Firestore for metadata and upload records
- **Storage**: Firebase Storage for file uploads and data management
- **Upload Ingress**: External endpoint `POST https://binkhoale1812-obd-logger.hf.space/upload-csv/`
- **Session Management**: Automatic session ID generation based on latest Firebase uploads

### Key Dependencies
```json
{
  "next": "15.2.4",
  "react": "^19",
  "firebase-admin": "^12.0.0",
  "@radix-ui/react-*": "latest",
  "tailwindcss": "^4.1.9",
  "recharts": "2.15.4",
  "papaparse": "^5.5.3"
}
```

## 🚀 Getting Started

### Prerequisites
- Node.js 18+ 
- npm, yarn, or pnpm
- Firebase project with Firestore and Storage enabled
- Firebase service account credentials (`firebase-sa.json`)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd SkyLedge_UI
   ```

2. **Install dependencies**
   ```bash
   npm install
   # or
   yarn install
   # or
   pnpm install
   ```

3. **Firebase Setup**
   
   Ensure you have:
   - `firebase-sa.json` file in the root directory with your service account credentials
   - Firebase project with Firestore and Storage enabled
   - Storage bucket: `skyledge-36b56.firebasestorage.app`

4. **Start the development server**
   ```bash
   npm run dev
   # or
   yarn dev
   # or
   pnpm dev
   ```

6. **Open your browser**
   
   Navigate to [http://localhost:3000](http://localhost:3000)

## 📱 Application Structure

### User Roles

#### 🧑‍💻 Developer Dashboard (`/developer`)
- **Upload Section**: File upload with automatic session ID generation
- **Manual Labeling**: Interactive timeline for labeling driving behaviors
- **Dataset Management**: View and manage labeled datasets
- **Reinforcement Learning**: Model training and optimization tools

#### 👤 User Dashboard (`/user`)
- **Trip Summary**: Overview of recent trips with driving scores
- **Performance Trends**: Historical data visualization
- **Driver Profile**: Personal statistics and vehicle information
- **Fuel Analytics**: Cost tracking and efficiency metrics
- **Upload Trip** (`/user/upload-trip`): Upload CSV to backend, see local history with date filtering

### Core Components

#### File Upload System
- **Driver Upload**: CSV upload to external endpoint with loader
- **Local History**: Stores filename + timestamp in `localStorage` with date filter
- **Developer Upload**: Developer dashboard upload refactored to use same endpoint

#### Labeling Tool
- **Interactive Timeline**: Visual timeline with playback controls
- **Behavior Classification**: Three driving styles: `idle`, `passive`, `aggressive`
- **Segment Management**: Drag-and-drop segment creation and editing
- **Real-time Metrics**: Live OBD-II data display (speed, RPM, throttle, brake, fuel)

#### Data Processing Pipeline
1. **Upload**: Raw OBD-II CSV posted to external endpoint
2. **Labeling**: Manual annotation using the interactive timeline
3. **Processing**: Downstream processing (external service / backend)
4. **Storage**: Firebase Firestore for metadata and Firebase Storage for files
5. **Analytics**: Driver insights generated from processed data

## 🔧 Configuration

### Firebase Setup

1. **Create Firebase Project**: Set up a new project in Firebase Console
2. **Enable Services**: Enable Firestore Database and Cloud Storage
3. **Create Service Account**: Generate and download service account credentials
4. **Place Credentials**: Save as `firebase-sa.json` in the project root
5. **Storage Bucket**: Ensure bucket is named `skyledge-36b56.firebasestorage.app`

### File Structure

Files are stored in Firebase Storage with the following structure:
- **Raw Data**: `skyledge/raw/{sessionId}_{date}_raw.csv`
- **Labeled Data**: `skyledge/labeled/{sessionId}_{date}_labeled.csv`
- **Session IDs**: Automatically generated based on latest uploads (001-999)

## 📊 Data Format

### OBD-II Data Structure
The application expects CSV files with the following columns:
- `timestamp`: Unix timestamp or ISO date string
- `speed`: Vehicle speed in mph
- `rpm`: Engine RPM
- `throttle`: Throttle position percentage
- `brake`: Brake pressure percentage
- `fuel`: Fuel level percentage

### Labeled Data Output
After processing, labeled files include an additional `driving_style` column with values:
- `idle`: Vehicle stationary or minimal movement
- `passive`: Normal, conservative driving
- `aggressive`: Hard acceleration, braking, or high-speed driving

## 🚀 Deployment

### Vercel Deployment
The application is configured for deployment on Vercel:

1. **Connect Repository**: Link your GitHub repository to Vercel
2. **Environment Variables**: Add the following Firebase service account variables:
   ```bash
   FIREBASE_PROJECT_ID=skyledge-36b56
   FIREBASE_PRIVATE_KEY_ID=[from firebase-adminsdk-fbsvc service account JSON]
   FIREBASE_PRIVATE_KEY="[from firebase-adminsdk-fbsvc service account JSON]"
   FIREBASE_CLIENT_EMAIL=firebase-adminsdk-fbsvc@skyledge-36b56.iam.gserviceaccount.com
   FIREBASE_CLIENT_ID=[from firebase-adminsdk-fbsvc service account JSON]
   FIREBASE_AUTH_URI=https://accounts.google.com/o/oauth2/auth
   FIREBASE_TOKEN_URI=https://oauth2.googleapis.com/token
   FIREBASE_AUTH_PROVIDER_X509_CERT_URL=https://www.googleapis.com/oauth2/v1/certs
   FIREBASE_CLIENT_X509_CERT_URL=[from firebase-adminsdk-fbsvc service account JSON]
   FIREBASE_UNIVERSE_DOMAIN=googleapis.com
   ```
3. **Build Settings**: Uses default Next.js build configuration
4. **Deploy**: Automatic deployment on push to main branch

### Firebase Integration
- **Firestore**: Stores upload metadata and user records
- **Storage**: Handles file uploads and data management
- **Session Management**: Automatic session ID generation based on latest uploads
- **File Naming**: Consistent naming convention `{sessionId}_{date}_raw.csv`

## 🧪 Development

### Available Scripts
```bash
npm run dev          # Start development server
npm run build        # Build for production
npm run start        # Start production server
npm run lint         # Run ESLint
```

### Project Structure
```
SkyLedge_UI/
├── app/                    # Next.js App Router pages
│   ├── developer/         # Developer dashboard
│   ├── user/             # User dashboard
│   └── login/            # Authentication
├── components/           # Reusable UI components
│   ├── ui/              # Radix UI components
│   └── layout/          # Layout components
├── lib/                 # Utilities and configurations
│   ├── data/           # Mock data and types
│   ├── mongo.ts        # MongoDB connection utility
│   ├── auth.ts         # Client auth persistence helpers
│   └── firebase.ts     # Neutralized placeholder (legacy)
├── functions/          # (Legacy) Firebase Cloud Functions
└── hooks/             # Custom React hooks
```

### Key Files
- `app/page.tsx`: Landing page
- `app/developer/page.tsx`: Developer dashboard
- `app/user/page.tsx`: User dashboard
- `lib/mongo.ts`: MongoDB connection utility
- `lib/auth.ts`: Local auth persistence helpers
- `app/user/upload-trip/page.tsx`: Driver upload page
- `components/Timeline.tsx`: Interactive labeling timeline

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Live Demo**: [https://v0-sky-ledge-p2.vercel.app](https://v0-sky-ledge-p2.vercel.app)
- **v0.app**: [https://v0.app](https://v0.app) (Original design source)

---

**SkyLedge** - Transforming automotive data into actionable insights for predictive maintenance and driver behavior analysis.