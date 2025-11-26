"use client";

import AppHeader from "@/components/layout/AppHeader";
import Sidebar from "@/components/layout/Sidebar";
import { Upload, Tag, Database, Brain, FileText, Cpu, BarChart3  } from "lucide-react";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { clearStoredUser, getStoredUser } from "@/lib/auth";

const devNav = [
  { label: "Developer Dashboard" },
  { href: "/developer#upload", label: "Upload / Buffer", icon: Upload },
  { href: "/developer#raw-logs", label: "Raw Logs", icon: FileText },
  { href: "/developer#data-processing", label: "Data Processing", icon: Cpu },
  { href: "/developer#labeling", label: "Manual Labeling", icon: Tag },
  { href: "/developer#dataset", label: "Labeled Dataset", icon: Database, badge: "3" },
  { href: "/developer#reinforcement", label: "Reinforcement", icon: Brain },
  { href: "/developer#processed-trips", label: "Processed Trips", icon: BarChart3 },
];

export default function DevLayout({ children }: { children: React.ReactNode }) {
  const [email, setEmail] = useState<string>("");
  const router = useRouter();

  useEffect(() => {
    const u = getStoredUser();
    if (!u || u.role !== "developer") {
      router.replace("/login");
      return;
    }
    setEmail(u.email);
  }, [router]);
  return (
    <div className="min-h-screen bg-background">
      <AppHeader
        title="SkyLedge"
        roleLabel="Developer"
        email={email}
        onLogout={() => {
          clearStoredUser();
          window.location.href = "/login";
        }}
        sidebarItems={devNav}
        sidebarTitle="SkyLedge — Dev"
        sidebarSubtitle="Predictive Maintenance"
      />

      <div className="pt-14 flex">
        {/* Desktop sidebar */}
        <div className="hidden md:block sticky top-14 h-[calc(100vh-3.5rem)]">
          <Sidebar title="SkyLedge — Dev" subtitle="Predictive Maintenance" items={devNav} />
        </div>

        {/* No extra ml! */}
        <main className="flex-1 p-6">{children}</main>
      </div>
    </div>
  );
}
