"use client";

import AppHeader from "@/components/layout/AppHeader";
import Sidebar from "@/components/layout/Sidebar";
import { LayoutDashboard, TrendingUp, MapPin, User as UserIcon, Upload } from "lucide-react";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { clearStoredUser, getStoredUser } from "@/lib/auth";

const userNav = [
  { label: "Driver Dashboard" },
  { href: "/user#summary", label: "Trip Summary", icon: MapPin },
  { href: "/user#trends", label: "Trends", icon: TrendingUp },
  { href: "/user#profile", label: "Profile", icon: UserIcon },
  { href: "/user/upload-trip", label: "Upload Trip", icon: Upload },
];

export default function UserLayout({ children }: { children: React.ReactNode }) {
  const [email, setEmail] = useState<string>("");
  const router = useRouter();

  useEffect(() => {
    const u = getStoredUser();
    if (!u) {
      router.replace("/login");
      return;
    }
    setEmail(u.email);
  }, [router]);

  if (!email) return null;
  return (
    <div className="min-h-screen bg-background">
      <AppHeader
        title="SkyLedge"
        roleLabel="Driver"
        email={email}
        onLogout={() => {
          clearStoredUser();
          window.location.href = "/login";
        }}
        sidebarItems={userNav}
        sidebarTitle="SkyLedge — Driver"
        sidebarSubtitle="Predictive Maintenance"
      />

      <div className="pt-14 flex">
        {/* Desktop Sidebar */}
        <div className="hidden md:block sticky top-14 h-[calc(100vh-3.5rem)]">
          <Sidebar title="SkyLedge — Driver" subtitle="Predictive Maintenance" items={userNav} />
        </div>

        {/* NOTE: no md:ml-64 here — flex handles spacing */}
        <main className="flex-1 p-6">{children}</main>
      </div>
    </div>
  );
}
