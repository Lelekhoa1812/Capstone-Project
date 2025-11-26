"use client";

import Link from "next/link";
import { Car } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function LandingPage() {
  return (
    <main className="min-h-screen flex flex-col items-center justify-center gap-6 p-6">
      <div className="flex items-center gap-3">
        <div className="p-2 bg-primary rounded-lg"><Car className="h-6 w-6 text-primary-foreground" /></div>
        <h1 className="text-3xl font-bold">SkyLedge</h1>
      </div>
      <p className="text-muted-foreground text-center max-w-md">
        Upload OBD-II data, label segments, and view driver insights.
      </p>
      <Button asChild><Link href="/login">Sign in</Link></Button>
    </main>
  );
}
