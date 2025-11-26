"use client";

import { useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import Sidebar from "./Sidebar";
import { Menu, LogOut } from "lucide-react";
import Image from "next/image";
import ThemeToggle from "@/components/ThemeToggle";

export default function AppHeader({
  title = "SkyLedge",
  roleLabel,
  email,
  onLogout,
  sidebarItems,
  sidebarTitle,
  sidebarSubtitle,
}: {
  title?: string;
  roleLabel?: string;
  email?: string;
  onLogout?: () => void;
  sidebarItems: React.ComponentProps<typeof Sidebar>["items"];
  sidebarTitle: string;
  sidebarSubtitle?: string;
}) {
  const [open, setOpen] = useState(false);

  return (
    <header className="fixed top-0 inset-x-0 h-14 border-b border-border bg-card/70 backdrop-blur supports-[backdrop-filter]:bg-card/60 z-50">
      <div className="flex h-full items-center px-4">
        {/* Mobile open drawer */}
        <Button variant="ghost" size="icon" className="md:hidden mr-2" onClick={() => setOpen(true)}>
          <Menu className="h-5 w-5" />
        </Button>

        <Dialog open={open} onOpenChange={setOpen}>
          <DialogContent className="p-0 max-w-xs">
            <DialogHeader className="p-4 border-b">
              <DialogTitle>{sidebarTitle}</DialogTitle>
            </DialogHeader>
            <Sidebar
              title={sidebarTitle}
              subtitle={sidebarSubtitle}
              items={sidebarItems}
              onNavigate={() => setOpen(false)}
              className="border-0"
            />
          </DialogContent>
        </Dialog>

        <div className="flex items-center gap-2">
          <Image src="/skyledge.jpeg" alt="SkyLedge logo" width={20} height={20} priority />
          <div className="text-sm font-semibold">{title}</div>
        </div>

        <div className="ml-auto flex items-center gap-3">
          <ThemeToggle />
          {roleLabel ? <span className="text-xs px-2 py-1 rounded bg-primary text-primary-foreground">{roleLabel}</span> : null}
          {email ? <span className="text-sm text-muted-foreground">{email}</span> : null}
          {onLogout ? (
            <Button variant="ghost" size="sm" onClick={onLogout}>
              <LogOut className="h-4 w-4" />
            </Button>
          ) : null}
        </div>
      </div>
    </header>
  );
}
