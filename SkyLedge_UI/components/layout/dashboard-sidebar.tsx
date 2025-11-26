"use client"

import { Button } from "@/components/ui/button"
import { Upload, BarChart3, Baseline as Timeline, Database, Brain, User } from "lucide-react"

interface DashboardSidebarProps {
  userRole: "developer" | "user"
  activeTab: string
  onTabChange: (tab: string) => void
}

export function DashboardSidebar({ userRole, activeTab, onTabChange }: DashboardSidebarProps) {
  const developerTabs = [
    { id: "upload", label: "Upload/Buffer", icon: Upload },
    { id: "labeling", label: "Manual Labeling", icon: Timeline },
    { id: "dataset", label: "Labeled Dataset", icon: Database },
    { id: "reinforcement", label: "Reinforcement", icon: Brain },
  ]

  const userTabs = [
    { id: "summary", label: "Trip Summary", icon: BarChart3 },
    { id: "trends", label: "Trends", icon: Timeline },
    { id: "profile", label: "Profile", icon: User },
  ]

  const tabs = userRole === "developer" ? developerTabs : userTabs

  return (
    <aside className="w-64 border-r border-border bg-card/30 min-h-[calc(100vh-4rem)]">
      <nav className="p-4 space-y-2">
        <div className="px-3 py-2 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
          {userRole === "developer" ? "Developer Tools" : "Dashboard"}
        </div>
        {tabs.map((tab) => {
          const Icon = tab.icon
          return (
            <Button
              key={tab.id}
              variant={activeTab === tab.id ? "default" : "ghost"}
              className={`w-full justify-start gap-3 h-10 ${
                activeTab === tab.id
                  ? "bg-primary text-primary-foreground shadow-sm"
                  : "hover:bg-muted text-muted-foreground hover:text-foreground"
              }`}
              onClick={() => onTabChange(tab.id)}
            >
              <Icon className="h-4 w-4" />
              {tab.label}
            </Button>
          )
        })}
      </nav>
    </aside>
  )
}
