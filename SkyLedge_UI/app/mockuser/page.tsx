"use client"

import { useState } from "react"
import { DashboardHeader } from "@/components/layout/dashboard-header"
import { DashboardSidebar } from "@/components/layout/dashboard-sidebar"

// Import tab components (to be created)
import { TripSummaryTab } from "@/components/user/trip-summary-tab"
import { TrendsTab } from "@/components/user/trends-tab"
import { ProfileTab } from "@/components/user/profile-tab"

export default function UserDashboard() {
  const [activeTab, setActiveTab] = useState("summary")

  return (
    <div className="min-h-screen bg-background">
      <DashboardHeader userRole="user" userEmail="driver@example.com" />

      <div className="flex">
        <DashboardSidebar userRole="user" activeTab={activeTab} onTabChange={setActiveTab} />

        <main className="flex-1 p-6 bg-background">
          <div className="space-y-6">
            <div>
              <h2 className="text-3xl font-bold text-balance">Driver Dashboard</h2>
              <p className="text-muted-foreground">
                View your trip summaries, driving insights, and performance trends
              </p>
            </div>

            {activeTab === "summary" && <TripSummaryTab />}
          </div>
        </main>
      </div>
    </div>
  )
}
