"use client"

import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Car, LogOut } from "lucide-react"
import { useRouter } from "next/navigation"

interface DashboardHeaderProps {
  userRole: "developer" | "user"
  userEmail: string
}

export function DashboardHeader({ userRole, userEmail }: DashboardHeaderProps) {
  const router = useRouter()

  const handleLogout = () => {
    localStorage.removeItem("isAuthenticated")
    localStorage.removeItem("userRole")
    localStorage.removeItem("userEmail")
    router.push("/login")
  }

  return (
    <header className="border-b border-border bg-card/50 backdrop-blur-sm">
      <div className="flex h-16 items-center px-6">
        <div className="flex items-center gap-3">
          <div className="p-1.5 bg-primary rounded-lg">
            <Car className="h-5 w-5 text-primary-foreground" />
          </div>
          <h1 className="text-xl font-bold text-foreground">SkyLedge</h1>
        </div>
        <div className="ml-auto flex items-center gap-4">
          <Badge
            variant={userRole === "developer" ? "default" : "secondary"}
            className={
              userRole === "developer" ? "bg-primary text-primary-foreground" : "bg-accent text-accent-foreground"
            }
          >
            {userRole === "developer" ? "Developer" : "Driver"}
          </Badge>
          <span className="text-sm text-muted-foreground font-medium">{userEmail}</span>
          <Button variant="ghost" size="sm" onClick={handleLogout} className="hover:bg-muted">
            <LogOut className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </header>
  )
}
