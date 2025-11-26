"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { User, Car, Award, Settings, Download } from "lucide-react"

export function ProfileTab() {
  return (
    <div className="space-y-6">
      <div className="grid gap-6 md:grid-cols-2">
        {/* Profile Information */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <User className="h-5 w-5" />
              Profile Information
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center gap-4">
              <Avatar className="h-16 w-16">
                <AvatarFallback>JD</AvatarFallback>
              </Avatar>
              <div>
                <h3 className="font-semibold">John Driver</h3>
                <p className="text-sm text-muted-foreground">john.driver@example.com</p>
                <Badge variant="secondary" className="mt-1">
                  Premium Member
                </Badge>
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm">Member since:</span>
                <span className="text-sm font-medium">January 2024</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm">Total trips:</span>
                <span className="text-sm font-medium">247</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm">Miles driven:</span>
                <span className="text-sm font-medium">12,450</span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Vehicle Information */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Car className="h-5 w-5" />
              Vehicle Information
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-3">
              <div>
                <h4 className="font-medium">Primary Vehicle</h4>
                <p className="text-sm text-muted-foreground">2022 Toyota Camry</p>
              </div>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm">VIN:</span>
                  <span className="text-sm font-mono">****789012</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">Engine:</span>
                  <span className="text-sm">2.5L 4-Cylinder</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">Fuel Type:</span>
                  <span className="text-sm">Regular Gasoline</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">OBD-II Device:</span>
                  <Badge variant="outline" className="text-xs">
                    Connected
                  </Badge>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Achievements */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Award className="h-5 w-5" />
              Achievements
            </CardTitle>
            <CardDescription>Your driving milestones and badges</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-3">
              <div className="flex items-center gap-2 p-2 rounded-lg bg-muted/50">
                <div className="w-8 h-8 rounded-full bg-green-100 flex items-center justify-center">🌱</div>
                <div>
                  <p className="text-sm font-medium">Eco Driver</p>
                  <p className="text-xs text-muted-foreground">50+ efficient trips</p>
                </div>
              </div>
              <div className="flex items-center gap-2 p-2 rounded-lg bg-muted/50">
                <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center">🎯</div>
                <div>
                  <p className="text-sm font-medium">Consistent</p>
                  <p className="text-xs text-muted-foreground">30 days streak</p>
                </div>
              </div>
              <div className="flex items-center gap-2 p-2 rounded-lg bg-muted/50">
                <div className="w-8 h-8 rounded-full bg-purple-100 flex items-center justify-center">📊</div>
                <div>
                  <p className="text-sm font-medium">Data Lover</p>
                  <p className="text-xs text-muted-foreground">100+ trips logged</p>
                </div>
              </div>
              <div className="flex items-center gap-2 p-2 rounded-lg bg-muted/50">
                <div className="w-8 h-8 rounded-full bg-yellow-100 flex items-center justify-center">⭐</div>
                <div>
                  <p className="text-sm font-medium">Top Performer</p>
                  <p className="text-xs text-muted-foreground">90%+ avg score</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Settings & Actions */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Settings className="h-5 w-5" />
              Settings & Actions
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <Button variant="outline" className="w-full justify-start bg-transparent">
              <Download className="h-4 w-4 mr-2" />
              Export Trip Data
            </Button>
            <Button variant="outline" className="w-full justify-start bg-transparent">
              <Settings className="h-4 w-4 mr-2" />
              Notification Preferences
            </Button>
            <Button variant="outline" className="w-full justify-start bg-transparent">
              <Car className="h-4 w-4 mr-2" />
              Manage Vehicles
            </Button>
            <Button variant="outline" className="w-full justify-start bg-transparent">
              <User className="h-4 w-4 mr-2" />
              Account Settings
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
