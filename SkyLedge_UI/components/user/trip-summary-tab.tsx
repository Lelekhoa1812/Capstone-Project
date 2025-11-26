"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Calendar, Clock, Fuel, TrendingUp, TrendingDown, DollarSign, Lightbulb, Award, MapPin } from "lucide-react"
import { mockUserTrips, calculateUserStats } from "@/lib/data/user-trips"

export function TripSummaryTab() {
  const stats = calculateUserStats()

  const getEfficiencyColor = (score: number) => {
    if (score >= 85) return "text-green-600"
    if (score >= 70) return "text-blue-600"
    return "text-orange-600"
  }

  const getEfficiencyBadge = (score: number) => {
    if (score >= 85) return { variant: "default" as const, label: "Excellent" }
    if (score >= 70) return { variant: "secondary" as const, label: "Good" }
    return { variant: "destructive" as const, label: "Needs Work" }
  }

  return (
    <div className="space-y-6">
      {/* Overall Statistics */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Efficiency</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.avgEfficiency}/100</div>
            <Progress value={stats.avgEfficiency} className="mt-2" />
            <p className="text-xs text-muted-foreground mt-2">Across {stats.totalTrips} trips</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Fuel Saved</CardTitle>
            <Fuel className="h-4 w-4 text-green-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">{stats.totalFuelSaved} gal</div>
            <p className="text-xs text-muted-foreground mt-2">vs. average driver</p>
            <p className="text-xs font-medium text-green-600 mt-1">↓ 18% reduction</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Money Saved</CardTitle>
            <DollarSign className="h-4 w-4 text-green-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">${stats.totalMoneySaved}</div>
            <p className="text-xs text-muted-foreground mt-2">This week</p>
            <p className="text-xs font-medium text-green-600 mt-1">~$12.80/month projected</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Fuel Cost</CardTitle>
            <DollarSign className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">${stats.totalCost}</div>
            <p className="text-xs text-muted-foreground mt-2">{stats.totalFuelUsed} gal used</p>
            <p className="text-xs font-medium mt-1">$3.50/gal avg</p>
          </CardContent>
        </Card>
      </div>

      {/* Recent Trips */}
      <div>
        <h3 className="text-lg font-semibold mb-4">Recent Trips</h3>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {mockUserTrips.map((trip) => {
            const badge = getEfficiencyBadge(trip.efficiencyScore)
            const isSaving = trip.moneySaved >= 0

            return (
              <Card key={trip.id} className="hover:shadow-md transition-shadow">
                <CardHeader className="pb-3">
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex-1">
                      
                      <CardDescription className="flex items-center gap-2 mt-1">
                        <Calendar className="h-3 w-3" />
                        {trip.date} • {trip.startTime}
                      </CardDescription>
                    </div>
                    <Badge variant={badge.variant} className="shrink-0">
                      {trip.efficiencyScore}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent className="space-y-3">
                  {/* Efficiency Score */}
                  <div className="space-y-1">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">Driving Efficiency</span>
                      <span className={`font-semibold ${getEfficiencyColor(trip.efficiencyScore)}`}>{badge.label}</span>
                    </div>
                    <Progress value={trip.efficiencyScore} className="h-2" />
                  </div>

                  {/* Trip Details */}
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div className="flex items-center gap-1">
                      <Clock className="h-3 w-3 text-muted-foreground" />
                      <span className="text-muted-foreground">Duration:</span>
                    </div>
                    <span className="font-medium text-right">{trip.duration}</span>

                    <div className="flex items-center gap-1">
                      <MapPin className="h-3 w-3 text-muted-foreground" />
                      <span className="text-muted-foreground">Distance:</span>
                    </div>
                    <span className="font-medium text-right">{trip.distance}</span>
                  </div>

                  {/* Fuel & Cost */}
                  <div className="border-t pt-3 space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="flex items-center gap-1 text-muted-foreground">
                        <Fuel className="h-3 w-3" />
                        Fuel Used
                      </span>
                      <span className="font-medium">{trip.fuelUsed}</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="flex items-center gap-1 text-muted-foreground">
                        <DollarSign className="h-3 w-3" />
                        Cost
                      </span>
                      <span className="font-medium">${trip.fuelCost.toFixed(2)}</span>
                    </div>
                    
                  </div>

                  {/* Tips */}
                  {trip.tips && trip.tips.length > 0 && (
                    <div className="border-t pt-3 space-y-2">
                      {trip.tips.map((tip, index) => (
                        <div key={index} className="flex gap-2 text-xs bg-blue-50 dark:bg-blue-950/20 p-2 rounded">
                          <Lightbulb className="h-3 w-3 text-blue-600 shrink-0 mt-0.5" />
                          <span className="text-blue-900 dark:text-blue-100">{tip}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>
            )
          })}
        </div>
      </div>

      {/* Improvement Tips Section */}
      <Card className="border-blue-200 bg-blue-50/50 dark:bg-blue-950/20 dark:border-blue-900">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Award className="h-5 w-5 text-blue-600" />
            How to Improve Your Efficiency Score
          </CardTitle>
          <CardDescription>Follow these tips to save more fuel and money</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid gap-3 md:grid-cols-2">
            <div className="flex gap-3 p-3 bg-white dark:bg-gray-900 rounded-lg">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-blue-100 dark:bg-blue-900">
                <span className="text-sm font-bold text-blue-600">1</span>
              </div>
              <div>
                <h4 className="font-semibold text-sm">Smooth Acceleration</h4>
                <p className="text-xs text-muted-foreground">Take 5-10 seconds to reach cruising speed</p>
              </div>
            </div>

            <div className="flex gap-3 p-3 bg-white dark:bg-gray-900 rounded-lg">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-blue-100 dark:bg-blue-900">
                <span className="text-sm font-bold text-blue-600">2</span>
              </div>
              <div>
                <h4 className="font-semibold text-sm">Anticipate Traffic</h4>
                <p className="text-xs text-muted-foreground">Coast to stops instead of hard braking</p>
              </div>
            </div>

            <div className="flex gap-3 p-3 bg-white dark:bg-gray-900 rounded-lg">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-blue-100 dark:bg-blue-900">
                <span className="text-sm font-bold text-blue-600">3</span>
              </div>
              <div>
                <h4 className="font-semibold text-sm">Maintain Speed</h4>
                <p className="text-xs text-muted-foreground">Use cruise control on highways when possible</p>
              </div>
            </div>

            <div className="flex gap-3 p-3 bg-white dark:bg-gray-900 rounded-lg">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-blue-100 dark:bg-blue-900">
                <span className="text-sm font-bold text-blue-600">4</span>
              </div>
              <div>
                <h4 className="font-semibold text-sm">Reduce Idle Time</h4>
                <p className="text-xs text-muted-foreground">Turn off engine if stopped for over 30 seconds</p>
              </div>
            </div>
          </div>

          <div className="mt-4 p-3 bg-green-50 dark:bg-green-950/20 border border-green-200 dark:border-green-900 rounded-lg">
            <p className="text-sm font-medium text-green-900 dark:text-green-100">
              💰 Potential Savings: Improving your efficiency by 10 points could save you an extra $8-12 per month!
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
