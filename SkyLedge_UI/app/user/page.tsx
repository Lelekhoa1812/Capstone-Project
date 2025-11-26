"use client";

import { useEffect, useMemo, useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { MapPin, Award, Fuel, Target, LineChart, Leaf, Zap, User as UserIcon } from "lucide-react";
import { mockUserTrips } from "@/lib/data/mockUserTrips";
import { mockTrendData } from "@/lib/data/mockTrendData";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { Label } from "@/components/ui/label";

const scoreColor = (n: number) => (n >= 90 ? "text-chart-1" : n >= 80 ? "text-chart-2" : n >= 70 ? "text-chart-3" : "text-chart-4");
const scoreBadge = (n: number) =>
  n >= 90 ? { variant: "default" as const, label: "Excellent" } :
  n >= 80 ? { variant: "secondary" as const, label: "Good" } :
  n >= 70 ? { variant: "outline" as const, label: "Fair" } :
            { variant: "destructive" as const, label: "Needs Improvement" };

export default function UserDashboard() {
  const averageScore = useMemo(() => Math.round(mockUserTrips.reduce((s, t) => s + t.drivingScore, 0) / mockUserTrips.length), []);
  const averageEfficiency = useMemo(() => (mockUserTrips.reduce((s, t) => s + t.fuelEfficiency, 0) / mockUserTrips.length).toFixed(1), []);
  const totalFuelCost = useMemo(() => mockUserTrips.reduce((s, t) => s + t.estimatedFuelCost, 0).toFixed(2), []);

  const [tab, setTab] = useState<"summary" | "trends" | "profile">("summary");

  // Sync tab with URL hash for sidebar navigation and deep-linking
  useEffect(() => {
    const applyFromHash = () => {
      const h = (typeof window !== "undefined" ? window.location.hash.replace("#", "") : "") as typeof tab;
      if (h === "summary" || h === "trends" || h === "profile") {
        setTab(h);
      } else {
        // default to summary when visiting /user without a hash
        setTab("summary");
        if (typeof window !== "undefined") {
          history.replaceState(null, "", `#summary`);
        }
      }
    };
    applyFromHash();
    window.addEventListener("hashchange", applyFromHash);
    return () => window.removeEventListener("hashchange", applyFromHash);
  }, []);

  const setTabAndHash = (next: typeof tab) => {
    setTab(next);
    if (typeof window !== "undefined") {
      history.replaceState(null, "", `#${next}`);
    }
  };

  return (
    <div className="p-6 space-y-6">
      <header>
        <h2 className="text-3xl font-bold">Driver Dashboard</h2>
        <p className="text-muted-foreground">View your trip summaries, driving insights, and performance trends</p>
      </header>

      {/* simple tabs */}
      <div className="flex gap-2">
        <Badge onClick={() => setTabAndHash("summary")} variant={tab === "summary" ? "default" : "secondary"} className="cursor-pointer">Trip Summary</Badge>
        <Badge onClick={() => setTabAndHash("trends")} variant={tab === "trends" ? "default" : "secondary"} className="cursor-pointer">Trends</Badge>
        <Badge onClick={() => setTabAndHash("profile")} variant={tab === "profile" ? "default" : "secondary"} className="cursor-pointer">Profile</Badge>
      </div>

      {tab === "summary" && (
        <div className="space-y-6">
          <div className="grid gap-6 md:grid-cols-3">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Overall Driving Score</CardTitle>
                <Award className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{averageScore}</div>
                <div className="flex items-center gap-2 mt-2">
                  <Badge {...scoreBadge(averageScore)}>{scoreBadge(averageScore).label}</Badge>
                </div>
                <Progress value={averageScore} className="mt-3" />
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Fuel Efficiency</CardTitle>
                <Fuel className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{averageEfficiency} MPG</div>
                <p className="text-xs text-muted-foreground mt-2">+2.3 MPG from last month</p>
                <div className="flex items-center gap-1 mt-2">
                  <Leaf className="h-3 w-3 text-chart-1" />
                  <span className="text-xs text-chart-1">Eco-friendly driving</span>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Fuel Cost This Week</CardTitle>
                <Target className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">${totalFuelCost}</div>
                <p className="text-xs text-muted-foreground mt-2">-$3.20 vs last week</p>
                <div className="flex items-center gap-1 mt-2">
                  <LineChart className="h-3 w-3 text-chart-1" />
                  <span className="text-xs text-chart-1">Savings improved</span>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Recent Trips */}
          <Card>
            <CardHeader><CardTitle>Recent Trips</CardTitle></CardHeader>
            <CardContent>
              <div className="space-y-4">
                {mockUserTrips.map((trip) => (
                  <div key={trip.id} className="border rounded-lg p-4">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-3">
                        <div className="flex items-center gap-2">
                          <MapPin className="h-4 w-4 text-muted-foreground" />
                          <span className="font-medium">{trip.startLocation} → {trip.endLocation}</span>
                        </div>
                        <Badge variant="outline">{trip.date}</Badge>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className={`text-lg font-bold ${scoreColor(trip.drivingScore)}`}>{trip.drivingScore}</span>
                        <Badge {...scoreBadge(trip.drivingScore)}>{scoreBadge(trip.drivingScore).label}</Badge>
                      </div>
                    </div>

                    <div className="grid grid-cols-4 gap-4 mb-4">
                      <div className="text-center">
                        <div className="text-sm text-muted-foreground">Duration</div>
                        <div className="font-medium">{trip.duration}</div>
                      </div>
                      <div className="text-center">
                        <div className="text-sm text-muted-foreground">Distance</div>
                        <div className="font-medium">{trip.distance}</div>
                      </div>
                      <div className="text-center">
                        <div className="text-sm text-muted-foreground">Efficiency</div>
                        <div className="font-medium">{trip.fuelEfficiency} MPG</div>
                      </div>
                      <div className="text-center">
                        <div className="text-sm text-muted-foreground">Fuel Cost</div>
                        <div className="font-medium">${trip.estimatedFuelCost.toFixed(2)}</div>
                      </div>
                    </div>

                    <div className="mb-4">
                      <div className="text-sm font-medium mb-2">Driving Behavior</div>
                      <div className="flex gap-2 mb-2">
                        <div className="flex-1">
                          <div className="flex justify-between text-xs mb-1"><span>Idle</span><span>{trip.breakdown.idle}%</span></div>
                          <Progress value={trip.breakdown.idle} className="h-2" />
                        </div>
                        <div className="flex-1">
                          <div className="flex justify-between text-xs mb-1"><span>Passive</span><span>{trip.breakdown.passive}%</span></div>
                          <Progress value={trip.breakdown.passive} className="h-2" />
                        </div>
                        <div className="flex-1">
                          <div className="flex justify-between text-xs mb-1"><span>Aggressive</span><span>{trip.breakdown.aggressive}%</span></div>
                          <Progress value={trip.breakdown.aggressive} className="h-2" />
                        </div>
                      </div>
                    </div>

                    <div className="bg-muted rounded-lg p-3">
                      <div className="text-sm font-medium mb-2 flex items-center gap-2">
                        <Zap className="h-4 w-4 text-primary" />Driving Tips
                      </div>
                      <ul className="text-sm text-muted-foreground space-y-1">
                        {trip.tips.map((tip, i) => (
                          <li key={i} className="flex items-start gap-2">
                            <span className="text-primary">•</span><span>{tip}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {tab === "trends" && (
        <div className="space-y-6">
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader><CardTitle>Driving Score Trend</CardTitle></CardHeader>
              <CardContent className="space-y-4">
                {mockTrendData.map((d) => (
                  <div key={d.week} className="flex items-center justify-between">
                    <span className="text-sm font-medium">{d.week}</span>
                    <div className="flex items-center gap-3">
                      <div className="w-32 bg-muted rounded-full h-2"><div className="bg-primary h-2 rounded-full" style={{ width: `${d.score}%` }} /></div>
                      <span className={`font-bold ${scoreColor(d.score)}`}>{d.score}</span>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>

            <Card>
              <CardHeader><CardTitle>Fuel Efficiency Trend</CardTitle></CardHeader>
              <CardContent className="space-y-4">
                {mockTrendData.map((d) => (
                  <div key={d.week} className="flex items-center justify-between">
                    <span className="text-sm font-medium">{d.week}</span>
                    <div className="flex items-center gap-3">
                      <div className="w-32 bg-muted rounded-full h-2"><div className="bg-chart-1 h-2 rounded-full" style={{ width: `${(d.efficiency / 35) * 100}%` }} /></div>
                      <span className="font-bold text-chart-1">{d.efficiency} MPG</span>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader><CardTitle>Weekly Fuel Cost Analysis</CardTitle></CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-4">
                {mockTrendData.map((d, i) => (
                  <div key={d.week} className="text-center p-4 bg-muted rounded-lg">
                    <div className="text-sm text-muted-foreground">{d.week}</div>
                    <div className="text-xl font-bold text-chart-3">${d.cost.toFixed(2)}</div>
                    {i > 0 && (
                      <div className="text-xs mt-1">
                        {d.cost < mockTrendData[i - 1].cost
                          ? <span className="text-chart-1">↓ Saved ${(mockTrendData[i - 1].cost - d.cost).toFixed(2)}</span>
                          : <span className="text-chart-4">↑ +${(d.cost - mockTrendData[i - 1].cost).toFixed(2)}</span>}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {tab === "profile" && (
        <div className="space-y-6">
          <Card>
            <CardHeader><CardTitle>Driver Profile</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center gap-4">
                <div className="w-16 h-16 bg-primary rounded-full flex items-center justify-center">
                  <UserIcon className="h-8 w-8 text-primary-foreground" />
                </div>
                <div>
                  <h3 className="text-lg font-medium">John Driver</h3>
                  <p className="text-muted-foreground">Member since January 2024</p>
                </div>
              </div>
              <Separator />
              <div className="grid gap-4 md:grid-cols-2">
                <div><Label>Email</Label><Input value="driver@company.com" disabled /></div>
                <div><Label>Vehicle</Label><Input value="2023 Toyota Camry" disabled /></div>
                <div><Label>Total Trips</Label><Input value="47 trips" disabled /></div>
                <div><Label>Total Distance</Label><Input value="1,247 miles" disabled /></div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
