"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { getStoredUser, setStoredUser } from "@/lib/auth";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Car } from "lucide-react";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const router = useRouter();

  useEffect(() => {
    const u = getStoredUser();
    if (u) {
      router.replace(u.role === "developer" ? "/developer" : "/user");
    }
  }, [router]);

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!email || !password) return;
    // simple demo routing based on email like original file
    const role = email.includes("dev") ? "developer" : "user";
    setStoredUser({ email, role, createdAt: Date.now() });
    router.push(role === "developer" ? "/developer" : "/user");
  };

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <Card className="w-full max-w-md shadow-lg border-0 bg-card">
        <CardHeader className="text-center pb-8">
          <div className="flex items-center justify-center gap-2 mb-4">
            <div className="p-2 bg-primary rounded-lg">
              <Car className="h-6 w-6 text-primary-foreground" />
            </div>
            <CardTitle className="text-2xl font-bold text-foreground">SkyLedge</CardTitle>
          </div>
          <p className="text-muted-foreground text-sm">Predictive Maintenance Dashboard</p>
        </CardHeader>
        <CardContent className="pt-0">
          <form onSubmit={onSubmit} className="space-y-6">
            <div className="space-y-2">
              <Label htmlFor="email">Email</Label>
              <Input id="email" type="email" placeholder="developer@company.com"
                     value={email} onChange={(e) => setEmail(e.target.value)} required />
            </div>
            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <Input id="password" type="password" placeholder="Enter your password"
                     value={password} onChange={(e) => setPassword(e.target.value)} required />
            </div>
            <Button type="submit" className="w-full">Sign In</Button>
            <div className="text-xs text-muted-foreground text-center bg-muted/50 p-3 rounded-lg">
              <strong>Demo:</strong> Use "developer@company.com" for dev view
            </div>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
