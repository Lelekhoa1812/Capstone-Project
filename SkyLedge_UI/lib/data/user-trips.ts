export const mockUserTrips = [
  {
    id: "trip_user_001",
    date: "2024-01-16",
    startTime: "08:30",
    endTime: "09:15",
    duration: "45 min",
    distance: "23.5 miles",
    route: "Home → Office",
    efficiencyScore: 85,
    fuelUsed: "0.82 gal",
    fuelCost: 2.87,
    baselineFuel: "1.05 gal",
    fuelSaved: "0.23 gal",
    moneySaved: 0.81,
    tips: [
      "Excellent! You saved $0.81 by maintaining steady speeds.",
      "Try coasting to stops instead of braking hard to improve further.",
    ],
  },
  {
    id: "trip_user_002",
    date: "2024-01-15",
    startTime: "17:45",
    endTime: "18:30",
    duration: "45 min",
    distance: "23.5 miles",
    route: "Office → Home",
    efficiencyScore: 72,
    fuelUsed: "0.95 gal",
    fuelCost: 3.33,
    baselineFuel: "1.05 gal",
    fuelSaved: "0.10 gal",
    moneySaved: 0.35,
    tips: [
      "You're using 10% more fuel than optimal due to rapid acceleration.",
      "Anticipate traffic lights to reduce stop-and-go driving.",
    ],
  },
  {
    id: "trip_user_003",
    date: "2024-01-15",
    startTime: "12:15",
    endTime: "12:45",
    duration: "30 min",
    distance: "8.2 miles",
    route: "Office → Restaurant",
    efficiencyScore: 92,
    fuelUsed: "0.26 gal",
    fuelCost: 0.91,
    baselineFuel: "0.35 gal",
    fuelSaved: "0.09 gal",
    moneySaved: 0.32,
    tips: [
      "Outstanding! Your smooth driving saved $0.32 on this trip.",
      "Keep up this efficiency - you're in the top 10% of drivers!",
    ],
  },
  {
    id: "trip_user_004",
    date: "2024-01-14",
    startTime: "08:25",
    endTime: "09:10",
    duration: "45 min",
    distance: "23.5 miles",
    route: "Home → Office",
    efficiencyScore: 78,
    fuelUsed: "0.89 gal",
    fuelCost: 3.12,
    baselineFuel: "1.05 gal",
    fuelSaved: "0.16 gal",
    moneySaved: 0.56,
    tips: [
      "Good job! Reducing idle time by 3 minutes could save an extra $0.20.",
      "Your highway driving is efficient - focus on city driving improvements.",
    ],
  },
  {
    id: "trip_user_005",
    date: "2024-01-13",
    startTime: "09:00",
    endTime: "09:50",
    duration: "50 min",
    distance: "28.3 miles",
    route: "Home → Client Site",
    efficiencyScore: 88,
    fuelUsed: "0.92 gal",
    fuelCost: 3.22,
    baselineFuel: "1.15 gal",
    fuelSaved: "0.23 gal",
    moneySaved: 0.81,
    tips: [
      "Great efficiency! You're saving about $16/month at this rate.",
      "Maintain tire pressure at recommended levels for even better results.",
    ],
  },
  {
    id: "trip_user_006",
    date: "2024-01-12",
    startTime: "18:15",
    endTime: "19:00",
    duration: "45 min",
    distance: "23.5 miles",
    route: "Office → Home",
    efficiencyScore: 65,
    fuelUsed: "1.08 gal",
    fuelCost: 3.78,
    baselineFuel: "1.05 gal",
    fuelSaved: "-0.03 gal",
    moneySaved: -0.11,
    tips: [
      "You used $0.11 more fuel than average due to aggressive acceleration.",
      "Try accelerating more gradually - aim for 5-10 seconds to reach speed.",
      "Reducing speed by 5 mph on highways can improve efficiency by 7%.",
    ],
  },
]

// Calculate overall statistics
export const calculateUserStats = () => {
  const totalTrips = mockUserTrips.length
  const avgEfficiency = Math.round(mockUserTrips.reduce((sum, trip) => sum + trip.efficiencyScore, 0) / totalTrips)
  const totalFuelSaved = mockUserTrips.reduce((sum, trip) => sum + Number.parseFloat(trip.fuelSaved), 0).toFixed(2)
  const totalMoneySaved = mockUserTrips.reduce((sum, trip) => sum + trip.moneySaved, 0).toFixed(2)
  const totalFuelUsed = mockUserTrips.reduce((sum, trip) => sum + Number.parseFloat(trip.fuelUsed), 0).toFixed(2)
  const totalCost = mockUserTrips.reduce((sum, trip) => sum + trip.fuelCost, 0).toFixed(2)

  return {
    totalTrips,
    avgEfficiency,
    totalFuelSaved,
    totalMoneySaved,
    totalFuelUsed,
    totalCost,
  }
}

export const mockTrendData = [
  { week: "Week 1", score: 82, efficiency: 27.5, cost: 45.2 },
  { week: "Week 2", score: 85, efficiency: 28.1, cost: 43.8 },
  { week: "Week 3", score: 88, efficiency: 29.2, cost: 41.5 },
  { week: "Week 4", score: 85, efficiency: 28.5, cost: 42.9 },
]
