import { UserTrip } from '@/lib/types'

export const mockUserTrips: UserTrip[] = [
  {
    id: "trip_user_001",
    date: "2024-01-16",
    startTime: "08:30",
    endTime: "09:15",
    duration: "45 min",
    distance: "23.5 miles",
    startLocation: "Home",
    endLocation: "Office",
    drivingScore: 85,
    fuelEfficiency: 28.5,
    estimatedFuelCost: 4.2,
    breakdown: {
      idle: 15,
      passive: 70,
      aggressive: 15,
    },
    tips: [
      "Great job maintaining steady speeds on the highway!",
      "Consider reducing idle time at traffic lights by 2 minutes to save fuel.",
    ],
  },
  {
    id: "trip_user_002",
    date: "2024-01-15",
    startTime: "17:45",
    endTime: "18:30",
    duration: "45 min",
    distance: "23.5 miles",
    startLocation: "Office",
    endLocation: "Home",
    drivingScore: 78,
    fuelEfficiency: 26.2,
    estimatedFuelCost: 4.6,
    breakdown: {
      idle: 20,
      passive: 60,
      aggressive: 20,
    },
    tips: [
      "Try to anticipate traffic better to reduce sudden braking.",
      "Your acceleration was smooth - keep it up!"
    ],
  },
  {
    id: "trip_user_003",
    date: "2024-01-15",
    startTime: "12:15",
    endTime: "12:45",
    duration: "30 min",
    distance: "8.2 miles",
    startLocation: "Office",
    endLocation: "Restaurant",
    drivingScore: 92,
    fuelEfficiency: 31.8,
    estimatedFuelCost: 2.1,
    breakdown: {
      idle: 10,
      passive: 85,
      aggressive: 5,
    },
    tips: [
      "Excellent driving! Your fuel efficiency was outstanding.",
      "Perfect example of eco-friendly driving habits.",
    ],
  },
]