
weights:
  RPM: 0.25              
  THROTTLE_POS: 0.25    
  ENGINE_LOAD: 0.25     
  MAF: 0.25              


# Spike Detection Thresholds
spike_thresholds:
  moderate_spike: 65     
  extreme_spike: 85      
  spike_percentage_threshold: 3   
  extreme_percentage_threshold: 1 

# Penalty Multipliers
penalty_multipliers:
  p95_multiplier: 0.3    
  p99_multiplier: 0.5    
  spike_freq_multiplier: 2.0    
  extreme_freq_multiplier: 3.0 

# Aggregate Score Calculation
aggregate_weights:
  mean_weight: 0.7       
  p75_weight: 0.3        

style_categories:
  very_calm: [0, 20]
  calm: [20, 40]
  moderate: [40, 55]
  aggressive: [55, 70]
  very_aggressive: [70, 100]

bounds:
  file: "obd_bounds.json"
  auto_update: true      # Automatically update bounds with new data
  

theoretical_maxes:
  RPM: 6000
  THROTTLE_POS: 100
  ENGINE_LOAD: 100
  MAF: 300
  SPEED: 250
  INTAKE_PRESSURE: 250

theoretical_mins:
  RPM: 0
  THROTTLE_POS: 0
  ENGINE_LOAD: 0
  MAF: 0
  SPEED: 0
  INTAKE_PRESSURE: 0

output:
  save_scored_csv: true
  visualization: true
  verbose: true