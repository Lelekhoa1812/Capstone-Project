import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime


class DrivingAggressivenessScorer:
    def __init__(self, bounds_file: str = 'obd_bounds.json', weights: Dict = None):
        self.bounds_file = Path(bounds_file)
        self.weights = weights if weights else self.weights.copy()
        self.bounds = self._load_bounds()
        
        weight_sum = sum(self.weights.values())
        if not np.isclose(weight_sum, 1.0):
            print(f"Warning: Weights sum to {weight_sum:.3f}, normalizing to 1.0")
            self.weights = {k: v/weight_sum for k, v in self.weights.items()}
    
    def _load_bounds(self) -> Dict:
        if self.bounds_file.exists():
            with open(self.bounds_file, 'r') as f:
                return json.load(f)
    
    def _save_bounds(self):
        with open(self.bounds_file, 'w') as f:
            json.dump(self.bounds, f, indent=2)
        print(f"✓ Bounds updated and saved to {self.bounds_file}")
    
    def update_bounds(self, df: pd.DataFrame):
        updated = False
        for param in self.weights.keys():
            if param in df.columns:
                data_min = df[param].min()
                data_max = df[param].max()
                
                # Update bounds if new extremes found
                if data_min < self.bounds[param]['min']:
                    self.bounds[param]['min'] = data_min
                    updated = True
                    print(f"  New MIN for {param}: {data_min:.2f}")
                
                if data_max > self.bounds[param]['max']:
                    self.bounds[param]['max'] = data_max
                    updated = True
                    print(f"  New MAX for {param}: {data_max:.2f}")
        
        if updated:
            self._save_bounds()
        return updated
    
    def normalize_value(self, value: float, param: str) -> float:
       
        min_val = self.bounds[param]['min']
        max_val = self.bounds[param]['max']
        
        if max_val == min_val:
            return 0.0
        
        normalized = (value - min_val) / (max_val - min_val)
        return np.clip(normalized, 0.0, 1.0)
    
    def calculate_row_score(self, row: pd.Series) -> float:
        
        weighted_score = 0.0
        
        for param, weight in self.weights.items():
            if param in row and pd.notna(row[param]):
                normalized = self.normalize_value(row[param], param)
                weighted_score += normalized * weight
        
        # Convert to 0-100 scale
        return weighted_score * 100
    
    def calculate_drive_scores(self, df: pd.DataFrame) -> pd.DataFrame:
       
        df = df.copy()
        df['aggressiveness_score'] = df.apply(self.calculate_row_score, axis=1)
        return df
    
    def calculate_aggregate_score(self, scores: np.ndarray) -> Dict:
      
        mean_score = np.mean(scores)
        median_score = np.median(scores)
        std_score = np.std(scores)
        
        # Percentile analysis for spike detection
        p50 = np.percentile(scores, 50)
        p75 = np.percentile(scores, 75)
        p90 = np.percentile(scores, 90)
        p95 = np.percentile(scores, 95)
        p99 = np.percentile(scores, 99)
        max_score = np.max(scores)
        
        # Detect aggressive spikes (scores > 70)
        spike_threshold = 70
        spike_count = np.sum(scores >= spike_threshold)
        spike_percentage = (spike_count / len(scores)) * 100
        
        # Detect extreme spikes (scores > 85)
        extreme_threshold = 85
        extreme_count = np.sum(scores >= extreme_threshold)
        extreme_percentage = (extreme_count / len(scores)) * 100
        
        # Penalty increases exponentially with spike frequency and intensity
        spike_penalty = 0.0
        
        if p95 > 70:
            spike_penalty += (p95 - 70) * 0.3
        if p99 > 80:
            spike_penalty += (p99 - 80) * 0.5
        
        # Penalty for frequency of spikes
        if spike_percentage > 5:
            spike_penalty += (spike_percentage - 5) * 2.0
        if extreme_percentage > 2:
            spike_penalty += (extreme_percentage - 2) * 3.0
        
        # Calculate final aggregate score
        base_score = (mean_score * 0.7) + (p75 * 0.3)
        
        # Apply spike penalty
        final_score = np.clip(base_score + spike_penalty, 0, 100)
        
        return {
            'final_score': round(final_score, 2),
            'mean_score': round(mean_score, 2),
            'median_score': round(median_score, 2),
            'std_score': round(std_score, 2),
            'p75_score': round(p75, 2),
            'p90_score': round(p90, 2),
            'p95_score': round(p95, 2),
            'p99_score': round(p99, 2),
            'max_score': round(max_score, 2),
            'spike_percentage': round(spike_percentage, 2),
            'extreme_percentage': round(extreme_percentage, 2),
            'spike_penalty': round(spike_penalty, 2)
        }
    
    def analyze_drive(self, csv_path: str, update_bounds: bool = True) -> Tuple[pd.DataFrame, Dict]:
      
        print(f"\n{'='*60}")
        print(f"ANALYZING DRIVE: {csv_path}")
        print(f"{'='*60}")
        
        # Load data
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} data points")
        
        # Update bounds if requested
        if update_bounds:
            print("\nUpdating bounds...")
            self.update_bounds(df)
        
        # Calculate scores
        print("\nCalculating aggressiveness scores...")
        df_scored = self.calculate_drive_scores(df)
        
        # Calculate aggregate
        aggregate = self.calculate_aggregate_score(df_scored['aggressiveness_score'].values)
        
        return df_scored, aggregate
    
    def get_current_bounds(self) -> Dict:
        return self.bounds
    
    def print_bounds(self):
        print("\nCurrent Parameter Bounds:")
        print("-" * 50)
        for param in self.weights.keys():
            min_val = self.bounds[param]['min']
            max_val = self.bounds[param]['max']
            print(f"{param:20s}: {min_val:8.2f} to {max_val:8.2f}")


if __name__ == "__main__":
    scorer = DrivingAggressivenessScorer()
    
    # Analyze a drive
    df_scored, results = scorer.analyze_drive('obd_data_log_20251012_121810.csv')
    
    # Save scored data
    output_path = 'obd_data_scored.csv'
    df_scored.to_csv(output_path, index=False)
    print(f"✓ Scored data saved to {output_path}")
    
    # Display current bounds
    scorer.print_bounds()