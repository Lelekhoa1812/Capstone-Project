import yaml
from driving_aggressiveness_scorer import DrivingAggressivenessScorer
from driving_analyzer import visualize_drive, compare_drives


def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        print("Using default configuration.")
        return None


def create_scorer_from_config(config_path: str = 'config.yaml') -> DrivingAggressivenessScorer:
    """Create scorer instance from configuration file."""
    config = load_config(config_path)
    
    if config:
        weights = config.get('weights', None)
        bounds_file = config.get('bounds', {}).get('file', 'obd_bounds.json')
        scorer = DrivingAggressivenessScorer(bounds_file=bounds_file, weights=weights)
        print(f"✓ Scorer initialized with config from {config_path}")
    else:
        scorer = DrivingAggressivenessScorer()
        print("✓ Scorer initialized with default settings")
    
    return scorer


# Quick start examples
if __name__ == "__main__":
    
    # METHOD 1: Use with config file (recommended)
    print("\n" + "="*60)
    print("METHOD 1: Config-based scoring")
    print("="*60)
    scorer = create_scorer_from_config('config.yaml')
    df_scored, results = scorer.analyze_drive('obd_data_log_20251012_121810.csv')
    visualize_drive(df_scored, results, save_path='drive_analysis_config.png')
    
    
    # METHOD 2: Use with custom weights (no config file)
    print("\n" + "="*60)
    print("METHOD 2: Custom weights")
    print("="*60)
    custom_weights = {
        'RPM': 0.20,
        'THROTTLE_POS': 0.35,  # More emphasis on throttle
        'ENGINE_LOAD': 0.25,
        'MAF': 0.10,
        'SPEED': 0.05,
        'INTAKE_PRESSURE': 0.05
    }
    scorer_custom = DrivingAggressivenessScorer(weights=custom_weights)
    df_scored2, results2 = scorer_custom.analyze_drive('obd_data_log_20251012_121810.csv')
    
    
    # METHOD 3: Analyze without updating bounds (testing)
    print("\n" + "="*60)
    print("METHOD 3: Analysis without updating bounds")
    print("="*60)
    scorer_test = DrivingAggressivenessScorer()
    df_test, results_test = scorer_test.analyze_drive(
        'obd_data_log_20251012_121810.csv', 
        update_bounds=False  # Don't update global bounds
    )
    
    
    # METHOD 4: Quick comparison script
    print("\n" + "="*60)
    print("METHOD 4: Compare multiple drives")
    print("="*60)
    """
    # Uncomment when you have multiple CSV files:
    comparison = compare_drives(scorer, [
        'obd_data_log_20251012_121810.csv',
        'obd_data_log_20251013_101234.csv',
        'obd_data_log_20251014_155030.csv'
    ])
    """
    
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nYour system is ready to:")
    print("  1. Analyze individual drives")
    print("  2. Compare multiple drives")
    print("  3. Batch process folders")
    print("  4. Dynamically update bounds")
    print("  5. Generate visualizations")
    print("\nBounds file: obd_bounds.json")
    print("Config file: config.yaml")
    print("="*60 + "\n")