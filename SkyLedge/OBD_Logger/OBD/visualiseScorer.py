import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from driving_aggressiveness_scorer import DrivingAggressivenessScorer


def visualize_drive(df_scored: pd.DataFrame, results: dict, save_path: str = None):
    """
    Create comprehensive visualization of drive analysis.
    
    Args:
        df_scored: DataFrame with aggressiveness scores
        results: Aggregate results dictionary
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f"Drive Analysis - Score: {results['final_score']:.1f}/100", 
                 fontsize=16, fontweight='bold')
    
    # 1. Aggressiveness Score Over Time
    ax = axes[0, 0]
    ax.plot(df_scored['aggressiveness_score'], linewidth=1, color='#2E86AB')
    ax.axhline(y=results['mean_score'], color='green', linestyle='--', 
               label=f"Mean: {results['mean_score']:.1f}")
    ax.axhline(y=70, color='orange', linestyle='--', alpha=0.5, label='Spike Threshold')
    ax.axhline(y=85, color='red', linestyle='--', alpha=0.5, label='Extreme Threshold')
    ax.set_title('Aggressiveness Score Timeline')
    ax.set_ylabel('Score (0-100)')
    ax.set_xlabel('Sample Number')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Score Distribution
    ax = axes[0, 1]
    ax.hist(df_scored['aggressiveness_score'], bins=50, color='#A23B72', alpha=0.7, edgecolor='black')
    ax.axvline(x=results['mean_score'], color='green', linestyle='--', linewidth=2, label='Mean')
    ax.axvline(x=results['median_score'], color='blue', linestyle='--', linewidth=2, label='Median')
    ax.set_title('Score Distribution')
    ax.set_xlabel('Aggressiveness Score')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. RPM vs Throttle Position (colored by score)
    ax = axes[1, 0]
    scatter = ax.scatter(df_scored['THROTTLE_POS'], df_scored['RPM'], 
                        c=df_scored['aggressiveness_score'], cmap='RdYlGn_r', 
                        s=10, alpha=0.6)
    ax.set_title('RPM vs Throttle Position')
    ax.set_xlabel('Throttle Position (%)')
    ax.set_ylabel('RPM')
    plt.colorbar(scatter, ax=ax, label='Aggressiveness')
    ax.grid(True, alpha=0.3)
    
    # 4. Speed vs Engine Load (colored by score)
    ax = axes[1, 1]
    scatter = ax.scatter(df_scored['SPEED'], df_scored['ENGINE_LOAD'], 
                        c=df_scored['aggressiveness_score'], cmap='RdYlGn_r', 
                        s=10, alpha=0.6)
    ax.set_title('Speed vs Engine Load')
    ax.set_xlabel('Speed (km/h)')
    ax.set_ylabel('Engine Load (%)')
    plt.colorbar(scatter, ax=ax, label='Aggressiveness')
    ax.grid(True, alpha=0.3)
    
    # 5. Key Metrics Over Time
    ax = axes[2, 0]
    ax2 = ax.twinx()
    
    ln1 = ax.plot(df_scored['RPM'] / 100, label='RPM/100', color='#E63946', linewidth=0.8)
    ln2 = ax.plot(df_scored['THROTTLE_POS'], label='Throttle %', color='#F77F00', linewidth=0.8)
    ln3 = ax2.plot(df_scored['SPEED'], label='Speed', color='#06FFA5', linewidth=0.8)
    
    ax.set_title('Key Metrics Timeline')
    ax.set_xlabel('Sample Number')
    ax.set_ylabel('RPM/100 & Throttle %')
    ax2.set_ylabel('Speed (km/h)')
    
    # Combine legends
    lns = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 6. Score Statistics Summary
    ax = axes[2, 1]
    ax.axis('off')
    
    stats_text = f"""
    AGGREGATE SCORE BREAKDOWN
    {'─' * 40}
    
    Final Score:          {results['final_score']:.1f} / 100
    
    SCORE STATISTICS
    Mean:                 {results['mean_score']:.1f}
    Median:               {results['median_score']:.1f}
    Std Dev:              {results['std_score']:.1f}
    
    PERCENTILES
    75th:                 {results['p75_score']:.1f}
    90th:                 {results['p90_score']:.1f}
    95th:                 {results['p95_score']:.1f}
    99th:                 {results['p99_score']:.1f}
    Max:                  {results['max_score']:.1f}
    
    SPIKE ANALYSIS
    Spikes (>70):         {results['spike_percentage']:.1f}%
    Extreme (>85):        {results['extreme_percentage']:.1f}%
    Spike Penalty:        +{results['spike_penalty']:.1f}
    """
    
    ax.text(0.1, 0.95, stats_text, transform=ax.transAxes, 
            fontfamily='monospace', fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to {save_path}")
        plt.close() 
    else:
        plt.show()


def compare_drives(scorer: DrivingAggressivenessScorer, csv_paths: list):
    """
    Compare multiple drives side-by-side.
    
    Args:
        scorer: DrivingAggressivenessScorer instance
        csv_paths: List of CSV file paths to compare
    """
    results_list = []
    
    for csv_path in csv_paths:
        _, results = scorer.analyze_drive(csv_path, update_bounds=True)
        results['file'] = csv_path
        results_list.append(results)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results_list)
    
    print("\n" + "="*80)
    print("DRIVE COMPARISON")
    print("="*80)
    print(comparison_df[['file', 'final_score', 'mean_score', 
                         'spike_percentage', 'spike_penalty']].to_string(index=False))
    print("="*80 + "\n")
    
    return comparison_df


def batch_analyze_folder(folder_path: str, pattern: str = "*.csv"):
  
    from pathlib import Path
    
    scorer = DrivingAggressivenessScorer()
    csv_files = list(Path(folder_path).glob(pattern))
    
    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    all_results = []
    for csv_file in csv_files:
        try:
            df_scored, results = scorer.analyze_drive(str(csv_file), update_bounds=True)
            results['filename'] = csv_file.name
            all_results.append(results)
            
            # Save individual scored file
            output_path = csv_file.parent / f"{csv_file.stem}_scored.csv"
            df_scored.to_csv(output_path, index=False)
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
    
    summary_df = pd.DataFrame(all_results)
    summary_path = Path(folder_path) / "drive_summary_report.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✓ Summary report saved to {summary_path}")
    
    return summary_df


def export_bounds_report(scorer: DrivingAggressivenessScorer, output_path: str = "bounds_report.txt"):
    bounds = scorer.get_current_bounds()
    
    report = []
    report.append("="*60)
    report.append("DRIVING AGGRESSIVENESS SCORER - BOUNDS REPORT")
    report.append("="*60)
    report.append(f"\nGenerated: {pd.Timestamp.now()}\n")
    
    report.append("PARAMETER WEIGHTS:")
    report.append("-"*60)
    for param, weight in scorer.weights.items():
        report.append(f"{param:20s}: {weight:.3f} ({weight*100:.1f}%)")
    
    report.append("\n\nCURRENT BOUNDS:")
    report.append("-"*60)
    report.append(f"{'Parameter':<20s} {'Min':>12s} {'Max':>12s} {'Range':>12s}")
    report.append("-"*60)
    
    for param in scorer.weights.keys():
        min_val = bounds[param]['min']
        max_val = bounds[param]['max']
        range_val = max_val - min_val
        report.append(f"{param:<20s} {min_val:>12.2f} {max_val:>12.2f} {range_val:>12.2f}")
    
    report.append("="*60)
    
    report_text = "\n".join(report)
    
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n✓ Report saved to {output_path}")


# Example usage
if __name__ == "__main__":
    scorer = DrivingAggressivenessScorer()
    
    csv_path = 'obd_data_log_20251012_121810.csv'
    df_scored, results = scorer.analyze_drive(csv_path)
    visualize_drive(df_scored, results, save_path='drive_analysis.png')
    
    
    # Export bounds report
    export_bounds_report(scorer)