# To be executed on a Colab session
from pathlib import Path
import pandas as pd
import os

# Mount Google Drive if not already done
from google.colab import drive
drive.mount('/content/drive')

# Define directory paths
data_dir = Path('/content/drive/My Drive/EAT40005/Logs')
output_dir = data_dir / 'merge'
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / 'merged_w13.csv'

# Collect all CSV files (excluding previously merged output if exists) - change filepath if needed
csv_files = [f for f in data_dir.glob('*.csv') if f.name != 'merged_w13.csv']

# Merge all CSVs
merged_df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

# Save to output path
merged_df.to_csv(output_file, index=False)

print(f"Merged {len(csv_files)} files into {output_file}")
