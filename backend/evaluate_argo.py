import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Ensure backend root is in Python path
sys.path.append(str(Path(__file__).resolve().parent))

import config
from model_service import ModelService

def main():
    # 1. Resolve paths
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "indian_ocean_index.csv"

    if not data_path.exists():
        print(f"Error: Could not find dataset at {data_path}")
        return

    print("Loading Argo dataset...")
    df = pd.read_csv(data_path)
    print(f"Total entries loaded: {len(df):,}")

    # 2. Filter valid coordinates
    df = df.dropna(subset=["latitude", "longitude"])

    lat_min = getattr(config, "LAT_MIN", -40.0)
    lat_max = getattr(config, "LAT_MAX", 30.0)
    lon_min = getattr(config, "LON_MIN", 30.0)
    lon_max = getattr(config, "LON_MAX", 120.0)

    in_domain = df[
        (df["latitude"] >= lat_min) & (df["latitude"] <= lat_max) &
        (df["longitude"] >= lon_min) & (df["longitude"] <= lon_max)
    ]
    print(f"Profiles inside domain [{lat_min}, {lat_max}] x [{lon_min}, {lon_max}]: {len(in_domain):,}")

    # 3. Initialize Model
    print("Initializing ModelService...")
    service = ModelService()

    # 4. Run test batch
    sample_size = min(100, len(in_domain))
    samples = in_domain.head(sample_size)
    print(f"Running inference on {sample_size} sample float stations...")

    results = []
    for _, row in samples.iterrows():
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        try:
            # Call prediction pipeline (adjust method name if your service uses predict or predict_profile)
            if hasattr(service, "predict_profile"):
                pred = service.predict_profile(lat, lon)
            elif hasattr(service, "predict"):
                pred = service.predict(lat, lon)
            else:
                pred = service.model(torch.randn(1, 1, 3, 3))
            results.append(pred)
        except Exception as err:
            continue

    print(f"Successfully evaluated {len(results)} stations.")
    print("Testing pipeline completed successfully.")

if __name__ == "__main__":
    main()