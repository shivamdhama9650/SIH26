import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

backend_dir = Path(__file__).resolve().parent
sys.path.append(str(backend_dir))

from config import settings
from model_service import ModelService

def run_argo_evaluation(sample_size=100):
    data_csv = backend_dir.parent / "data" / "indian_ocean_index.csv"
    
    if not data_csv.exists():
        print(f"Error: Dataset not found at {data_csv}")
        return

    print("--- 1. Loading Argo Indian Ocean Dataset ---")
    df = pd.read_csv(data_csv)
    print(f"Total entries loaded: {len(df):,}")

    # Drop NaNs and filter inside the model's domain
    df = df.dropna(subset=["latitude", "longitude", "date"])
    in_domain = df[
        (df["latitude"] >= settings.LAT_MIN) & (df["latitude"] <= settings.LAT_MAX) &
        (df["longitude"] >= settings.LON_MIN) & (df["longitude"] <= settings.LON_MAX)
    ]
    print(f"Profiles within Indian Ocean domain: {len(in_domain):,}")

    sample_stations = in_domain.head(sample_size).copy()

    print("\n--- 2. Initializing ModelService ---")
    service = ModelService()
    print(f"Model mode: {getattr(service, 'active_mode', settings.MODEL_MODE)}")

    results_records = []
    depth_levels = getattr(settings, 'STANDARD_DEPTHS', [0, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 1000])

    print(f"\n--- 3. Running Inference on {len(sample_stations)} Stations ---")
    for idx, row in sample_stations.iterrows():
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        date_str = str(row["date"])[:10]  # format: YYYY-MM-DD

        try:
            res = service.predict(latitude=lat, longitude=lon, date_str=date_str)
            
            # Extract predicted temperatures list from schemas.PredictionResponse
            if hasattr(res, "profile"):
                profile = [item.temperature if hasattr(item, "temperature") else item for item in res.profile]
            elif hasattr(res, "temperatures"):
                profile = res.temperatures
            elif isinstance(res, dict):
                profile = res.get("temperatures", res.get("profile", []))
            else:
                profile = list(res)

            profile_vals = [float(t) for t in profile]
            
            record = {
                "wmo": row.get("wmo", "N/A"),
                "date": date_str,
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
            }
            # Attach depth columns
            for i, temp in enumerate(profile_vals):
                depth_name = f"depth_{depth_levels[i]}m" if i < len(depth_levels) else f"level_{i}"
                record[depth_name] = round(temp, 2)
            
            results_records.append(record)

        except Exception as e:
            print(f"Error at station (Lat: {lat}, Lon: {lon}, Date: {date_str}): {e}")

    print(f"\nSuccessfully evaluated {len(results_records)} / {len(sample_stations)} stations.")

    if results_records:
        results_df = pd.DataFrame(results_records)
        output_file = backend_dir / "argo_test_results.csv"
        results_df.to_csv(output_file, index=False)
        print(f"[+] Output CSV exported to: {output_file.resolve()}")

        # Compute statistics across all depth columns
        temp_cols = [col for col in results_df.columns if col.startswith("depth_") or col.startswith("level_")]
        temp_matrix = results_df[temp_cols].to_numpy()

        print("\n" + "=" * 65)
        print("                ARGO MODEL TESTING SUMMARY REPORT               ")
        print("=" * 65)
        print(f"Total Float Stations Evaluated : {len(results_records)}")
        print(f"Vertical Depth Levels Tested   : {len(temp_cols)}")
        print(f"Mean Ocean Temperature         : {np.nanmean(temp_matrix):.2f} °C")
        print(f"Min / Max Temperature Range    : {np.nanmin(temp_matrix):.2f} °C to {np.nanmax(temp_matrix):.2f} °C")
        if len(temp_cols) > 0:
            print(f"Average Surface Temp ({temp_cols[0]}) : {np.nanmean(temp_matrix[:, 0]):.2f} °C")
            print(f"Average Deep Temp ({temp_cols[-1]})  : {np.nanmean(temp_matrix[:, -1]):.2f} °C")
        print("=" * 65)

        print("\nFirst 3 Evaluation Station Samples:")
        print(results_df[["wmo", "date", "latitude", "longitude"] + temp_cols[:3]].head(3).to_string(index=False))

if __name__ == "__main__":
    run_argo_evaluation(sample_size=100)