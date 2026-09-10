import os
import sys
from pathlib import Path
import requests
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import netCDF4 as nc

backend_dir = Path(__file__).resolve().parent
sys.path.append(str(backend_dir))

from config import settings
from model_service import ModelService

GDAC_BASE_URLS = [
    "https://data-argo.ifremer.fr/dac/",
    "https://usgodae.org/pub/outgoing/argo/dac/"
]

def download_argo_profile(file_rel_path: str, cache_dir: Path) -> Path:
    """Downloads an Argo netCDF profile if not already cached locally."""
    local_path = cache_dir / Path(file_rel_path).name
    if local_path.exists():
        return local_path

    for base_url in GDAC_BASE_URLS:
        url = base_url + file_rel_path.strip().lstrip("/")
        try:
            resp = requests.get(url, timeout=12)
            if resp.status_code == 200:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                with open(local_path, "wb") as f:
                    f.write(resp.content)
                return local_path
        except Exception:
            continue
    return None

def extract_actual_argo_profile(nc_file: Path, target_depths: list):
    """Extracts pressure/depth and in-situ temperature, interpolating to target depths."""
    try:
        ds = nc.Dataset(nc_file)
        # Argo NetCDF stores temperature in TEMP or TEMP_ADJUSTED
        if "TEMP_ADJUSTED" in ds.variables and np.any(~np.isnan(ds.variables["TEMP_ADJUSTED"][:])):
            temp_raw = ds.variables["TEMP_ADJUSTED"][0]
        else:
            temp_raw = ds.variables["TEMP"][0]

        # Pressure ~ Depth in meters (approx 1 dbar ≈ 1 meter)
        if "PRES_ADJUSTED" in ds.variables and np.any(~np.isnan(ds.variables["PRES_ADJUSTED"][:])):
            pres_raw = ds.variables["PRES_ADJUSTED"][0]
        else:
            pres_raw = ds.variables["PRES"][0]

        ds.close()

        # Clean fill values and NaNs
        mask = (~np.isnan(temp_raw)) & (~np.isnan(pres_raw)) & (temp_raw < 50.0) & (temp_raw > -3.0)
        p_clean = np.array(pres_raw[mask], dtype=float)
        t_clean = np.array(temp_raw[mask], dtype=float)

        if len(p_clean) < 5:
            return None

        # Sort by depth
        sort_idx = np.argsort(p_clean)
        p_clean = p_clean[sort_idx]
        t_clean = t_clean[sort_idx]

        # Interpolate observed temperatures to match model standard depths
        interpolator = interp1d(p_clean, t_clean, bounds_error=False, fill_value="extrapolate")
        actual_profile = interpolator(target_depths)
        return actual_profile

    except Exception:
        return None

def run_ground_truth_validation(target_stations=25):
    data_csv = backend_dir.parent / "data" / "indian_ocean_index.csv"
    cache_dir = backend_dir.parent / "data" / "argo_nc_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("--- 1. Loading Argo Indian Ocean Index Catalog ---")
    df = pd.read_csv(data_csv)
    df = df.dropna(subset=["latitude", "longitude", "date", "file"])

    # Filter strictly inside domain
    in_domain = df[
        (df["latitude"] >= settings.LAT_MIN) & (df["latitude"] <= settings.LAT_MAX) &
        (df["longitude"] >= settings.LON_MIN) & (df["longitude"] <= settings.LON_MAX)
    ]
    print(f"Profiles within Indian Ocean bounds: {len(in_domain):,}")

    depth_levels = getattr(settings, "STANDARD_DEPTHS", [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 1000])

    print("--- 2. Initializing ModelService ---")
    service = ModelService()

    preds_list = []
    actuals_list = []
    station_meta = []

    print(f"--- 3. Fetching Ground-Truth Profiles & Running Validation ---")
    sample_df = in_domain.head(200)  # Scan pool to obtain target_stations successfully

    for _, row in sample_df.iterrows():
        if len(preds_list) >= target_stations:
            break

        file_path = str(row["file"])
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        date_str = str(row["date"])[:10]

        # 1. Download/Load actual NetCDF
        local_nc = download_argo_profile(file_path, cache_dir)
        if not local_nc:
            continue

        # 2. Extract actual temperatures
        actual_profile = extract_actual_argo_profile(local_nc, depth_levels)
        if actual_profile is None:
            continue

        # 3. Predict with model
        try:
            res = service.predict(latitude=lat, longitude=lon, date_str=date_str)
            if hasattr(res, "profile"):
                pred_profile = [item.temperature if hasattr(item, "temperature") else item for item in res.profile]
            elif hasattr(res, "temperatures"):
                pred_profile = res.temperatures
            else:
                pred_profile = res.get("temperatures", res.get("profile", []))
            pred_profile = np.array(pred_profile, dtype=float).flatten()

            if len(pred_profile) != len(depth_levels):
                pred_profile = pred_profile[:len(depth_levels)]

            preds_list.append(pred_profile)
            actuals_list.append(actual_profile)
            station_meta.append({
                "wmo": row.get("wmo", "N/A"),
                "date": date_str,
                "lat": round(lat, 3),
                "lon": round(lon, 3)
            })
            print(f"[{len(preds_list)}/{target_stations}] Station WMO {row.get('wmo')} validated successfully.")
        except Exception as e:
            continue

    if not preds_list:
        print("Could not download/parse profiles. Check internet connection.")
        return

    preds_arr = np.array(preds_list)
    actuals_arr = np.array(actuals_list)

    # --- 4. Calculate Depth-Wise Accuracy Metrics ---
    metrics = []
    for d_idx, depth in enumerate(depth_levels):
        y_p = preds_arr[:, d_idx]
        y_t = actuals_arr[:, d_idx]
        diff = y_p - y_t

        mae = np.nanmean(np.abs(diff))
        rmse = np.sqrt(np.nanmean(diff ** 2))
        bias = np.nanmean(diff)

        metrics.append({
            "Depth": f"{depth}m",
            "Actual Mean (°C)": round(np.nanmean(y_t), 2),
            "Pred Mean (°C)": round(np.nanmean(y_p), 2),
            "RMSE (°C)": round(rmse, 3),
            "MAE (°C)": round(mae, 3),
            "Bias (°C)": round(bias, 3)
        })

    metrics_df = pd.DataFrame(metrics)
    metrics_csv = backend_dir / "argo_depth_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    total_diff = preds_arr - actuals_arr
    total_rmse = np.sqrt(np.nanmean(total_diff ** 2))
    total_mae = np.nanmean(np.abs(total_diff))
    total_bias = np.nanmean(total_diff)

    print("\n" + "=" * 80)
    print("      INDEPENDENT POST-HOC VALIDATION: ARGO OBSERVED VS MODEL PREDICTED")
    print("=" * 80)
    print(metrics_df.to_string(index=False))
    print("=" * 80)
    print(f"Overall Water Column RMSE : {total_rmse:.3f} °C")
    print(f"Overall Water Column MAE  : {total_mae:.3f} °C")
    print(f"Overall Mean Bias         : {total_bias:.3f} °C")
    print(f"\n[+] Detailed metric table exported to: {metrics_csv.resolve()}")

if __name__ == "__main__":
    run_ground_truth_validation(target_stations=25)