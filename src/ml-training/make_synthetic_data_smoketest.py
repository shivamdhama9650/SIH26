"""
make_synthetic_data_smoketest.py
==================================
TEST-ONLY. Same idea as make_synthetic_data.py but with a much smaller
time window (5 months) and coarser grid, purely so a full Day1->2->3
smoke test (needed to produce a real cnn_best.pt checkpoint for Stage 4
ARGO integration testing) fits in the sandbox's ~5 min wall-time /
3.9GB RAM budget. NOT for production use - production still uses
make_synthetic_data.py or (for real runs) real downloaded data.

Also temporarily needs config.SPLIT_CONFIG's seasonal_block months
narrowed to match this window - see run_smoketest.sh.
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

RAW = Path("data/raw")
np.random.seed(0)

lat = np.arange(4.0, 32.0, 1.5)
lon = np.arange(44.0, 106.0, 1.5)
time = pd.date_range("2018-01-01", "2018-05-31", freq="D")  # 5 months

LAT, LON = np.meshgrid(lat, lon, indexing="ij")


def seasonal_field(base, amp, time, lat, lon, noise=0.5):
    doy = time.dayofyear.values.reshape(-1, 1, 1)
    lat_r = lat.reshape(1, -1, 1)
    lon_r = lon.reshape(1, 1, -1)
    field = (base + amp * np.sin(2 * np.pi * doy / 365.0)
             + 0.05 * lat_r + 0.01 * lon_r
             + noise * np.random.randn(len(time), len(lat), len(lon)))
    return field.astype("float32")


def add_land(field, lat, lon):
    LATg, LONg = np.meshgrid(lat, lon, indexing="ij")
    land = (LATg > 20) & (LONg < 55)
    field = field.copy()
    field[:, land] = np.nan
    return field


print("Generating SST...")
sst = add_land(seasonal_field(28, 2, time, lat, lon), lat, lon)
ds_sst = xr.Dataset(
    {"analysed_sst": (["time", "latitude", "longitude"], sst, {"units": "degC"})},
    coords={"time": time, "latitude": lat, "longitude": lon},
)
(RAW / "sst").mkdir(parents=True, exist_ok=True)
ds_sst.to_netcdf(RAW / "sst" / "sst_2018.nc")

print("Generating SSH...")
ssh = add_land(seasonal_field(0.1, 0.05, time, lat, lon, noise=0.02), lat, lon)
ds_ssh = xr.Dataset(
    {"sla": (["time", "lat", "lon"], ssh, {"units": "m"})},
    coords={"time": time, "lat": lat, "lon": lon},
)
(RAW / "ssh").mkdir(parents=True, exist_ok=True)
ds_ssh.to_netcdf(RAW / "ssh" / "ssh_2018.nc")

print("Generating currents...")
u_cur = add_land(seasonal_field(0.1, 0.05, time, lat, lon, noise=0.03), lat, lon)
v_cur = add_land(seasonal_field(0.05, 0.03, time, lat, lon, noise=0.03), lat, lon)
ds_cur = xr.Dataset(
    {
        "uo": (["time", "lat", "lon"], u_cur, {"units": "m/s"}),
        "vo": (["time", "lat", "lon"], v_cur, {"units": "m/s"}),
    },
    coords={"time": time, "lat": lat, "lon": lon},
)
(RAW / "currents").mkdir(parents=True, exist_ok=True)
ds_cur.to_netcdf(RAW / "currents" / "currents_2018.nc")

print("Generating winds (6-hourly, will be resampled to daily)...")
time_hourly = pd.date_range("2018-01-01", "2018-05-31 18:00:00", freq="6h")
doy_h = time_hourly.dayofyear.values.reshape(-1, 1, 1)
u_wind = (3 + 2 * np.sin(2 * np.pi * doy_h / 365.0)
          + 0.5 * np.random.randn(len(time_hourly), len(lat), len(lon))).astype("float32")
v_wind = (1 + 1 * np.cos(2 * np.pi * doy_h / 365.0)
          + 0.5 * np.random.randn(len(time_hourly), len(lat), len(lon))).astype("float32")
u_wind = add_land(u_wind, lat, lon)
v_wind = add_land(v_wind, lat, lon)
ds_wind = xr.Dataset(
    {
        "u10": (["time", "latitude", "longitude"], u_wind, {"units": "m/s"}),
        "v10": (["time", "latitude", "longitude"], v_wind, {"units": "m/s"}),
    },
    coords={"time": time_hourly, "latitude": lat, "longitude": lon},
)
(RAW / "winds").mkdir(parents=True, exist_ok=True)
ds_wind.to_netcdf(RAW / "winds" / "winds_2018.nc")

print("Skipping SSS on purpose (testing optional-variable path)...")
(RAW / "sss").mkdir(parents=True, exist_ok=True)

print("Generating GLORYS...")
native_depths = np.array([0.5, 5, 10, 20, 35, 50, 80, 100, 150, 200,
                           400, 700, 1000, 1200], dtype="float32")
n_d = len(native_depths)
temp4d = np.zeros((len(time), n_d, len(lat), len(lon)), dtype="float32")
surf = seasonal_field(28, 2, time, lat, lon, noise=0.3)
for k, d in enumerate(native_depths):
    decay = np.exp(-d / 300.0)
    temp4d[:, k, :, :] = 4 + (surf - 4) * decay
LATg, LONg = np.meshgrid(lat, lon, indexing="ij")
land = (LATg > 20) & (LONg < 55)
temp4d[:, :, land] = np.nan

ds_glorys = xr.Dataset(
    {"thetao": (["time", "depth", "latitude", "longitude"], temp4d, {"units": "degC"})},
    coords={"time": time, "depth": native_depths, "latitude": lat, "longitude": lon},
)
(RAW / "glorys").mkdir(parents=True, exist_ok=True)
ds_glorys.to_netcdf(RAW / "glorys" / "glorys_2018.nc")

print("\nSmoke-test synthetic data generation complete.")
print(f"Grid: {len(lat)} lat x {len(lon)} lon, {len(time)} days "
      f"({time.min().date()} -> {time.max().date()})")
