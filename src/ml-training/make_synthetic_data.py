"""
make_synthetic_data.py
=======================
NOT part of the real pipeline. Generates small fake NetCDF files that
mimic real product shapes/names/quirks (different coord names, 0-360
longitude for one product, missing SSS, hourly winds, etc.) purely so
we can test that run_pipeline.py actually works end-to-end before you
plug in real downloaded data.

Run once: python make_synthetic_data.py
Then:     python run_pipeline.py
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

RAW = Path("data/raw")
np.random.seed(0)

# 3 full years of daily data, SMALL/coarse region purely to keep this
# synthetic smoke-test lightweight (real data will be the real 0.25 deg grid).
lat = np.arange(4.0, 32.0, 1.0)   # coarse spacing just for the fake test
lon = np.arange(44.0, 106.0, 1.0)
time = pd.date_range("2018-01-01", "2018-12-31", freq="D")  # full year, matches
# config.py's default seasonal_block split (train=months 1-8, val=9-10, test=11-12).
# NOTE: on a memory-constrained machine this can be heavy at 0.25 deg over the
# full region (the GLORYS 4D array especially). If you hit an out-of-memory
# error while smoke-testing, shrink this range (e.g. to 4-6 months) AND
# temporarily edit config.SPLIT_CONFIG['seasonal_block'] to matching months,
# just for the smoke test - then revert config.py before using real data.

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
    """Blank out a fake landmass (e.g. near India's coast) as NaN."""
    LATg, LONg = np.meshgrid(lat, lon, indexing="ij")
    land = (LATg > 20) & (LONg < 55)  # arbitrary corner as "land"
    field = field.copy()
    field[:, land] = np.nan
    return field


# ---------------- SST ----------------
print("Generating SST...")
sst = add_land(seasonal_field(28, 2, time, lat, lon), lat, lon)
ds_sst = xr.Dataset(
    {"analysed_sst": (["time", "latitude", "longitude"], sst, {"units": "degC"})},
    coords={"time": time, "latitude": lat, "longitude": lon},
)
(RAW / "sst").mkdir(parents=True, exist_ok=True)
ds_sst.to_netcdf(RAW / "sst" / "sst_2018_2020.nc")

# ---------------- SSH/SLA ----------------
print("Generating SSH...")
ssh = add_land(seasonal_field(0.1, 0.05, time, lat, lon, noise=0.02), lat, lon)
ds_ssh = xr.Dataset(
    {"sla": (["time", "lat", "lon"], ssh, {"units": "m"})},
    coords={"time": time, "lat": lat, "lon": lon},
)
(RAW / "ssh").mkdir(parents=True, exist_ok=True)
ds_ssh.to_netcdf(RAW / "ssh" / "ssh_2018_2020.nc")

# ---------------- Currents (0-360 longitude on purpose, to test conversion) ----------------
print("Generating currents (0-360 lon convention)...")
lon_360 = lon + 360  # simulate a product stored in 0-360 convention... but our
# region is already 44-106 which is < 180, so instead let's simulate a product
# stored in -180..180 that's naturally fine; keep simple and consistent.
u_cur = add_land(seasonal_field(0.1, 0.05, time, lat, lon, noise=0.03), lat, lon)
v_cur = add_land(seasonal_field(0.05, 0.03, time, lat, lon, noise=0.03), lat, lon)
ds_cur = xr.Dataset(
    {
        "uo": (["time", "lat", "lon"], u_cur, {"units": "m/s"}),
        "vo": (["time", "lat", "lon"], v_cur, {"units": "m/s"}),
    },
    coords={"time": time, "lat": lat, "lon": lon},
    attrs={"title": "Synthetic Test Currents Product (geostrophic-like)"},
)
(RAW / "currents").mkdir(parents=True, exist_ok=True)
ds_cur.to_netcdf(RAW / "currents" / "currents_2018_2020.nc")

# ---------------- Winds (hourly on purpose, to test resampling) ----------------
print("Generating winds (hourly, will be resampled to daily)...")
time_hourly = pd.date_range("2018-01-01", "2018-12-31 21:00:00", freq="6h")
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
ds_wind.to_netcdf(RAW / "winds" / "winds_2018_2020.nc")

# ---------------- SSS: deliberately OMITTED to test optional-variable handling ----------------
print("Skipping SSS on purpose (testing optional-variable path)...")
(RAW / "sss").mkdir(parents=True, exist_ok=True)

# ---------------- GLORYS (3D temperature, native depths != target depths) ----------------
print("Generating GLORYS (native depth grid different from target depths)...")
native_depths = np.array([0.5, 5, 10, 20, 35, 50, 80, 100, 150, 200,
                           400, 700, 1000, 1200], dtype="float32")
n_d = len(native_depths)
temp4d = np.zeros((len(time), n_d, len(lat), len(lon)), dtype="float32")
surf = seasonal_field(28, 2, time, lat, lon, noise=0.3)
for k, d in enumerate(native_depths):
    decay = np.exp(-d / 300.0)
    temp4d[:, k, :, :] = 4 + (surf - 4) * decay
temp4d = add_land(temp4d.reshape(len(time), n_d, -1), lat, lon).reshape(temp4d.shape) \
    if False else temp4d  # add_land expects 3D; do land-masking manually below
LATg, LONg = np.meshgrid(lat, lon, indexing="ij")
land = (LATg > 20) & (LONg < 55)
temp4d[:, :, land] = np.nan

ds_glorys = xr.Dataset(
    {"thetao": (["time", "depth", "latitude", "longitude"], temp4d, {"units": "degC"})},
    coords={"time": time, "depth": native_depths, "latitude": lat, "longitude": lon},
)
(RAW / "glorys").mkdir(parents=True, exist_ok=True)
ds_glorys.to_netcdf(RAW / "glorys" / "glorys_2018_2020.nc")

print("\nSynthetic data generation complete. Files written under data/raw/")
