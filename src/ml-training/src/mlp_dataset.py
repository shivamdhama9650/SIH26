import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch.utils.data import Dataset

TIME_CHUNK = 50   # process 50 days at a time to keep RAM low


def find_valid_point_samples(ds: xr.Dataset, time_index: pd.DatetimeIndex,
                              channels: list) -> np.ndarray:
    """Returns bool array (n_time, n_lat, n_lon) using chunked processing."""
    n_lat  = ds.sizes['lat']
    n_lon  = ds.sizes['lon']
    n_time = len(time_index)
    valid  = np.zeros((n_time, n_lat, n_lon), dtype=bool)
    ocean  = ds["ocean_mask"].values.astype(bool)          # (lat, lon)

    for start in range(0, n_time, TIME_CHUNK):
        sl   = slice(start, min(start + TIME_CHUNK, n_time))
        chunk_times = time_index[sl]
        sub  = ds.sel(time=chunk_times)
        n_t  = len(chunk_times)

        v = np.broadcast_to(ocean, (n_t,) + ocean.shape).copy()  # (t, lat, lon)

        for ch in channels:
            arr = sub[ch].values
            if arr.ndim == 4: arr = arr.squeeze(axis=1)
            v &= np.isfinite(arr)

        tgt = sub["temperature_target"].values                    # (t, depth, lat, lon)
        v  &= np.isfinite(tgt).all(axis=1)

        valid[sl] = v
        del sub, tgt, v

    return valid


def gather_point_samples(ds: xr.Dataset, time_index: pd.DatetimeIndex,
                          channels: list, valid_mask: np.ndarray):
    """Gathers point samples in time chunks to keep RAM usage low."""
    n_time = len(time_index)
    n_depth = ds.sizes['depth']
    n_ch = len(channels)

    time_idx, lat_idx, lon_idx = np.where(valid_mask)
    n_samples = len(time_idx)
    print(f"[mlp_dataset] Gathering {n_samples:,} valid samples "
          f"({100*n_samples/valid_mask.size:.2f}% usable). Processing in chunks...")

    X = np.empty((n_samples, n_ch),    dtype=np.float32)
    Y = np.empty((n_samples, n_depth), dtype=np.float32)

    filled = 0
    for start in range(0, n_time, TIME_CHUNK):
        end = min(start + TIME_CHUNK, n_time)
        # find which samples belong to this time chunk
        mask_t = (time_idx >= start) & (time_idx < end)
        if not mask_t.any():
            continue

        t_idx_chunk   = time_idx[mask_t] - start   # local time offset
        lat_idx_chunk = lat_idx[mask_t]
        lon_idx_chunk = lon_idx[mask_t]

        chunk_times = time_index[start:end]
        sub = ds.sel(time=chunk_times)

        for i, ch in enumerate(channels):
            arr = sub[ch].values
            if arr.ndim == 4: arr = arr.squeeze(axis=1)
            X[mask_t, i] = arr[t_idx_chunk, lat_idx_chunk, lon_idx_chunk]

        tgt = sub["temperature_target"].values   # (t, depth, lat, lon)
        for d in range(n_depth):
            Y[mask_t, d] = tgt[t_idx_chunk, d, lat_idx_chunk, lon_idx_chunk]

        filled += mask_t.sum()
        del sub, tgt

    print(f"[mlp_dataset] Done — {filled:,} samples gathered.")
    all_times = pd.DatetimeIndex(ds.sel(time=time_index)["time"].values)
    return X, Y, {"time_idx": time_idx, "lat_idx": lat_idx, "lon_idx": lon_idx}, all_times[time_idx]


class PointDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X, self.Y = torch.from_numpy(X), torch.from_numpy(Y)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.Y[i]
