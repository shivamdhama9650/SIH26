
import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch.utils.data import Dataset
from config import PATCH_SIZE
from src.patch_extraction import find_valid_patch_centers

_HALF = PATCH_SIZE // 2
_OFFSETS = [(di, dj) for di in range(-_HALF, _HALF + 1) for dj in range(-_HALF, _HALF + 1)]
TIME_CHUNK = 30   # smaller chunks for patches (they use more RAM per sample)


def find_valid_patch_samples(ds, time_index, channels, patch_size=PATCH_SIZE):
    half = patch_size // 2
    static_centers = find_valid_patch_centers(ds["ocean_mask"], patch_size=patch_size)
    static_mask = static_centers.values   # (lat, lon) bool
    n_lat, n_lon = static_mask.shape
    n_time = len(time_index)

    valid = np.zeros((n_time, n_lat, n_lon), dtype=bool)

    for start in range(0, n_time, TIME_CHUNK):
        sl = slice(start, min(start + TIME_CHUNK, n_time))
        sub = ds.sel(time=time_index[sl])
        n_t = sl.stop - sl.start if sl.stop else n_time - start
        n_t = len(time_index[sl])

        v = np.broadcast_to(static_mask, (n_t, n_lat, n_lon)).copy()

        for ch in channels:
            arr = sub[ch].values
            if arr.ndim == 4: arr = arr.squeeze(axis=1)
            finite = np.isfinite(arr)
            core = np.ones((n_t, n_lat - 2*half, n_lon - 2*half), dtype=bool)
            for di, dj in _OFFSETS:
                core &= finite[:, half+di: n_lat-half+di, half+dj: n_lon-half+dj]
            ch_valid = np.zeros((n_t, n_lat, n_lon), dtype=bool)
            ch_valid[:, half:n_lat-half, half:n_lon-half] = core
            v &= ch_valid

        tgt = sub["temperature_target"].values
        v &= np.isfinite(tgt).all(axis=1)
        valid[sl] = v
        del sub, tgt, v

    return valid


def gather_patch_samples(ds, time_index, channels, valid_mask, patch_size=PATCH_SIZE):
    half = patch_size // 2
    n_time = len(time_index)
    n_depth = ds.sizes["depth"]

    time_idx, lat_idx, lon_idx = np.where(valid_mask)
    n_samples = len(time_idx)
    print(f"[cnn_dataset] Gathering {n_samples:,} valid {patch_size}x{patch_size} "
          f"patch samples ({100*n_samples/valid_mask.size:.2f}% usable). "
          f"Processing in chunks...")

    X = np.empty((n_samples, len(channels), patch_size, patch_size), dtype=np.float32)
    Y = np.empty((n_samples, n_depth), dtype=np.float32)

    for start in range(0, n_time, TIME_CHUNK):
        end = min(start + TIME_CHUNK, n_time)
        mask_t = (time_idx >= start) & (time_idx < end)
        if not mask_t.any():
            continue

        t_local = time_idx[mask_t] - start
        lat_c   = lat_idx[mask_t]
        lon_c   = lon_idx[mask_t]

        sub = ds.sel(time=time_index[start:end])

        for c, ch in enumerate(channels):
            arr = sub[ch].values
            if arr.ndim == 4: arr = arr.squeeze(axis=1)
            for di, dj in _OFFSETS:
                X[mask_t, c, di+half, dj+half] = arr[t_local, lat_c+di, lon_c+dj]

        tgt = sub["temperature_target"].values
        for d in range(n_depth):
            Y[mask_t, d] = tgt[t_local, d, lat_c, lon_c]
        del sub, tgt

    print(f"[cnn_dataset] Done.")
    all_times = pd.DatetimeIndex(ds.sel(time=time_index)["time"].values)
    idx = {"time_idx": time_idx, "lat_idx": lat_idx, "lon_idx": lon_idx}
    return X, Y, idx, all_times[time_idx]


class PatchDataset(Dataset):
    def __init__(self, X, Y):
        self.X, self.Y = torch.from_numpy(X), torch.from_numpy(Y)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.Y[i]
