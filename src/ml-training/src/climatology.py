"""
climatology.py
===============
Stage: DAY 2, PART 1 — Climatology baseline (Stage 1 of the model ladder)

Rule from the brief: use training data ONLY. Prefer a day-of-year based
climatology if enough data exists (i.e. training spans multiple calendar
years, so every day-of-year has more than one sample to average). If
training only covers a single year (or, worse, only PART of a year, as
happens with the seasonal_block split), day-of-year climatology cannot
generalize to val/test months it never saw — so we fall back to the
simplest defensible option: a pure per-cell SPATIAL mean (same profile
predicted for every day, only varies by lat/lon). This choice is always
printed explicitly so it's never a silent assumption.

Works entirely in NORMALIZED target space (the temperature_target array
in ml_ready_dataset.nc is already per-depth normalized from Day 1) so
climatology and MLP predictions live in the same space and can be
inverse-transformed to degC identically at evaluation time.
"""

import numpy as np
import pandas as pd
import xarray as xr


class Climatology:
    """
    Holds a fitted climatology table plus the info needed to predict for
    arbitrary query dates/points.
    """

    def __init__(self, mode: str, table: np.ndarray, dayofyear_values=None):
        """
        mode: "day_of_year" or "spatial_only"
        table:
            - if mode == "day_of_year": shape (n_dayofyear, depth, lat, lon)
            - if mode == "spatial_only": shape (depth, lat, lon)
        dayofyear_values: sorted array of the actual day-of-year integers
            present in the table (only used when mode == "day_of_year").
        """
        self.mode = mode
        self.table = table  # numpy array
        self.dayofyear_values = dayofyear_values

    def predict(self, query_times: pd.DatetimeIndex, lat_idx: np.ndarray,
                lon_idx: np.ndarray) -> np.ndarray:
        """
        query_times: length-N array-like of Timestamps, ONE per sample.
        lat_idx, lon_idx: length-N integer arrays, the grid index of each
        sample (same convention as mlp_dataset.py's point samples).

        Returns: (N, depth) array of predicted (normalized) temperature.
        """
        lat_idx = np.asarray(lat_idx)
        lon_idx = np.asarray(lon_idx)
        n = len(lat_idx)

        if self.mode == "spatial_only":
            # table: (depth, lat, lon) -> just index by lat/lon, same for
            # every sample regardless of date.
            preds = self.table[:, lat_idx, lon_idx]  # (depth, N)
            return preds.T  # (N, depth)

        # mode == "day_of_year"
        query_doy = pd.DatetimeIndex(query_times).dayofyear.values  # (N,)

        # Map each query day-of-year to the NEAREST day-of-year actually
        # present in the training table (handles Dec31<->Jan1 wraparound
        # and any leap-year gaps). This is a documented approximation.
        available = self.dayofyear_values
        # searchsorted-based nearest neighbor lookup, vectorized
        insert_pos = np.searchsorted(available, query_doy)
        insert_pos = np.clip(insert_pos, 1, len(available) - 1)
        left = available[insert_pos - 1]
        right = available[insert_pos]
        use_right = (right - query_doy) < (query_doy - left)
        nearest_doy = np.where(use_right, right, left)

        n_fallback = int(np.sum(nearest_doy != query_doy))
        if n_fallback > 0:
            print(f"[climatology] NOTE: {n_fallback}/{n} query samples had a "
                  f"day-of-year not seen in training and were matched to the "
                  f"nearest available training day-of-year instead.")

        doy_to_row = {doy: i for i, doy in enumerate(available)}
        row_idx = np.array([doy_to_row[d] for d in nearest_doy])

        preds = self.table[row_idx, :, lat_idx, lon_idx]  # (N, depth)
        return preds


def fit_climatology(target_da: xr.DataArray, train_time_index: pd.DatetimeIndex) -> Climatology:
    """
    target_da: (time, depth, lat, lon) NORMALIZED temperature DataArray
        (i.e. ml_ready['temperature_target']).
    train_time_index: the Day 1 training split dates (pd.DatetimeIndex).
    """
    train_dt = pd.DatetimeIndex(train_time_index)
    train_years = sorted(set(train_dt.year))
    n_years = len(train_years)

    # IMPORTANT: day-of-year climatology is only trustworthy if training
    # actually covers close to the full annual cycle. Counting distinct
    # YEARS alone is not enough — under the seasonal_block split (used
    # whenever <3 years of data are available), train/val/test are
    # DISJOINT MONTH RANGES applied to every year present. So with e.g.
    # 2 years of data and train_months=[1..8], train_time_index contains
    # Jan-Aug of BOTH years (n_years == 2) but STILL never contains
    # Sep-Dec at all. Using day-of-year climatology in that case would
    # look "multi-year" but still can't generalize to the val/test
    # months, exactly like the single-year case. So the real decision
    # criterion is DAY-OF-YEAR COVERAGE, not year count.
    n_unique_doy = len(set(train_dt.dayofyear))
    coverage_fraction = n_unique_doy / 366.0

    train_target = target_da.sel(time=train_time_index)

    if n_years >= 2 and coverage_fraction >= 0.9:
        mode = "day_of_year"
        print(f"[climatology] Training spans {n_years} distinct years "
              f"({train_years}) and covers {n_unique_doy}/366 "
              f"({coverage_fraction*100:.0f}%) of the annual cycle -> using "
              f"DAY-OF-YEAR climatology (mean temperature profile per "
              f"calendar day-of-year, per lat/lon, averaged across "
              f"training years).")
        grouped = train_target.groupby("time.dayofyear").mean(dim="time", skipna=True)
        # grouped dims: (dayofyear, depth, lat, lon)
        dayofyear_values = grouped["dayofyear"].values
        table = grouped.transpose("dayofyear", "depth", "lat", "lon").values
        clim = Climatology(mode="day_of_year", table=table,
                            dayofyear_values=dayofyear_values)
    else:
        mode = "spatial_only"
        months_covered = sorted(set(train_dt.month))
        print(f"[climatology] Training spans {n_years} distinct year(s) but "
              f"covers only {n_unique_doy}/366 ({coverage_fraction*100:.0f}%) "
              f"of the annual cycle (months present: {months_covered}). "
              f"NOT enough annual-cycle coverage for a day-of-year "
              f"climatology to generalize to val/test months it never saw "
              f"(this happens even with multiple years, e.g. under the "
              f"seasonal_block split where train/val/test are disjoint "
              f"month ranges repeated every year). ASSUMPTION: falling "
              f"back to the simplest defensible option — a pure SPATIAL "
              f"climatology (mean profile per lat/lon across all training "
              f"days, no seasonal/day-of-year dependence at all). This is "
              f"a real MVP limitation: predictions will be identical for "
              f"every day, varying only by location. Revisit once "
              f"training data covers the full annual cycle (either a "
              f"single full year, or multiple full years).")
        table = train_target.mean(dim="time", skipna=True) \
                             .transpose("depth", "lat", "lon").values
        clim = Climatology(mode="spatial_only", table=table, dayofyear_values=None)

    n_nan = int(np.isnan(clim.table).sum())
    if n_nan > 0:
        print(f"[climatology] NOTE: climatology table has {n_nan} NaN entries "
              f"(land cells / dayofyear-lat-lon combos with zero valid "
              f"training samples). These will naturally be excluded since "
              f"predictions are only ever requested at valid ocean sample "
              f"points.")

    return clim
