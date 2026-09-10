"""
splitting.py
============
Stage: TRAIN / VALIDATION / TEST SPLIT

No random day-level split, ever.

- If >= 2 distinct calendar years are present: whole-year blocks
  (train = earlier years, val = one later year, test = one later year).
- If only 1 calendar year is present: contiguous seasonal/monthly blocks
  spread across train/val/test (config.SPLIT_CONFIG['seasonal_block']).

Returns boolean time masks (not indices) so they can be applied directly
with xr.DataArray.sel(time=mask) - actually we return the split as a
dict of {split_name: pd.DatetimeIndex} for clarity and print the
resulting date ranges.
"""

import numpy as np
import pandas as pd

from config import SPLIT_CONFIG


def _years_present(calendar: pd.DatetimeIndex):
    return sorted(calendar.year.unique().tolist())


def compute_split(calendar: pd.DatetimeIndex) -> dict:
    years = _years_present(calendar)
    print(f"[splitting] Calendar years present: {years}")

    mode = SPLIT_CONFIG["mode"]
    if mode == "auto":
        mode = "whole_year" if len(years) >= 3 else "seasonal_block"
        print(f"[splitting] mode='auto' resolved to '{mode}' "
              f"({len(years)} distinct year(s) available).")

    if mode == "whole_year":
        train_years = SPLIT_CONFIG["train_years"]
        val_years = SPLIT_CONFIG["val_years"]
        test_years = SPLIT_CONFIG["test_years"]

        if train_years is None or val_years is None or test_years is None:
            if len(years) < 3:
                raise ValueError(
                    f"[splitting] mode='whole_year' requires >=3 distinct years "
                    f"(got {len(years)}: {years}). VERIFY REQUIRED: either "
                    f"provide more data, set SPLIT_CONFIG explicitly, or use "
                    f"mode='seasonal_block'."
                )
            # default: last year = test, second-last = val, rest = train
            test_years = [years[-1]]
            val_years = [years[-2]]
            train_years = years[:-2]
            print(f"[splitting] No explicit years configured - defaulting to "
                  f"train={train_years}, val={val_years}, test={test_years}. "
                  f"VERIFY REQUIRED: confirm this split makes sense, or set "
                  f"config.SPLIT_CONFIG explicitly.")

        overlap = (set(train_years) & set(val_years)) | \
                  (set(train_years) & set(test_years)) | \
                  (set(val_years) & set(test_years))
        if overlap:
            raise ValueError(f"[splitting] Years {overlap} appear in more than "
                              f"one split - a year must never be split.")

        splits = {
            "train": calendar[calendar.year.isin(train_years)],
            "val": calendar[calendar.year.isin(val_years)],
            "test": calendar[calendar.year.isin(test_years)],
        }

    elif mode == "seasonal_block":
        blocks = SPLIT_CONFIG["seasonal_block"]
        splits = {
            "train": calendar[calendar.month.isin(blocks["train_months"])],
            "val": calendar[calendar.month.isin(blocks["val_months"])],
            "test": calendar[calendar.month.isin(blocks["test_months"])],
        }
        print(f"[splitting] Using seasonal/monthly blocks (single-year pilot "
              f"fallback): train_months={blocks['train_months']}, "
              f"val_months={blocks['val_months']}, "
              f"test_months={blocks['test_months']}.")
    else:
        raise ValueError(f"[splitting] Unknown split mode '{mode}'.")

    for name, idx in splits.items():
        if len(idx) == 0:
            raise ValueError(f"[splitting] Split '{name}' is EMPTY. Check your "
                              f"SPLIT_CONFIG against the years/months actually "
                              f"present in the data.")
        print(f"[splitting] {name}: {idx.min().date()} -> {idx.max().date()} "
              f"({len(idx)} days)")

    return splits
