
import glob, sys
import xarray as xr
from pathlib import Path
from config import RAW_FILES
from src.coord_utils import standardize_coord_names, normalize_longitude_convention

_ENGINES = ["netcdf4", "h5netcdf", "scipy"]

def _resolve_files(pattern):
    return sorted(glob.glob(str(pattern)))

def _open_one(filepath):
    for engine in _ENGINES:
        for dt in [True, False]:
            try:
                return xr.open_dataset(filepath, engine=engine, decode_times=dt)
            except Exception:
                pass
    raise RuntimeError(f"Cannot open {filepath} with any engine")

def _normalise_time(ds):
    """Rename valid_time -> time if needed."""
    if "valid_time" in ds.dims and "time" not in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    elif "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.assign_coords(time=ds["valid_time"]).drop_vars("valid_time")
    return ds

def _open(pattern, name, required=True):
    files = _resolve_files(pattern)
    if not files:
        msg = f"[loaders] No files found for '{name}' at pattern: {pattern}"
        if required:
            raise FileNotFoundError(msg)
        print(msg + " -- treating as UNAVAILABLE (optional).")
        return None
    print(f"[loaders] Opening '{name}': {len(files)} file(s) matched.")

    if len(files) == 1:
        ds = _open_one(files[0])
    else:
        # Try open_mfdataset fast paths first
        ds = None
        for engine in _ENGINES:
            for kw in [dict(combine="by_coords"), dict(combine="nested", concat_dim="time"),
                       dict(combine="nested", concat_dim="valid_time")]:
                try:
                    ds = xr.open_mfdataset(files, engine=engine, parallel=False, **kw)
                    break
                except Exception:
                    pass
            if ds is not None:
                break

        if ds is None:
            # Per-file concat fallback
            print(f"[loaders] open_mfdataset failed — falling back to per-file concat...")
            datasets = [_normalise_time(_open_one(f)) for f in files]
            ds = xr.concat(datasets, dim="time")

    ds = _normalise_time(ds)
    ds = standardize_coord_names(ds)
    ds = normalize_longitude_convention(ds)
    return ds

def _load_winds_split(files):
    """Handle ERA5 ZIPs extracted as 1-variable-per-file (u OR v, not both)."""
    u_ds, v_ds = [], []
    for f in files:
        ds = _normalise_time(_open_one(f))
        varnames = list(ds.data_vars)
        has_u = any("u" in v.lower() and ("10" in v or "component" in v) for v in varnames)
        has_v = any("v" in v.lower() and ("10" in v or "component" in v) for v in varnames)
        if has_u and has_v:
            return None   # files have both — caller should use normal path
        elif has_u:
            # Rename to canonical u10
            ukey = next(v for v in varnames if "u" in v.lower())
            u_ds.append(ds.rename({ukey: "u10"}))
        elif has_v:
            vkey = next(v for v in varnames if "v" in v.lower())
            v_ds.append(ds.rename({vkey: "v10"}))
    # Concat each component, then outer-merge
    parts = []
    if u_ds: parts.append(xr.concat(u_ds, dim="time"))
    if v_ds: parts.append(xr.concat(v_ds, dim="time"))
    if not parts:
        raise RuntimeError("[loaders] No usable wind variables found")
    merged = xr.merge(parts, join="outer") if len(parts) > 1 else parts[0]
    u_months = len(u_ds); v_months = len(v_ds)
    print(f"[loaders] Winds: {u_months} files with u10, {v_months} files with v10")
    if u_months != v_months:
        print(f"[loaders] WARNING: u10/v10 coverage mismatch "
              f"({u_months} vs {v_months} months). Re-run the re-download cell for complete data.")
    return merged

def load_winds():
    files = _resolve_files(RAW_FILES["winds"])
    if not files:
        raise FileNotFoundError(f"No wind files at {RAW_FILES['winds']}")
    print(f"[loaders] Opening 'Winds': {len(files)} file(s) matched.")

    # Peek at first file to check if it has both u+v
    ds0 = _open_one(files[0])
    varnames = list(ds0.data_vars)
    ds0.close()
    split = not (any("u" in v.lower() for v in varnames) and
                 any("v" in v.lower() for v in varnames))

    if split:
        print("[loaders] Detected split u/v files (CDS ZIP extraction artefact) — merging...")
        ds = _load_winds_split(files)
    else:
        ds = _open(RAW_FILES["winds"], "Winds", required=True)
        return ds  # already standardised inside _open

    ds = standardize_coord_names(ds)
    ds = normalize_longitude_convention(ds)
    return ds

def load_sst():     return _open(RAW_FILES["sst"],     "SST",     required=True)
def load_sss():     return _open(RAW_FILES["sss"],     "SSS",     required=False)
def load_ssh():     return _open(RAW_FILES["ssh"],     "SSH/SLA", required=True)
def load_glorys():  return _open(RAW_FILES["glorys"],  "GLORYS",  required=True)


def _open_currents_chunked(pattern):
    """Open currents with Dask chunks to avoid OOM during reindex.
    Currents (1461x1x301x721 x 2 vars) = ~2.6 GB — needs lazy loading."""
    files = _resolve_files(pattern)
    if not files:
        raise FileNotFoundError(f"No currents files at {pattern}")
    print(f"[loaders] Opening 'Currents': {len(files)} file(s) matched.")
    chunks = {"time": 60}  # 60-day chunks ~ manageable
    for engine in _ENGINES:
        for combine in [dict(combine="by_coords"), dict(combine="nested", concat_dim="time")]:
            try:
                ds = xr.open_mfdataset(files, engine=engine, parallel=False,
                                       chunks=chunks, **combine)
                ds = _normalise_time(ds)
                ds = standardize_coord_names(ds)
                ds = normalize_longitude_convention(ds)
                return ds
            except Exception:
                pass
    # Fallback: normal open (not chunked) if mfdataset fails
    print("[loaders] Currents: dask open failed, falling back to in-memory", flush=True)
    return _open(pattern, "Currents", required=True)

def load_currents():
    ds = _open_currents_chunked(RAW_FILES["currents"])
    if ds is not None:
        title = ds.attrs.get("title") or ds.attrs.get("source") or "unknown"
        print(f"[loaders] Currents: '{title}'.")
    return ds

def load_all():
    print("\n" + "="*70 + "\nLOADING RAW DATASETS\n" + "="*70)
    datasets = {
        "sst": load_sst(), "sss": load_sss(), "ssh": load_ssh(),
        "currents": load_currents(), "winds": load_winds(), "glorys": load_glorys(),
    }
    available = [k for k,v in datasets.items() if v is not None]
    missing   = [k for k,v in datasets.items() if v is None]
    print(f"\n[loaders] Available: {available}")
    print(f"[loaders] Missing/optional-and-absent: {missing}")
    return datasets
