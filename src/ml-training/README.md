# OceanXRay — Day 1: Preprocessing Pipeline

Tested end-to-end (with synthetic data mimicking real product quirks — different
coord names, missing SSS, hourly winds, non-matching GLORYS depth levels,
etc.) and confirmed working. See "How this was tested" at the bottom.

## 1. Folder structure

```
oceanxray_project/
├── config.py                  # ALL settings: paths, variable/coord name candidates,
│                               # region, target depths, split config, mask threshold
├── requirements.txt
├── run_pipeline.py             # MAIN ENTRY POINT — run this
├── make_synthetic_data.py      # optional: generates small fake data to smoke-test
│                               # the pipeline before you plug in real downloads
├── data/
│   ├── raw/
│   │   ├── sst/                # put your SST .nc file(s) here
│   │   ├── sss/                # optional — leave empty if you don't have SSS
│   │   ├── ssh/                # SSH / SLA
│   │   ├── currents/           # whichever current product you actually have
│   │   ├── winds/               # U/V wind
│   │   └── glorys/             # GLORYS 3D temperature (thetao)
│   └── processed/              # OUTPUT — created automatically
│       ├── ml_ready_dataset.nc
│       ├── normalization_stats.json
│       ├── metadata.json
│       └── sanity_plots/
└── src/
    ├── coord_utils.py          # detects lat/lon/time/depth + var names, never hard-coded
    ├── loaders.py               # one loader per source, SSS optional, current product not assumed
    ├── inspect_data.py          # prints full diagnostic report per dataset
    ├── region.py                 # crop to 5N-30N, 45E-105E
    ├── temporal_align.py         # inspect native freq -> align to common daily calendar
    ├── regrid.py                  # horizontal regrid to 0.25 deg (xESMF or documented fallback)
    ├── glorys_vertical.py         # native GLORYS depths -> 15 target depths
    ├── masking.py                 # land/sea mask (explicit if available, else fallback)
    ├── splitting.py                # whole-year or seasonal-block train/val/test split
    ├── normalization.py            # train-only stats, per-depth for targets
    ├── assemble.py                  # merge inputs+target, save ML-ready dataset + metadata
    ├── patch_extraction.py          # reusable 3x3 patch-center finder (no CNN yet)
    └── sanity_checks.py             # all the required checks + 3 diagnostic plots
```

## 2. Setup

```bash
cd oceanxray_project
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

xESMF is optional. If `conda install -c conda-forge esmf xesmf` isn't feasible
in your 5-day window, the pipeline auto-detects its absence and uses a
documented xarray-based fallback (coarsen + linear interp) instead — it will
NOT crash, just print a clear limitation notice.

## 3. Point it at your real data

Edit `config.py`:
- `RAW_FILES`: glob patterns to your actual downloaded files.
- If `inspect_data.py`'s output shows "VERIFY REQUIRED: could not detect
  variable/coordinate X", add the real name you see in the printed
  `data_vars`/`coords` list to `VARIABLE_CANDIDATES` / `COORD_CANDIDATES`.
- `SPLIT_CONFIG`: once you know how many calendar years you actually have,
  set `train_years`/`val_years`/`test_years` explicitly (or leave `mode:
  "auto"` and it'll pick sensible defaults and print them for you to confirm).

## 4. Run

```bash
python run_pipeline.py
```

Optional smoke test first (no real data needed):
```bash
python make_synthetic_data.py   # writes small fake files into data/raw/
python run_pipeline.py          # confirms the whole chain runs before you
                                 # spend time debugging against real files
# then delete data/raw/*/*.nc and put your real files in before the real run
```

## 5. Expected output (in `data/processed/`)

- `ml_ready_dataset.nc` — NetCDF with normalized input variables
  `(time, lat, lon)` each, `temperature_target (time, depth, lat, lon)`
  normalized per depth, and `ocean_mask (lat, lon)`.
- `normalization_stats.json` — per-variable and per-depth train mean/std,
  needed again at inference time later.
- `metadata.json` — region, grid resolution, target depths, channels
  actually used, split date ranges, split config used.
- `sanity_plots/` — 3 PNGs: one surface variable map, one GLORYS target
  depth map, one example 15-depth profile.

## 6. Sanity checks printed at the end

Latitude/longitude range, grid spacing ≈0.25°, daily time spacing, depth
count = 15, depth values, number of input channels actually available,
train/val/test date ranges, NaN fraction per variable, per-variable/per-depth
train mean/std, shapes of every input and the target, and confirmation that
patch centers never touch masked land.

## 7. Known limitations to keep in mind (explicitly logged by the code, not hidden)

- **Regridding fallback**: without xESMF, horizontal regridding is
  block-mean coarsening + linear interpolation rather than a true
  conservative-area regrid. Fine for an MVP; install xESMF before final
  numbers if time allows.
- **GLORYS shallow-depth extrapolation**: if your target depth 0m is
  shallower than GLORYS's shallowest native level (common — many products'
  top cell is centered around 0.5m), the code linearly extrapolates and
  prints a note. This is standard practice for such a small gap.
- **Coarser-than-daily inputs** (e.g. some altimetry products): forward-filled
  onto the daily calendar — clearly logged, not a silent assumption.
- **Fallback land/sea mask**: if no product ships an explicit mask, ocean is
  defined as "SST valid ≥90% of training days" — printed explicitly so
  you can change `MASK_FALLBACK_MIN_VALID_FRACTION` in config.py if needed.
- **Memory**: for the FULL region at 0.25° over several years, GLORYS's
  4D array (time × 15 depths × lat × lon) can get large. If you hit memory
  errors, either process fewer years first or add `chunks="auto"` to the
  `xr.open_dataset` calls in `loaders.py` for lazy/dask loading — the code
  is structured so that's a one-line change per loader.

## 8. How this was tested

Before handing this over, I generated small synthetic NetCDF files that
deliberately mimic real-world messiness — different coordinate names per
product (`lat`/`latitude`, `lon`/`longitude`), SSS completely absent,
hourly wind data needing resampling to daily, a fake land mass to test
masking, and GLORYS native depth levels that do NOT include 0m or match
the 15 target depths — and ran the full pipeline against it end-to-end.
It ran clean and every sanity check passed. One real bug this caught and
fixed: linear interpolation to a target depth shallower than GLORYS's
shallowest native level was producing NaN; the code now extrapolates for
that specific edge case and logs why.

---

# DAY 2 — Climatology Baseline + Point-based MLP

Day 2 builds on Day 1's `data/processed/ml_ready_dataset.nc` and adds:

```
src/
├── climatology.py    # Stage 1: climatology baseline
├── mlp_dataset.py     # turns the grid into flat point samples for the MLP
├── mlp_model.py         # the PointMLP architecture
├── train_mlp.py           # training loop, early stopping, checkpointing
├── metrics.py               # RMSE/MAE/Bias/Correlation, overall + per-depth
└── evaluation.py              # runs both models, saves results, makes plots, prints the gate

run_day2.py            # MAIN ENTRY POINT for Day 2
```

## Run

```bash
python run_pipeline.py   # Day 1 first, if not already done
python run_day2.py       # Day 2
```

## Output

```
models/
├── mlp_best.pt              # best checkpoint (by validation loss)
└── training_config.json     # hyperparameters, channels, depths, best epoch

results/
├── climatology_metrics.json # overall + per-depth, normalized AND degC space
├── mlp_metrics.json
├── comparison.json          # climatology vs MLP RMSE/MAE + mlp_beats_climatology flag
└── plots/
    ├── 01_rmse_comparison.png
    ├── 02_depthwise_rmse.png
    ├── 03_example_profile.png
    ├── 04_training_curve.png
    └── 05_scatter_mlp.png
```

At the end, the script prints the **DAY 2 GATE**: climatology RMSE vs MLP
RMSE (in °C) and a clear PASSED/FAILED verdict. Per the brief, if the MLP
does not beat climatology, do not tune the model just to force a pass —
investigate the data/training first.

## Day 2 design notes / limitations (explicitly logged by the code)

- **Climatology mode is picked automatically**: if training data spans
  ≥2 distinct calendar years, a **day-of-year climatology** is fit
  (mean profile per day-of-year per lat/lon, averaged across years).
  If training only covers a single year — which includes the
  `seasonal_block` split case, where train/val/test are disjoint
  month ranges — day-of-year climatology cannot generalize to months
  it never saw, so the code falls back to a **pure spatial climatology**
  (same predicted profile every day, varies only by location). This is
  printed explicitly, not silent. With real multi-year data and
  `whole_year` split mode, you'll get the day-of-year version.
- **Point samples**: a sample is one `(time, lat, lon)` triplet where the
  cell is ocean AND every input channel AND every one of the 15 target
  depths is finite there. Only valid samples are gathered into compact
  `(n_samples, n_channels)` / `(n_samples, 15)` arrays — the full grid
  is never duplicated wastefully.
- **Split is reconstructed, not re-invented**: `run_day2.py` calls the
  exact same `src.splitting.compute_split()` Day 1 used, applied to the
  saved dataset's actual time coordinate — so Day 2 always uses exactly
  Day 1's train/val/test dates, never a fresh random split.
- **Metrics in both spaces**: normalized-space metrics are saved for
  debugging; the ones that matter (and the ones used for the gate) are
  inverse-transformed to °C using Day 1's saved per-depth
  `normalization_stats.json`.
- **Gate uses VALIDATION, final numbers reported from TEST**: per the
  locked architecture's hard gate ("MLP measurably beats climatology on
  the validation split"), `run_day2.py` evaluates on val first — that
  result drives the printed DAY 2 GATE and is saved to
  `results/comparison_val.json` (+ `climatology_metrics_val.json`,
  `mlp_metrics_val.json`, and the 5 plots). It then evaluates on test
  exactly once, purely for an honest final report, saved to
  `results/comparison_test.json` (+ `*_test.json`) — test never
  influences the gate decision.

---

# DAY 3 — Stage 3 CNN (encoder + embedding + MLP decoder)

Day 3 builds on Day 1's `ml_ready_dataset.nc` AND Day 2's trained MLP
checkpoint (`models/mlp_best.pt`, REUSED not retrained) and adds:

```
src/
├── cnn_dataset.py     # 3x3 patch samples with strict time-varying validity
├── cnn_model.py          # CNNEncoder -> embedding -> MLPDecoder (model.encode(x) exposes the embedding)
├── train_cnn.py             # training loop, early stopping on VAL, checkpointing
└── cnn_evaluation.py           # 3-way (climatology/MLP/CNN) comparison, embeddings, plots

run_day3.py             # MAIN ENTRY POINT for Day 3
```

## Run

```bash
python run_pipeline.py   # Day 1 first, if not already done
python run_day2.py       # Day 2 (trains the MLP checkpoint Day 3 reuses)
python run_day3.py       # Day 3
```

## Output

```
models/
├── cnn_best.pt                  # best CNN checkpoint (by validation loss)
└── cnn_training_config.json     # hyperparameters, channels, depths, patch size, best epoch

results/
├── climatology_metrics_val.json / _test.json
├── mlp_metrics_val.json / _test.json
├── cnn_metrics_val.json / _test.json
├── comparison_3way_val.json     # drives the STAGE 3 result (validation-based)
├── comparison_3way_test.json    # final reporting only
├── embeddings.npz               # CNN embeddings + time/month/lat/lon metadata
│                                 # for train+val+test, ready for later PCA/t-SNE/UMAP
└── plots/
    ├── 01_rmse_comparison_3way.png
    ├── 02_depthwise_rmse_3way.png
    ├── 03_depthwise_mae_3way.png
    ├── 04_example_profile_3way.png     # target vs climatology vs MLP vs CNN
    ├── 05_cnn_training_curve.png
    ├── 06_scatter_cnn.png
    └── 07_embedding_pca.png            # optional bonus sanity plot (needs scikit-learn)
```

At the end, the script prints **STAGE 3 CNN RESULT**: climatology vs MLP
vs CNN RMSE on validation (this is what "CNN beats MLP" / "CNN beats
climatology" is judged on) and again on test (reporting only).

## Day 3 design notes / limitations (explicitly logged by the code)

- **Fair 3-way comparison**: the CNN can only use a point if its FULL
  3x3 neighborhood is valid — a stricter requirement than the point-MLP's
  own single-pixel validity rule from Day 2. Evaluating each model on its
  own separately-computed valid set would silently compare them on
  different points. Instead, `cnn_evaluation.py` always evaluates all
  three models on the CNN's (stricter) valid sample set — the MLP's
  point features for those exact samples are derived directly from the
  CNN patch's own center pixel (mathematically identical to what Day 2's
  gather would produce, since `cnn_dataset.py` explicitly asserts the
  patch center equals the point value at that time/lat/lon).
- **The Day 2 MLP is reused, not retrained**: `run_day3.py` loads
  `models/mlp_best.pt` and fails loudly if its saved channel list doesn't
  match the current Day 1 data (protects against silent misalignment).
- **No land/NaN patch ever reaches the model**: `cnn_dataset.py` checks,
  per exact `(time, lat, lon)`, that every one of the 9 patch cells is
  finite for every channel (not just the center, and not just
  statically-ocean — actual per-day finiteness). Failing that, the
  sample is excluded, never zero-filled. An explicit alignment assertion
  confirms the patch center always equals the point value at that
  location.
- **Embedding**: `model.encode(x)` returns the CNN's learned embedding
  only (never the final 15-depth output — `model(x)`/`model.forward(x)`
  for that). `results/embeddings.npz` holds embeddings + time/month/
  lat/lon for every train+val+test sample, ready for later PCA/t-SNE/
  UMAP colored by region/season — that deeper analysis is deferred; only
  a quick PCA-by-month sanity plot is included now.
- **Validation drives all decisions, test is final-report-only** — same
  audited rule as Day 2, applied identically here.

---

# STAGE 4 — Independent ARGO validation

This stage evaluates the **final, already-trained** CNN checkpoint against
**Argo float** observations — a data source with **no relationship at all**
to GLORYS (the training/reference target). It never touches training,
normalization, tuning, model selection, or early stopping. It only runs
after the checkpoint from Day 3 is locked.

## Run

```bash
# one-time, needs Day 1's ml_ready_dataset.nc to already exist:
python make_synthetic_argo_fixture.py

# main entry point:
python run_argo_validation.py
```

`run_argo_validation.py` tries **real** Argo data first
(`argo_config.ARGO_REAL_FILES_GLOB`); if none is present it automatically
falls back to the synthetic fixture and labels every output
`"data_source": "synthetic_fixture"` — it never fabricates real-looking
numbers.

## ARGO source (real data — not yet fetched in this environment)

This sandbox has no network route to any Argo data center, so real ARGO
data has **not** been downloaded or processed here. To run this stage for
real:

- **Recommended**: the [`argopy`](https://argopy.readthedocs.io) Python
  package, e.g.
  ```python
  from argopy import DataFetcher
  ds = DataFetcher().region(
      [45, 105, 5, 30, 0, 1000, '<start-date>', '<end-date>']
  ).to_xarray()
  ```
  matching the region `5N-30N, 45E-105E` and the actual time coverage of
  your `ml_ready_dataset.nc` (check `metadata.json`) — don't fetch more
  than that.
- **Alternative**: direct GDAC mirrors (ifremer / usgodae). Use the
  monthly global index file first (e.g. `ar_index_global_prof.txt`) to
  find which profile files actually fall in the region/time window
  *before* downloading anything — never bulk-download the global archive.
- Save whatever you obtain as NetCDF file(s) under
  `data/raw_argo/real/` (or point `argo_config.ARGO_REAL_FILES_GLOB` at
  wherever you put them). Expected variables:
  `LATITUDE, LONGITUDE, TIME (or JULD), PRES, TEMP, PRES_QC, TEMP_QC,
  POSITION_QC, JULD_QC` on an `(N_PROF, N_LEVELS)` grid — see the full
  format docstring at the top of `src/argo_loader.py`.

## Processing

- **QC**: only Argo flags **1 (good)** and **2 (probably good)** are
  accepted, for `TEMP_QC`/`PRES_QC` (per level), and for
  `POSITION_QC`/`JULD_QC` (per profile — a profile with an untrustworthy
  fix or timestamp is dropped entirely, not partially trusted).
  Everything else becomes `NaN`, never silently kept.
- **Depth interpolation**: each QC-passed profile is linearly interpolated
  onto the same 15 locked target depths the CNN predicts. A target depth
  **outside** that profile's own observed pressure range is left `NaN` —
  **never extrapolated**. Profiles with fewer than
  `argo_config.ARGO_MIN_VALID_OBS_FOR_INTERP` (default 2) valid levels are
  skipped entirely.
- **Depth vs. pressure**: `PRES` (dbar) is used directly as depth (m),
  without a TEOS-10 pressure→depth correction. At ≤1000 m in this
  latitude range the difference is small relative to the 15 target-depth
  spacing, but it's a known approximation — see
  `argo_config.ARGO_DEPTH_FROM_PRESSURE_APPROX` if you want to swap in
  `gsw.z_from_p` later.

## Matching tolerances (configurable in `argo_config.py`)

- **Temporal**: nearest available grid day within **±1 day** (the model
  consumes daily-resampled fields; an Argo ascent takes hours, so
  same-day is the natural target, with 1 day of slack for occasional
  same-day data gaps).
- **Spatial**: nearest grid cell, but only accepted if the true
  great-circle distance to that cell's center is **≤ 20 km** — roughly
  half the 0.25° grid diagonal, i.e. a profile is never matched to a grid
  cell that's actually a full cell-width away. On this grid, this means
  a genuine "too far" rejection can only occur for profiles that fall
  **outside** the model's actual grid coverage (confirmed in testing).
- Every match also requires the full 3×3 neighborhood around the matched
  cell to be (a) inside the grid (not an edge cell) and (b) all-ocean,
  **and** to actually be finite on that exact day (a statically-ocean
  patch can still have a transient data gap) — identical rule to what
  training samples had to satisfy.
- Exact matched/rejected diagnostics (distance, time offset, reject
  reason) for every input profile are saved to
  `results/argo/matched_profiles.csv`.

## Regional case study

Arabian Sea vs. Bay of Bengal is split at **77°E** (an explicit,
documented rectangular simplification — the real coastline/Sri-Lanka
boundary is more complex, but a straight reproducible split beats an
implicit one). A region is skipped (not forced) if it has fewer than
`argo_config.REGIONAL_MIN_MATCHED_PROFILES` (default 5) matched profiles.

## No data leakage (Step 8)

`run_argo_validation.py` checks this **first**, before loading any data:
`src/argo_validation.verify_no_argo_leakage()` scans every Day 1-3
training script/module's source for any mention of "argo" and refuses to
proceed if found, plus asserts its own module never imports the
training-only normalization-stat functions. ARGO is loaded, matched, and
scored **only** against the already-locked `models/cnn_best.pt` — no
gradient updates happen anywhere in this stage.

## Output

```
results/argo/
  argo_metrics.json          overall + per-depth RMSE/MAE/Bias/Corr (CNN vs ARGO)
  argo_depth_metrics.csv     same, as a table
  argo_regional_metrics.json Arabian Sea vs Bay of Bengal (or "skipped" + why)
  matched_profiles.csv       every input profile: matched?, reject reason,
                              distance_km, time_offset_days
  validation_summary.json    everything above + tolerances + QC rule used +
                              data_source (real_argo / synthetic_fixture)
  plots/
    argo_matched_locations.png
    argo_depthwise_rmse.png
    argo_depthwise_mae.png
    argo_example_profiles.png
```

`argo_metrics.json` explicitly notes it is **CNN vs ARGO (independent
observations)**, distinct from **CNN vs GLORYS**
(`results/comparison_3way_test.json` from `run_day3.py`, the
training/reference target) — the two should never be conflated.

## Testing approach (Step 9, done for real — not just written)

Before touching real data: `make_synthetic_argo_fixture.py` builds 8
synthetic profiles anchored to the *actual* Day 1 grid/time coverage,
with deliberately known outcomes — 2 full-depth matches, 1 missing-deep
match, 1 QC-partially-rejected match, and 4 profiles that must each fail
matching for a *different* documented reason (too far spatially, too far
temporally, lands on a masked/edge cell, bad `POSITION_QC`). The full
pipeline was run against a real (if tiny, smoke-test-scale) trained CNN
checkpoint and every expected outcome was verified to actually occur.

**2 real bugs were found and fixed during this testing**, not just
written and assumed correct:
1. The Step 8 leakage self-check originally searched its own source for
   the literal strings `"compute_input_stats"` / `"compute_target_stats_
   per_depth"` — which trivially matched the list of forbidden names
   itself, a guaranteed false positive. Fixed to check for actual
   `import` usage instead.
2. The synthetic fixture's "should fail — too far spatially" test case
   used a fixed offset that, by coincidence, landed exactly on a real
   grid point (0.25° grid ⇒ effectively nothing *inside* the domain can
   exceed the 20 km tolerance — the tolerance was deliberately set to
   about half the grid diagonal). Fixed by placing that test profile
   outside the grid's actual coverage instead, which is also the more
   realistic version of "too far" for this scenario.

## Known limitations

- No `gsw`-based pressure→depth correction (see above).
- Arabian Sea / Bay of Bengal split is a straight-meridian simplification.
- Real ARGO access/processing has not been exercised in this sandbox —
  only the synthetic fixture has. The loader, QC, interpolation,
  matching, and metrics code paths are identical either way, but the
  *real* run should be treated as un-smoke-tested against actual GDAC
  file quirks (e.g. inconsistent QC variable dtypes across floats) until
  it's actually run once against a real file.

---

# STAGE 5 — Deeper embedding analysis, regional case study, ablation

Strengthens the scientific/ML story without touching the locked
architecture. Nothing here retrains the CNN — it's loaded from
`models/cnn_best.pt` and reused throughout.

## Run

```bash
python run_stage5.py
```

Needs Day 1-3 outputs already present (`ml_ready_dataset.nc`,
`cnn_best.pt`, `results/embeddings.npz`). Stage 4's ARGO regional
results are used if present (`results/argo/argo_regional_metrics.json`)
but Stage 5 will still run without them — the ARGO section of the
regional report is simply marked unavailable.

## Part A — embedding analysis (`src/embedding_analysis.py`)

- Loads `results/embeddings.npz` (Day 3's `model.encode()` output +
  time/lat/lon/split for every train+val+test sample).
- Attaches **region** (reusing Stage 4's 77°E Arabian-Sea/Bay-of-Bengal
  split — never redefined here) and **season** (North-Indian-Ocean
  monsoon convention: winter monsoon DJF, pre-monsoon MAM, summer
  monsoon JJA, post-monsoon SON — not generic calendar quarters), plus
  the **normalized surface SST** the CNN actually saw at that point.
- **PCA is the primary method** (`sklearn.decomposition.PCA`, 2
  components). **t-SNE is optional** and only run on a further
  subsample (default ≤8,000 points) specifically because it's only
  practical at that scale — never run on the full dataset.
- Never loads the full multi-million-point set for plotting: a
  documented random subsample (default ≤20,000 points, seed=42) is
  used for every visualization.
- Outputs to `results/embedding/`: `pca_coordinates.npz`,
  `embedding_metadata.csv` (time/lat/lon/region/season/SST/PC1/PC2 per
  point), `explained_variance.json`, and
  `plots/embedding_{pca,tsne}_by_{region,season,sst}.png`.

**On "do the plots show clusters"**: this is answered by looking at the
actual saved plots for the run you did, not asserted in advance here —
see the smoke-test result reported in this session's summary for what
was actually observed on synthetic data (a real-data run may look
different).

## Part B — Arabian Sea vs Bay of Bengal case study (`src/regional_analysis.py`)

Same 77°E split as Stage 4, reused via
`src.argo_validation.regional_split()` — never redefined.

Two genuinely separate analyses, kept explicit rather than merged:
1. **GLORYS-based** (new here): CNN vs GLORYS, split by region, on the
   TEST split only. `results/regional/glorys_regional_metrics.json` +
   `plots/regional_rmse_bar.png`, `regional_depthwise_rmse.png`,
   `regional_representative_profiles.png`.
2. **ARGO-based** (not recomputed): Stage 4's own
   `argo_regional_metrics.json` is read as-is and reported alongside,
   explicitly labeled as independent-observation validation, separate
   from (1). If a region was skipped in Stage 4 (too few matched
   profiles), it stays skipped here too — never forced.

`results/regional/regional_summary.json` holds both, clearly separated
under `glorys_based_regional_analysis` and
`argo_independent_regional_validation` keys.

## Part C/D — spatial-context ablation (`src/ablation.py`, `src/train_ablation.py`)

**Model A — Center MLP (1x1 context)**: `src.mlp_model.PointMLP` (the
same architecture as Day 2), trained **fresh** on the CNN's own center
pixel.

**Model B — CNN (3x3 context)**: `models/cnn_best.pt`, reused as-is,
never retrained.

**Why Model A isn't just Day 2's existing `mlp_best.pt`**: Day 2's MLP
was trained on its own, looser valid-sample set (only the center pixel
itself needs to be valid). The CNN requires a stricter condition (the
full 3x3 neighborhood must be valid). Comparing Day 2's MLP against the
CNN would confound "spatial context vs. none" with "different training
samples" — not a fair ablation. So Model A is trained fresh on exactly
the CNN's own (stricter) sample set, with the same channels, target,
normalization, temporal splits, and — critically — the same
epochs/batch-size/lr/patience the CNN itself used (read back from
`models/cnn_training_config.json`, not re-guessed), with the same
validation-based checkpoint selection rule. Test is never touched until
final reporting for either model. Checkpoint saved separately to
`models/ablation/mlp_best.pt` so Day 2's own checkpoint is untouched.

Outputs to `results/ablation/`: `ablation_table.csv` (the Model/Context/
Val_RMSE/Test_RMSE/MAE/Bias/Correlation/N_Params table from the brief),
`ablation_depthwise.csv`, `ablation_summary.json`, and
`plots/ablation_rmse_bar.png`, `ablation_depthwise_rmse.png`.

## Part G — no data leakage

- `src.argo_validation.verify_no_argo_leakage()` is re-run here too
  (Stage 5 must not reintroduce ARGO leakage into training either).
- A new structural check, `verify_no_stage5_test_leakage()`, uses
  Python signature introspection to confirm `train_center_mlp()` has no
  test-split parameter at all — not just "we promise not to use it",
  but "it's not even possible to pass it in by mistake".
- Ablation, regional, and embedding analysis all reuse the SAME
  training-only normalization stats (`normalization_stats.json`, loaded
  read-only) - nothing here recomputes normalization.

## Known limitations

- PCA/t-SNE describe the CNN's *learned embedding* space, not the raw
  physical inputs — a lack of visible clustering there doesn't
  necessarily mean the underlying physical fields don't differ by
  region/season, only that the model's 32-D bottleneck doesn't obviously
  separate them in 2 components.
- The regional 77°E split remains a rectangular simplification (see
  Stage 4's own limitations section).
- The ablation isolates exactly one variable (1x1 vs 3x3 input) by
  construction, but only at a *single* patch size — 5x5/9x9 patches are
  explicitly out of scope per the brief.
- Smoke-test results (tiny synthetic data, 6 epochs) should not be read
  as a real scientific conclusion about spatial context — see this
  session's reported ablation numbers for what was actually observed,
  clearly labeled as a smoke test.

