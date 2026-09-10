# OceanXRay — Deploy Folder

This folder is a self-contained inference package extracted from your
trained checkpoints. It does **not** need the data pipeline (xarray,
netCDF4, dask, etc.) — just PyTorch + FastAPI.

```
deploy/
├── checkpoints/
│   ├── cnn_best.pt      # main/locked model (3x3 patch CNN encoder + MLP decoder)
│   └── mlp_best.pt      # Stage-2 MVP baseline
├── src/
│   ├── cnn_model.py     # architecture classes (copied from your src/)
│   └── mlp_model.py
├── predictor.py         # loads checkpoints + normalization, does pre/post-processing
├── app.py               # FastAPI server (/health, /predict/mlp, /predict/cnn)
├── test_predict.py       # smoke test, no server needed
├── requirements.txt
└── Dockerfile
```

## ⚠️ One file is missing — you MUST add it before this gives real results

`normalization_stats.json` is **not included**. It's the train-split
mean/std used to normalize inputs and denormalize the model's output
back into °C. It was computed in your Colab run and copied to:

```
/content/drive/MyDrive/oceanxray/data/processed/normalization_stats.json
```

Download that exact file from your Drive and drop it directly into
this `deploy/` folder (next to `predictor.py`). Until you do,
`predictor.py` runs in an **identity-stats fallback** so the code
still executes end-to-end, but the numbers it returns are not real
temperatures — every response includes `"stats_used": false` so this
is impossible to miss.

## How I tested it (and what "testing output" I'm giving you)

I ran `test_predict.py` and hit the FastAPI server directly, both with
identity-fallback stats since your zip had no `data/`. Results:

- Both `cnn_best.pt` and `mlp_best.pt` load cleanly into `PatchCNN` /
  `PointMLP` with **zero shape mismatches** — the checkpoints are
  self-describing (channels, patch size, hidden dims all stored
  inside the `.pt` file), so this part of your pipeline is solid.
- MLP forward pass: input `(1, 5)` → output `(1, 15)` ✅
- CNN forward pass: input `(1, 5, 3, 3)` → output `(1, 15)` ✅
- `uvicorn app:app` boots, `/health` returns 200, `/predict/mlp` returns
  a 200 with a 15-value profile ✅
- Once you drop in the real `normalization_stats.json`, re-run
  `python test_predict.py` — if `stats_used` still reads `false`, the
  file isn't in the right place; if it's `true`, the profile numbers
  you see are your model's real output in °C.

## Run it

```bash
cd deploy
pip install -r requirements.txt
python test_predict.py          # CLI smoke test, no server
uvicorn app:app --reload        # or run the real server
```

`POST /predict/mlp` body:
```json
{"values": {"sst": 29.2, "ssh": 0.10, "u_current": 0.15, "v_current": -0.05, "v_wind": 4.0}}
```

`POST /predict/cnn` body: same 5 keys, each a 3x3 nested list instead
of a single float (a real 3x3 spatial patch around the grid cell).

## Deploy it anywhere

`Dockerfile` is included — `docker build -t oceanxray . && docker run -p 8000:8000 oceanxray`
works on Render, Railway, Fly.io, Hugging Face Spaces (Docker SDK), or
any VM. No GPU needed — both models are tiny (MLP: 5.5K params, CNN: 15K params).

---

## Things worth fixing honestly before the next round (not hidden, just flagged)

I went through the full zip + architecture doc + Colab script. Two
things stood out as genuine weak points a judge is likely to probe —
better you know first:

1. **MLP baseline barely beats climatology, and stopped after epoch 1.**
   `training_config.json` → `best_epoch: 1`. On the val split, MLP RMSE
   (1.206°C) is only marginally better than climatology (1.216°C) —
   it technically clears your own "hard gate" but by a hair. This
   usually means the LR is too high / early-stopping patience is too
   tight, so it converges before actually learning much beyond the
   mean. The CNN doesn't have this problem (best_epoch: 5, clearly
   beats both) — so your headline result (CNN > MLP > climatology)
   still holds, but if asked "why does the MLP look so weak," the
   honest answer is "under-trained," and it's an easy fix (lower LR
   or raise patience) before the final round.

2. **ARGO independent validation scored 0 profiles.**
   `results/argo/validation_summary.json` → `profiles_final_scored: 0`,
   and `data_source: "synthetic_fixture"` — your own code already
   labels this `<-- SYNTHETIC, NOT REAL RESULTS` when it runs. This is
   the "scientific credibility" checkpoint in your architecture doc
   (§8.2), and right now it hasn't actually run against real ARGO
   data or produced a single matched, scored profile. That's a bigger
   gap than the MLP issue — worth prioritizing if you can fit a real
   ARGO download + a wider space/time matching tolerance before
   judging, since this is called out as a MUST HAVE in your own
   Definition of Done (§14).

Neither of these breaks the deploy package above — the deploy folder
works regardless, since it just serves the already-trained CNN. But
I'd rather you hear about these two from me now than from a judge
asking "why does ARGO validation say n=0."
