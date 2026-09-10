"""
embedding_analysis.py
========================
Stage: STAGE 5, PART A — deeper embedding analysis.

Loads Day 3's results/embeddings.npz (CNN model.encode() output +
time/lat/lon/split metadata for every train+val+test sample), attaches
region (Arabian Sea / Bay of Bengal, reusing Stage 4's 77E definition
— never redefined here) and season labels, optionally attaches surface
SST state, and runs PCA (primary method) plus an optional t-SNE on a
subsample (only because a subsample makes it fast enough to be
practical — not run on the full multi-million-point set).

IMPORTANT: this module only ever produces a visualization + numbers.
Whether "clusters exist" is a statement about what the plot actually
shows, decided by whoever reads plot_*.png — this module does not
assert clustering exists a priori anywhere in its code or printed
output.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.argo_validation import regional_split  # reused, not redefined

# North Indian Ocean seasonal convention (meteorologically standard for
# this basin, not a generic calendar quarter split):
#   DJF = winter monsoon (NE monsoon), MAM = pre-monsoon,
#   JJA = summer monsoon (SW monsoon), SON = post-monsoon.
_SEASON_BY_MONTH = {
    12: "winter_monsoon_DJF", 1: "winter_monsoon_DJF", 2: "winter_monsoon_DJF",
    3: "pre_monsoon_MAM", 4: "pre_monsoon_MAM", 5: "pre_monsoon_MAM",
    6: "summer_monsoon_JJA", 7: "summer_monsoon_JJA", 8: "summer_monsoon_JJA",
    9: "post_monsoon_SON", 10: "post_monsoon_SON", 11: "post_monsoon_SON",
}


def load_embeddings(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"[embedding_analysis] {path} not found - run run_day3.py first "
            f"(it saves results/embeddings.npz)."
        )
    npz = np.load(path, allow_pickle=True)
    out = {k: npz[k] for k in npz.files}
    print(f"[embedding_analysis] Loaded {out['embeddings'].shape[0]} embedding "
          f"points ({out['embeddings'].shape[1]}-D) from {path}, "
          f"split counts: {dict(zip(*np.unique(out['split'], return_counts=True)))}")
    return out


def attach_region_and_season(emb: dict) -> dict:
    """Adds 'region' (str, from Stage 4's 77E split - reused, never
    redefined) and 'season' (str) arrays. A point outside both defined
    regions (shouldn't happen inside 45-105E but handled defensively)
    is labeled 'other'."""
    emb = dict(emb)
    masks = regional_split(emb["lon"])
    region = np.full(len(emb["lon"]), "other", dtype=object)
    for name, mask in masks.items():
        region[mask] = name
    emb["region"] = region.astype(str)
    emb["season"] = np.array([_SEASON_BY_MONTH[m] for m in emb["month"]])
    return emb


def attach_surface_sst(emb: dict, ds: xr.Dataset) -> dict:
    """Adds 'sst_normalized' - the SAME normalized SST value the CNN
    actually saw at that (time, lat, lon) (ds's 'sst' channel is
    already z-scored per Day 1's normalization.py - NOT raw degC).
    Labeled '_normalized' explicitly so nobody mistakes this for a
    physical temperature."""
    emb = dict(emb)
    sst_arr = ds["sst"].values  # (time, lat, lon), already normalized
    time_grid = pd.DatetimeIndex(ds["time"].values)
    sample_time_idx = time_grid.get_indexer(pd.to_datetime(emb["time"]))
    if (sample_time_idx < 0).any():
        raise ValueError(
            "[embedding_analysis] Some embedding sample timestamps were not "
            "found in ds['time'] - metadata/time misalignment, refusing to "
            "silently drop them."
        )
    emb["sst_normalized"] = sst_arr[sample_time_idx, emb["lat_idx"], emb["lon_idx"]]
    return emb


def subsample(emb: dict, max_points: int, seed: int = 42) -> dict:
    """Documented random subsample (never load-and-plot millions of
    points). Stratified is unnecessary here since PCA/plotting just
    needs a representative cloud, not per-stratum guarantees."""
    n = len(emb["embeddings"])
    if n <= max_points:
        print(f"[embedding_analysis] {n} points <= max_points={max_points}, "
              f"using all of them (no subsampling needed).")
        return emb
    rng = np.random.default_rng(seed)
    sel = rng.choice(n, size=max_points, replace=False)
    print(f"[embedding_analysis] Subsampling {n} -> {max_points} points "
          f"(seed={seed}) for visualization/PCA. This is a random subsample, "
          f"not the full dataset.")
    return {k: v[sel] for k, v in emb.items()}


def run_pca(embeddings: np.ndarray, n_components: int = 2):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(embeddings)
    return coords, pca.explained_variance_ratio_


def run_tsne_optional(embeddings: np.ndarray, max_points_for_tsne: int = 8000,
                       seed: int = 42):
    """Optional, per the brief ('only if practical'). t-SNE is O(n^2)-ish
    and impractical on millions of points, so this ALWAYS subsamples
    further before running it, and returns None (skips) above
    max_points_for_tsne rather than silently running something slow.
    Uses sklearn.manifold.TSNE - already a project dependency
    (scikit-learn), so no new dependency is introduced."""
    from sklearn.manifold import TSNE
    n = len(embeddings)
    if n > max_points_for_tsne:
        rng = np.random.default_rng(seed)
        sel = rng.choice(n, size=max_points_for_tsne, replace=False)
        embeddings = embeddings[sel]
    else:
        sel = np.arange(n)
    print(f"[embedding_analysis] Running t-SNE on {len(embeddings)} points "
          f"(optional, subsampled for practicality).")
    tsne = TSNE(n_components=2, random_state=seed, init="pca",
                perplexity=min(30, max(5, len(embeddings) // 100)))
    coords = tsne.fit_transform(embeddings)
    return coords, sel


def _plot_categorical(coords, labels, title, out_path, label_order=None):
    plt.figure(figsize=(6, 5))
    labels = np.asarray(labels)
    cats = label_order if label_order is not None else sorted(set(labels))
    cmap = plt.get_cmap("tab10")
    for i, cat in enumerate(cats):
        mask = labels == cat
        if not mask.any():
            continue
        plt.scatter(coords[mask, 0], coords[mask, 1], s=4, alpha=0.4,
                    color=cmap(i % 10), label=f"{cat} (n={int(mask.sum())})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title(title)
    plt.legend(markerscale=3, fontsize=8)
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()


def _plot_continuous(coords, values, title, out_path, cbar_label):
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=values, cmap="coolwarm", s=4, alpha=0.5)
    plt.colorbar(sc, label=cbar_label)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title(title)
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()


def make_all_pca_plots(emb: dict, pca_coords: np.ndarray, explained_var: np.ndarray,
                        plots_dir: Path, method_label: str = "PCA"):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    var_str = f"({explained_var[0]*100:.1f}%, {explained_var[1]*100:.1f}% var)" \
        if method_label == "PCA" else ""

    _plot_categorical(
        pca_coords, emb["region"],
        f"CNN embedding {method_label} — colored by region {var_str}",
        plots_dir / f"embedding_{method_label.lower()}_by_region.png",
        label_order=["arabian_sea", "bay_of_bengal", "other"],
    )
    _plot_categorical(
        pca_coords, emb["season"],
        f"CNN embedding {method_label} — colored by season {var_str}",
        plots_dir / f"embedding_{method_label.lower()}_by_season.png",
        label_order=["winter_monsoon_DJF", "pre_monsoon_MAM",
                     "summer_monsoon_JJA", "post_monsoon_SON"],
    )
    if "sst_normalized" in emb:
        _plot_continuous(
            pca_coords, emb["sst_normalized"],
            f"CNN embedding {method_label} — colored by surface SST (normalized) {var_str}",
            plots_dir / f"embedding_{method_label.lower()}_by_sst.png",
            cbar_label="SST (normalized, z-score)",
        )
    print(f"[embedding_analysis] Saved {method_label} plots (region, season"
          f"{', sst' if 'sst_normalized' in emb else ''}) -> {plots_dir}")
