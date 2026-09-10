"""
test_predict.py
================
Smoke test for the deploy folder — no API server needed.
Loads both checkpoints and runs one MLP prediction + one CNN prediction
on a plausible synthetic Arabian-Sea-like surface reading, and prints
the resulting 15-depth temperature profile.

Run:
    python test_predict.py
"""

import json

from predictor import OceanXRayPredictor

if __name__ == "__main__":
    print("=" * 70)
    print("OceanXRay deploy folder — smoke test")
    print("=" * 70)

    predictor = OceanXRayPredictor()
    print(f"\nnormalization_stats.json found: {predictor.stats_available}")
    print(f"CNN channels (order matters):  {predictor.cnn_channels}")
    print(f"MLP channels (order matters):  {predictor.mlp_channels}")
    print(f"Output depths (m):              {predictor.depths}")

    # Plausible single-point surface reading (Arabian Sea, pre-monsoon-ish)
    point_values = {
        "sst": 29.2,        # degC
        "ssh": 0.10,        # m
        "u_current": 0.15,  # m/s
        "v_current": -0.05, # m/s
        "v_wind": 4.0,      # m/s
    }

    print("\n--- MLP (point-based baseline) ---")
    print("Input:", point_values)
    mlp_out = predictor.predict_mlp(point_values)
    print(json.dumps(mlp_out, indent=2))

    # Same reading repeated over a 3x3 patch (stand-in for real spatial data)
    patch_values = {c: [[v, v, v], [v, v, v], [v, v, v]] for c, v in point_values.items()}

    print("\n--- CNN (main/locked model, 3x3 patch) ---")
    cnn_out = predictor.predict_cnn(patch_values)
    print(json.dumps(cnn_out, indent=2))

    print("\n" + "=" * 70)
    if not predictor.stats_available:
        print("NOTE: normalization_stats.json was NOT found, so the numbers above")
        print("are placeholder/identity-normalized — NOT real degC. Copy the real")
        print("file from Drive (see README.md) and re-run this test before demo.")
    else:
        print("Both models loaded, ran inference, and returned 15-depth profiles OK.")
    print("=" * 70)
