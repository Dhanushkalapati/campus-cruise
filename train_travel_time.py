"""Retrain the experimental ML travel-time estimate (see src/ml_travel_time.py).

    python train_travel_time.py

Reads the cached OSRM matrix in data/cache/, runs grouped cross-validation,
fits the selected model on all pairs and writes models/travel_time_ml.json.
No network access. The app never retrains; it only loads this file.
"""

from __future__ import annotations

import sys

from src.ml_travel_time import MODEL_PATH, data_readiness, save, train


def main() -> int:
    ready = data_readiness()
    print("Data readiness:", ready["verdict"])
    print(f"  observed journey records: {ready['observed_journey_records']}; "
          f"OSRM matrix pairs: {ready['osrm_matrix_pairs']}")
    if not ready["osrm_surrogate_possible"]:
        print("Not enough cached OSRM data to train. Run the time model once with OSRM enabled.")
        return 2
    art = train()
    ev = art["evaluation"]
    print(f"\nEvaluation: {ev['method']}")
    for name, m in ev["metrics"].items():
        mape = "n/a" if m["mape_pct"] is None else f"{m['mape_pct']:.1f}%"
        print(f"  {name} {ev['candidates'][name]:52s} MAE {m['mae_min']:.2f} min  "
              f"RMSE {m['rmse_min']:.2f} min  MAPE {mape}  (n={m['n']})")
    print(f"Selected: {art['model']['name']}; beats the fixed-speed fallback on held-out schools: "
          f"{ev['beats_fallback']}")
    save(art)
    print(f"Wrote {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
