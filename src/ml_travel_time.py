"""Experimental ML travel-time estimate: a learned approximation of OSRM road times.

What this is, and what it is not
--------------------------------
The repository contains no observed journey times (no trip logs, timestamps
or traffic records). The only travel-time data is the cached OSRM matrix in
``data/cache/``: road-network *estimates* from the public OSRM server between
the 60 pickup points and the stadium. A model trained on it therefore learns
to **approximate OSRM estimates from coordinates**. It is not, and is never
labelled as, a predictor of real-world journey times or traffic.

Why it can still be useful: when OSRM is unreachable and nothing is cached
(for example, new pickup points), the app falls back to a fixed-speed formula
(straight-line miles x 1.3 / 35 mph). A model fitted to real road estimates
in the same region may approximate road times better than that formula. The
evaluation below tests exactly that, on schools held out of training.

Method (numpy only, no new dependency, no pickled code)
-------------------------------------------------------
* Samples: ordered pairs (origin school -> other school or stadium) from the
  cached OSRM table. Pairs leaving the stadium are excluded: no bus does that.
* Target: OSRM duration in hours.
* Features, all computable from coordinates before any trip: straight-line
  miles, its square root, each endpoint's straight-line miles to the
  stadium, and whether the destination is the stadium. No feature uses the
  target or any OSRM output (no leakage).
* Candidates: (B) the existing fixed-speed fallback, no fitting;
  (M1) calibrated speed: ridge regression on straight-line miles only;
  (M2) ridge regression on all features.
* Evaluation: 5-fold cross-validation grouped by school. In each fold the
  held-out schools appear in no training pair, and every test pair involves
  a held-out school, so the score measures new pickup points.
* Metrics: MAE and RMSE in minutes; MAPE only on pairs over 5 minutes.

The trained artefact is plain JSON (coefficients, scaling, metrics,
provenance). ``train_travel_time.py`` retrains it; the app only loads it.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .distances import AVERAGE_SPEED_MPH, CACHE_DIR, ROAD_FACTOR, straight_line_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "travel_time_ml.json"
ARTEFACT_VERSION = 1
SOURCE_ML = "ML ESTIMATE (experimental): ridge regression approximating cached OSRM road times"
FEATURES = ["miles", "sqrt_miles", "origin_miles_to_event", "dest_miles_to_event", "dest_is_event"]
CANDIDATES = {"B": "Fixed-speed fallback (existing formula, not fitted)",
              "M1": "Calibrated speed: ridge on straight-line miles",
              "M2": "Ridge on all features"}
FOLDS = 5
SEED = 42
ALPHA = 1.0
MAPE_MIN_MINUTES = 5.0
MIN_PAIRS = 200      # below this, no evaluation is reported
MIN_SCHOOLS = 10     # grouped CV needs enough distinct schools


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


@dataclass
class PairData:
    X: np.ndarray        # features, one row per pair
    y: np.ndarray        # OSRM hours
    origin: np.ndarray   # origin point index
    dest: np.ndarray     # destination point index (event = n - 1)
    n_points: int


def pair_features(points: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Features for every ordered pair i -> j (i != j, i not the event). Last point is the event."""
    miles = straight_line_matrix(points)
    n = len(points)
    event = n - 1
    rows, origins, dests = [], [], []
    for i in range(n - 1):
        for j in range(n):
            if i == j:
                continue
            d = miles[i, j]
            rows.append([d, math.sqrt(d), miles[i, event], miles[j, event], 1.0 if j == event else 0.0])
            origins.append(i)
            dests.append(j)
    return np.array(rows, float), np.array(origins), np.array(dests)


def load_osrm_cache(cache_dir: Path = CACHE_DIR) -> tuple[PairData, dict] | None:
    """Training pairs from the cached OSRM table, or None if no valid cache exists."""
    files = sorted(glob.glob(str(Path(cache_dir) / "osrm_*.json")))
    for f in files:
        try:
            body = json.loads(Path(f).read_text())
            points = [tuple(map(float, p)) for p in body["points"]]
            dur = np.array(body["durations"], float)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if dur.shape != (len(points), len(points)) or not np.all(np.isfinite(dur)) or np.any(dur < 0):
            continue
        X, o, d = pair_features(points)
        y = dur[o, d] / 3600.0
        provenance = {
            "file": Path(f).name,
            "sha256": hashlib.sha256(Path(f).read_bytes()).hexdigest()[:16],
            "modified": datetime.fromtimestamp(Path(f).stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            "source": "OSRM public demo server (router.project-osrm.org), table service, driving profile",
            "points": len(points),
            "pairs": int(len(y)),
            "target": "OSRM estimated driving duration (not an observed journey time)",
        }
        return PairData(X, y, o, d, len(points)), provenance
    return None


def data_readiness(project_root: Path = PROJECT_ROOT, cache_dir: Path = CACHE_DIR) -> dict:
    """What travel-time data exists, and whether a real-journey-time model is justified."""
    schools_csv = project_root / "attendants_dataset.csv"
    columns = []
    if schools_csv.exists():
        columns = [c.strip() for c in schools_csv.read_text(encoding="utf-8").splitlines()[0].split(",")]
    cache = load_osrm_cache(cache_dir)
    routes = len(glob.glob(str(Path(cache_dir) / "routes" / "route_*.json")))
    observed = []  # no file in the project holds observed trips; listed for transparency
    return {
        "school_file_columns": columns,
        "observed_journey_records": len(observed),
        "has_timestamps": False,
        "has_traffic_or_weather": False,
        "osrm_matrix_pairs": cache[1]["pairs"] if cache else 0,
        "osrm_route_paths": routes,
        "verdict": (
            "No observed journey times exist, so no model of real journey time can be trained or "
            "evaluated. The cached OSRM estimates support only a model that approximates OSRM."
        ),
        "osrm_surrogate_possible": bool(cache and cache[1]["pairs"] >= MIN_PAIRS
                                        and cache[0].n_points - 1 >= MIN_SCHOOLS),
    }


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


def fallback_hours(X: np.ndarray) -> np.ndarray:
    """The app's existing offline estimate: miles x ROAD_FACTOR / AVERAGE_SPEED_MPH."""
    return X[:, 0] * ROAD_FACTOR / AVERAGE_SPEED_MPH


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float = ALPHA) -> dict:
    """Standardised ridge regression with an unpenalised intercept."""
    mean, std = X.mean(axis=0), X.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    Z = (X - mean) / std
    A = Z.T @ Z + alpha * np.eye(Z.shape[1])
    coef = np.linalg.solve(A, Z.T @ (y - y.mean()))
    return {"mean": mean.tolist(), "std": std.tolist(), "coef": coef.tolist(), "intercept": float(y.mean())}


def predict_ridge(params: dict, X: np.ndarray) -> np.ndarray:
    Z = (X - np.array(params["mean"])) / np.array(params["std"])
    return Z @ np.array(params["coef"]) + params["intercept"]


def _columns(name: str) -> list[int]:
    return [0] if name == "M1" else list(range(len(FEATURES)))


def metrics(y_true_h: np.ndarray, y_pred_h: np.ndarray) -> dict:
    """MAE, RMSE (minutes); MAPE (%) on pairs over MAPE_MIN_MINUTES only."""
    err = (y_pred_h - y_true_h) * 60.0
    mask = y_true_h * 60.0 > MAPE_MIN_MINUTES
    mape = float(np.mean(np.abs(err[mask]) / (y_true_h[mask] * 60.0)) * 100) if mask.any() else None
    return {"mae_min": float(np.mean(np.abs(err))), "rmse_min": float(math.sqrt(np.mean(err ** 2))),
            "mape_pct": mape, "n": int(len(err))}


def school_folds(n_schools: int, folds: int = FOLDS, seed: int = SEED) -> list[np.ndarray]:
    order = np.random.default_rng(seed).permutation(n_schools)
    return [np.sort(order[k::folds]) for k in range(folds)]


def grouped_cv(data: PairData, folds: int = FOLDS, seed: int = SEED, alpha: float = ALPHA) -> dict:
    """Out-of-fold predictions per candidate; every school is held out exactly once."""
    n_schools = data.n_points - 1
    preds = {k: np.full(len(data.y), np.nan) for k in CANDIDATES}
    for held in school_folds(n_schools, folds, seed):
        held_set = set(held.tolist())
        test = np.array([o in held_set or d in held_set for o, d in zip(data.origin, data.dest)])
        train = np.array([o not in held_set and d not in held_set for o, d in zip(data.origin, data.dest)])
        # A test pair can appear in two folds (both ends held out in different folds);
        # it is scored in the first fold that holds out either end.
        first = test & np.isnan(preds["B"])
        preds["B"][first] = fallback_hours(data.X[first])
        for name in ("M1", "M2"):
            cols = _columns(name)
            params = fit_ridge(data.X[train][:, cols], data.y[train], alpha)
            preds[name][first] = predict_ridge(params, data.X[first][:, cols])
    return {name: metrics(data.y, p) for name, p in preds.items()}


# --------------------------------------------------------------------------- #
# Training and the artefact
# --------------------------------------------------------------------------- #


def train(cache_dir: Path = CACHE_DIR, alpha: float = ALPHA) -> dict:
    """Evaluate candidates, fit the selected model on all pairs, return the artefact."""
    loaded = load_osrm_cache(cache_dir)
    if loaded is None:
        raise ValueError("No valid cached OSRM matrix: nothing to train on.")
    data, provenance = loaded
    if len(data.y) < MIN_PAIRS or data.n_points - 1 < MIN_SCHOOLS:
        raise ValueError(f"Too little data: {len(data.y)} pairs, {data.n_points - 1} schools.")
    cv = grouped_cv(data, alpha=alpha)
    best = min(("M1", "M2"), key=lambda k: cv[k]["mae_min"])
    beats = cv[best]["mae_min"] < cv["B"]["mae_min"] and cv[best]["rmse_min"] < cv["B"]["rmse_min"]
    cols = _columns(best)
    params = fit_ridge(data.X[:, cols], data.y, alpha)
    return {
        "version": ARTEFACT_VERSION,
        "created": f"{datetime.now():%Y-%m-%d %H:%M}",
        "intended_use": "Experimental approximation of OSRM road-time estimates for pickup points "
                        "in this region when OSRM data is unavailable. Not a real journey-time "
                        "predictor: no observed journeys exist.",
        "training_data": provenance,
        "features": [FEATURES[c] for c in cols],
        "model": {"type": "ridge regression (standardised features)", "alpha": alpha, "name": best,
                  "description": CANDIDATES[best], **params},
        "evaluation": {
            "method": f"{FOLDS}-fold cross-validation grouped by school (seed {SEED}); held-out "
                      "schools appear in no training pair",
            "metrics": cv,
            "candidates": CANDIDATES,
            "beats_fallback": bool(beats),
        },
        "limitations": [
            "Target is OSRM's estimate, not observed travel: no traffic, time of day, weather or dwell time.",
            "Trained on 60 pickup points in South Wales and one destination; outside this region it is untested.",
            "Grouped CV estimates error for new points in the same region only.",
        ],
    }


def save(artefact: dict, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artefact, indent=2), encoding="utf-8")


@dataclass
class TravelTimeModel:
    artefact: dict

    @property
    def usable(self) -> bool:
        """Only a model that beat the fallback on held-out schools may be used."""
        return bool(self.artefact["evaluation"]["beats_fallback"])

    def predict_hours(self, points: list[tuple[float, float]]) -> np.ndarray:
        """Full matrix (schools then event). Diagonal 0; values finite and >= 0."""
        n = len(points)
        X, o, d = pair_features(points)
        cols = [FEATURES.index(f) for f in self.artefact["features"]]
        y = predict_ridge(self.artefact["model"], X[:, cols])
        if not np.all(np.isfinite(y)):
            raise ValueError("Model produced non-finite travel times.")
        out = np.zeros((n, n))
        out[o, d] = np.maximum(y, 0.0)
        # Rows from the event are never used by a route; fill them symmetrically for safety.
        out[n - 1, :] = out[:, n - 1]
        return out


def load_model(path: Path = MODEL_PATH) -> TravelTimeModel | None:
    """The saved model, or None if missing or malformed (the app then falls back)."""
    try:
        art = json.loads(Path(path).read_text(encoding="utf-8"))
        m = art["model"]
        k = len(art["features"])
        if art.get("version") != ARTEFACT_VERSION or not set(art["features"]) <= set(FEATURES):
            return None
        if not (len(m["coef"]) == len(m["mean"]) == len(m["std"]) == k):
            return None
        values = m["coef"] + m["mean"] + m["std"] + [m["intercept"]]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            return None
        art["evaluation"]["beats_fallback"]  # noqa: B018 - must exist
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return TravelTimeModel(art)
