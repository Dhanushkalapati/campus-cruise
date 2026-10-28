"""Experimental ML travel-time estimate (src/ml_travel_time.py).

Synthetic data is labelled SYNTHETIC. Tests on the real cache are skipped
when no cache file exists, so the suite runs on a fresh clone.
"""

import json

import numpy as np
import pytest

from src import ml_travel_time as ml
from src.distances import approximate_time_matrix

REAL = ml.load_osrm_cache()


def synthetic_cache(tmp_path, n=16, seed=0):
    """SYNTHETIC: random points near Cardiff; 'durations' = 1.4 x straight-line time + noise."""
    rng = np.random.default_rng(seed)
    pts = [(51.5 + rng.uniform(-0.3, 0.3), -3.2 + rng.uniform(-0.6, 0.6)) for _ in range(n)]
    pts.append((51.478283, -3.182652))
    base = approximate_time_matrix(pts) * 3600 * 1.4 + rng.uniform(0, 60, (n + 1, n + 1))
    np.fill_diagonal(base, 0)
    (tmp_path / "osrm_synthetic.json").write_text(json.dumps({"points": pts, "durations": base.tolist()}))
    return pts


def test_features_need_only_coordinates_and_exclude_event_origin():
    pts = [(51.6, -3.3), (51.7, -3.4), (51.478283, -3.182652)]
    X, o, d = ml.pair_features(pts)
    assert X.shape == (4, len(ml.FEATURES))           # 2 schools x 2 destinations each
    assert 2 not in o                                  # no pair leaves the event
    assert set(zip(o, d)) == {(0, 1), (0, 2), (1, 0), (1, 2)}
    assert np.all(X[d == 2, 4] == 1) and np.all(X[d != 2, 4] == 0)


def test_group_folds_cover_each_school_once_and_are_reproducible():
    a, b = ml.school_folds(23), ml.school_folds(23)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    flat = np.concatenate(a)
    assert sorted(flat.tolist()) == list(range(23))


def test_grouped_cv_never_trains_on_held_out_schools(tmp_path, monkeypatch):
    synthetic_cache(tmp_path)
    data, _ = ml.load_osrm_cache(tmp_path)
    seen = []
    real_fit = ml.fit_ridge

    def spy(X, y, alpha=ml.ALPHA):
        seen.append(len(y))
        return real_fit(X, y, alpha)

    monkeypatch.setattr(ml, "fit_ridge", spy)
    ml.grouped_cv(data)
    n = data.n_points - 1
    for held in ml.school_folds(n):
        k = len(held)
        expected_train = (n - k) * (n - k - 1) + (n - k)  # pairs among kept schools, plus to the event
        assert expected_train in seen


def test_metrics_are_correct():
    y = np.array([0.5, 1.0, 0.05])        # hours
    p = np.array([0.6, 0.9, 0.05])
    m = ml.metrics(y, p)
    assert m["mae_min"] == pytest.approx((6 + 6 + 0) / 3)
    assert m["rmse_min"] == pytest.approx(np.sqrt((36 + 36) / 3))
    assert m["mape_pct"] == pytest.approx((6 / 30 + 6 / 60) / 2 * 100)  # 3-minute pair excluded


def test_train_and_load_round_trip_with_safe_predictions(tmp_path):
    pts = synthetic_cache(tmp_path, n=20)
    art = ml.train(tmp_path)
    assert art["training_data"]["target"].startswith("OSRM estimated")
    path = tmp_path / "model.json"
    ml.save(art, path)
    model = ml.load_model(path)
    assert model is not None
    H = model.predict_hours(pts)
    assert H.shape == (21, 21) and np.all(np.isfinite(H)) and np.all(H >= 0)
    assert np.all(np.diag(H) == 0)


def test_too_little_data_is_refused(tmp_path):
    synthetic_cache(tmp_path, n=5)
    with pytest.raises(ValueError, match="Too little data"):
        ml.train(tmp_path)


@pytest.mark.parametrize("damage", ["missing", "not json", "nan coef", "wrong version", "bad feature"])
def test_malformed_artefact_loads_as_none(tmp_path, damage):
    synthetic_cache(tmp_path, n=20)
    art = ml.train(tmp_path)
    path = tmp_path / "model.json"
    if damage == "missing":
        assert ml.load_model(path) is None
        return
    if damage == "not json":
        path.write_text("{oops")
    else:
        if damage == "nan coef":
            art["model"]["coef"][0] = float("nan")
        elif damage == "wrong version":
            art["version"] = 99
        else:
            art["features"] = ["actual_duration"]  # a leaking feature name is rejected
        path.write_text(json.dumps(art))
    assert ml.load_model(path) is None


def test_readiness_reports_no_observed_journeys():
    r = ml.data_readiness()
    assert r["observed_journey_records"] == 0 and not r["has_timestamps"]
    assert "No observed journey times" in r["verdict"]


@pytest.mark.skipif(REAL is None, reason="no cached OSRM matrix")
def test_saved_artefact_matches_real_cache_and_is_honest():
    model = ml.load_model()
    assert model is not None, "run python train_travel_time.py"
    art = model.artefact
    assert art["training_data"]["sha256"] == REAL[1]["sha256"]  # trained on the current cache
    ev = art["evaluation"]
    assert ev["beats_fallback"] == (ev["metrics"][art["model"]["name"]]["mae_min"] < ev["metrics"]["B"]["mae_min"])
    assert "Not a real journey-time predictor" in art["intended_use"]


@pytest.mark.skipif(REAL is None, reason="no cached OSRM matrix")
def test_time_model_runs_on_ml_matrix_and_labels_it(real_dataset):
    from src.distances import TimeMatrix
    from src.model import Constraints
    from src.planner import optimise, points_of

    model = ml.load_model()
    H = model.predict_hours(points_of(real_dataset))
    res = optimise(real_dataset, "time", Constraints(), 5, TimeMatrix(H, ml.SOURCE_ML, []))
    sol = res.solution
    assert sol.status == "FEASIBLE" and sol.has_plan and not sol.violations   # validator passed
    assert sol.matrix_source == ml.SOURCE_ML and sol.matrix_source.startswith("ML ESTIMATE")
    assert any("experimental ML estimate" in n for n in sol.notes)
    assert res.baseline.matrix_source == ml.SOURCE_ML


def test_app_uses_ml_source_only_when_chosen():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=240).run()
    at.sidebar.radio[0].set_value("time").run()
    assert at.selectbox(key="time_source_choice").value == "osrm"            # OSRM stays the default
    at.sidebar.slider[0].set_value(5).run()
    at.selectbox(key="time_source_choice").set_value("ml").run()
    at.button(key="run_optimiser").click().run()
    sol = at.session_state["current"]["result"].solution
    expected = ml.SOURCE_ML if (ml.load_model() and ml.load_model().usable) else "APPROXIMATE"
    assert sol.matrix_source.startswith(expected[:12])
    at.selectbox(key="time_source_choice").set_value("estimate").run()      # source change = stale
    assert any("road-time source" in w.value for w in at.warning)
