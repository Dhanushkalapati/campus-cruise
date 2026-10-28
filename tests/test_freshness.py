"""Unit tests for result-freshness rules (src/freshness.py). No solver, no Streamlit."""

from src.freshness import (
    CURRENT,
    NONE,
    SETTINGS_CHANGED,
    STALE,
    cache_put,
    changed_inputs,
    freshness,
    run_key,
    should_auto_run,
)
from src.model import Constraints


def entry(key, limit=20.0):
    return {"key": key, "settings": {"time_limit": limit}}


def test_key_ignores_order_of_edits_and_pairs():
    a = run_key("fixed", {"NP11": 15, "CF15": 3}, Constraints([("CF15", "CF38")], [], ["NP4", "NP11"]))
    b = run_key("fixed", {"CF15": 3, "NP11": 15}, Constraints([("CF38", "CF15")], [], ["NP11", "NP4"]))
    assert a == b and hash(a) == hash(b)


def test_key_changes_with_every_input():
    base = run_key("distance", {}, Constraints())
    assert run_key("fixed", {}, Constraints()) != base
    assert run_key("distance", {"NP11": 15}, Constraints()) != base
    assert run_key("distance", {}, Constraints(must_share=[("A", "B")])) != base
    assert run_key("distance", {}, Constraints(cannot_share=[("A", "B")])) != base
    assert run_key("distance", {}, Constraints(solo=["A"])) != base


def test_online_routing_flag_matters_only_for_time_model():
    assert run_key("distance", {}, Constraints(), True) == run_key("distance", {}, Constraints(), False)
    on, off = run_key("time", {}, Constraints(), True), run_key("time", {}, Constraints(), False)
    assert on != off and changed_inputs(on, off) == ["road-time source"]


def test_freshness_states():
    key = run_key("fixed", {}, Constraints())
    assert freshness(None, key, 20).state == NONE
    assert freshness(entry(key), key, 20).state == CURRENT
    f = freshness(entry(key, 20), key, 30)
    assert f.state == SETTINGS_CHANGED and (f.run_limit, f.want_limit) == (20, 30) and not f.needs_run
    new = run_key("distance", {"NP11": 15}, Constraints(solo=["NP4"]))
    f = freshness(entry(key), new, 20)
    assert f.state == STALE and f.needs_run
    assert f.changes == ("cost model", "attendance", "constraints")


def test_auto_run_rules():
    old = run_key("fixed", {}, Constraints())
    new = run_key("fixed", {"NP11": 15}, Constraints())
    stale = freshness(entry(old), new, 20)
    assert should_auto_run(stale, True, new, old, False, True)
    assert not should_auto_run(stale, False, new, old, False, True)   # auto switched off
    assert not should_auto_run(stale, True, new, new, False, True)    # already tried these inputs
    assert not should_auto_run(stale, True, new, old, True, True)     # editor has errors
    assert not should_auto_run(stale, True, new, old, False, False)   # no manual run yet
    settings_only = freshness(entry(new, 20), new, 60)
    assert not should_auto_run(settings_only, True, new, old, False, True)
    assert not should_auto_run(freshness(entry(new), new, 20), True, new, old, False, True)


def test_cache_evicts_oldest_and_refreshes_on_put():
    cache = {}
    keys = [run_key("fixed", {"NP11": n}, Constraints()) for n in range(5)]
    for k in keys:
        cache_put(cache, k, {"n": k}, limit=3)
    assert list(cache) == keys[2:]
    cache_put(cache, keys[2], {"n": "again"}, limit=3)  # re-inserted: now newest
    cache_put(cache, keys[0], {}, limit=3)
    assert list(cache) == [keys[4], keys[2], keys[0]]


def test_freshness_survives_a_module_reload():
    # Regression: after a code edit Streamlit re-imports src.freshness; the stored key is then an
    # instance of the old RunKey class and never compared equal, so the plan showed as stale
    # with no listed change.
    from dataclasses import dataclass, fields

    import src.freshness as fr

    Reloaded = dataclass(frozen=True)(type("RunKey", (), {"__annotations__": {
        f.name: f.type for f in fields(fr.RunKey)}}))
    key = fr.run_key("fixed", {"NP11": 15}, fr.Constraints())
    old = Reloaded(*fr.astuple(key))
    assert old != key  # what plain dataclass equality does
    entry = {"key": old, "settings": {"time_limit": 20}}
    assert fr.freshness(entry, key, 20).state == fr.CURRENT
    other = fr.run_key("fixed", {"NP11": 16}, fr.Constraints())
    assert fr.freshness(entry, other, 20).state == fr.STALE
