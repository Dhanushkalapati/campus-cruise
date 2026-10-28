"""Result freshness: is the plan on screen built from the inputs now in the app?

Pure functions, no Streamlit, so the rules are unit-tested directly.

A run is identified by a ``RunKey``: everything that changes which plans are
valid or what they cost (cost model, attendance edits, sharing rules, and for
the time model whether online road times were allowed). The solver time limit
is a *setting*: a different limit does not make a plan wrong, it only means a
new run could search for longer.
"""

from __future__ import annotations

from dataclasses import astuple, dataclass

from .model import Constraints

# Freshness states of the displayed result.
NONE = "none"                  # no result yet
CURRENT = "current"            # built from the current inputs and settings
SETTINGS_CHANGED = "settings"  # current inputs, but a different time limit
STALE = "stale"                # inputs changed since the run: out of date

# Automatic re-optimisation when inputs change. The fixed model is exact and
# usually proves its optimum in seconds; routing models use their full time
# limit (300 s by default), so they wait for an explicit run.
AUTO_RUN_DEFAULT = {"fixed": True, "distance": False, "time": False}

CACHE_SIZE = 20


@dataclass(frozen=True)
class RunKey:
    model: str
    overrides: tuple[tuple[str, int], ...]
    must_share: tuple[tuple[str, str], ...]
    cannot_share: tuple[tuple[str, str], ...]
    solo: tuple[str, ...]
    online_routing: bool | None  # time model only; None for the other models
    time_source: str | None = None  # time model only: "osrm", "ml" or "estimate"


def run_key(model: str, overrides: dict, constraints: Constraints, allow_online: bool = True,
            time_source: str = "osrm") -> RunKey:
    """Order-independent fingerprint of the optimisation inputs."""
    def pairs(ps):
        return tuple(sorted(tuple(sorted(p)) for p in ps))

    return RunKey(
        model=model,
        overrides=tuple(sorted((str(k), int(v)) for k, v in overrides.items())),
        must_share=pairs(constraints.must_share),
        cannot_share=pairs(constraints.cannot_share),
        solo=tuple(sorted(constraints.solo)),
        online_routing=bool(allow_online) if model == "time" else None,
        time_source=time_source if model == "time" else None,
    )


def changed_inputs(old: RunKey, new: RunKey) -> list[str]:
    """Names of the inputs that differ, in the words the dashboard uses."""
    out = []
    if old.model != new.model:
        out.append("cost model")
    if old.overrides != new.overrides:
        out.append("attendance")
    if (old.must_share, old.cannot_share, old.solo) != (new.must_share, new.cannot_share, new.solo):
        out.append("constraints")
    if old.model == new.model and (old.online_routing, old.time_source) != (new.online_routing, new.time_source):
        out.append("road-time source")
    return out


@dataclass(frozen=True)
class Freshness:
    state: str
    changes: tuple[str, ...] = ()
    run_limit: float | None = None   # time limit the shown result used
    want_limit: float | None = None  # time limit now set in the sidebar

    @property
    def needs_run(self) -> bool:
        """True when the shown result does not reflect the current inputs."""
        return self.state in (NONE, STALE)


def same_inputs(a: RunKey, b: RunKey) -> bool:
    """Field-by-field comparison. Streamlit re-imports this module after a code edit, and a
    dataclass never equals an instance of the re-imported class, even with equal fields."""
    return astuple(a) == astuple(b)


def freshness(entry: dict | None, key: RunKey, time_limit: float) -> Freshness:
    """Compare a stored run (``entry["key"]``, ``entry["settings"]``) with the current inputs."""
    if entry is None:
        return Freshness(NONE)
    if not same_inputs(entry["key"], key):
        return Freshness(STALE, tuple(changed_inputs(entry["key"], key)))
    used = float(entry["settings"]["time_limit"])
    if used != float(time_limit):
        return Freshness(SETTINGS_CHANGED, (), used, float(time_limit))
    return Freshness(CURRENT, (), used, float(time_limit))


def should_auto_run(fresh: Freshness, auto_enabled: bool, key: RunKey, last_attempt: RunKey | None,
                    has_input_errors: bool, any_run_done: bool) -> bool:
    """Run automatically only when inputs changed, once per set of inputs.

    - never before the organiser's first deliberate run (page load stays fast);
    - never on a settings-only change (a longer limit is a deliberate choice);
    - never twice for the same inputs, even if that attempt failed, so a
      timeout or error cannot cause a loop of expensive runs;
    - never while the attendance editor holds rejected values.
    """
    return (auto_enabled and any_run_done and not has_input_errors
            and fresh.state == STALE and last_attempt != key)


def cache_put(cache: dict, key: RunKey, entry: dict, limit: int = CACHE_SIZE) -> None:
    """Insert, newest last; drop the oldest entries beyond ``limit``."""
    cache.pop(key, None)
    cache[key] = entry
    while len(cache) > limit:
        cache.pop(next(iter(cache)))
