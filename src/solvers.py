"""Optimisation models.

fixed     Exact mixed-size bin packing with OR-Tools CP-SAT. Location is not
          part of the cost; a second, lexicographic stage prefers grouping
          nearby schools without changing the optimal cost.
distance  Capacitated open routing with mixed bus sizes (OR-Tools Routing).
time      Same routing model on a travel-time matrix.

Route accounting (all models, from the challenge data README):
    * A bus starts at its first pickup. Nothing is charged before it.
    * It may stop at further schools, then ends at the event.
    * No return trip is charged. Dwell time at schools is not modelled.
    * Route cost = rate(bus size) x sum of legs (miles or hours).

The routing models use a virtual start node with zero-cost arcs to every
school. This is not a real depot: it encodes "the bus starts at its first
pickup", so the first real charged leg is first school -> next stop.
"""

from __future__ import annotations

import itertools
import math
import time

import numpy as np
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from ortools.sat.python import cp_model

from .distances import straight_line_matrix
from .model import (
    FEASIBLE,
    INFEASIBLE,
    INVALID,
    NO_SOLUTION,
    OPTIMAL,
    BusRoute,
    Constraints,
    Solution,
)
from .validation import route_metric, validate_solution

COST_SCALE = 100  # CP-SAT works in integers: costs in hundredths
ROUTING_SCALE = 1000  # routing arc costs in thousandths of a cost unit
SEED = 42


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def event_points(schools: pd.DataFrame, event_lat: float, event_lon: float) -> list[tuple[float, float]]:
    """(lat, lon) of every school in dataset order, then the event."""
    pts = list(zip(schools["latitude"].astype(float), schools["longitude"].astype(float)))
    return pts + [(float(event_lat), float(event_lon))]


class _UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def must_share_groups(ids: list[str], constraints: Constraints) -> list[list[str]]:
    """Schools linked by must-share pairs (transitively) form one group."""
    uf = _UnionFind(ids)
    for a, b in constraints.must_share:
        if a in uf.parent and b in uf.parent:
            uf.union(a, b)
    groups: dict[str, list[str]] = {}
    for sid in ids:
        groups.setdefault(uf.find(sid), []).append(sid)
    return list(groups.values())


def check_constraints(schools: pd.DataFrame, rates: dict[int, float], constraints: Constraints) -> list[str]:
    """Explain why the constraints cannot be met. Empty list means no conflict found.

    Because the number of buses is not limited, these checks are complete:
    if they pass, putting each must-share group on its own bus is feasible.
    """
    problems: list[str] = []
    attendees = dict(zip(schools["school_id"], schools["attendees"]))
    largest = max(rates)
    named = [s for pair in constraints.must_share + constraints.cannot_share for s in pair] + list(constraints.solo)
    for sid in named:
        if sid not in attendees:
            problems.append(f"Constraint refers to unknown school {sid}.")
        elif attendees[sid] == 0:
            problems.append(f"{sid} has zero attendees and needs no bus; remove its constraint.")
    for a, b in constraints.must_share + constraints.cannot_share:
        if a == b:
            problems.append(f"Constraint pairs {a} with itself.")
    if problems:
        return problems

    active = [s for s, a in attendees.items() if a > 0]
    groups = must_share_groups(active, constraints)
    group_of = {s: i for i, g in enumerate(groups) for s in g}
    for g in groups:
        load = sum(attendees[s] for s in g)
        if len(g) > 1 and load > largest:
            problems.append(
                f"Must-share group {', '.join(g)} has {load} attendees but the largest bus seats {largest}."
            )
    for a, b in constraints.cannot_share:
        if group_of[a] == group_of[b]:
            problems.append(f"{a} and {b} are required to share (directly or via other schools) and also forbidden to share.")
    for s in constraints.solo:
        g = groups[group_of[s]]
        if len(g) > 1:
            problems.append(f"{s} must travel alone but must also share with {', '.join(x for x in g if x != s)}.")
    return problems


def seat_cover_lower_bound(total: int, rates: dict[int, float]) -> float:
    """Cheapest multiset of buses whose seats add up to at least ``total``.

    Every valid plan rents buses that seat everyone, so this is a valid lower
    bound on the fixed-model cost (it ignores the no-split rule). Exact DP.
    """
    if total <= 0:
        return 0.0
    top = total + max(rates)
    best = [0.0] + [math.inf] * top
    for seats in range(1, top + 1):
        best[seats] = min(best[max(0, seats - s)] + c for s, c in rates.items())
    return min(best[total:])


def repair_plan(
    previous: list[list[str]],
    schools: pd.DataFrame,
    rates: dict[int, float],
    constraints: Constraints,
    matrix: np.ndarray,
) -> tuple[list[list[int]], str]:
    """Turn a previous plan into a valid starting plan for the current inputs.

    Returns bus groups as dataset-row indices in pickup order, plus a note.
    Repairs, in order: drop schools that no longer need a bus; take must-share
    group members, and solo schools that share, off their old buses (each
    must-share group and solo school gets its own bus); split must-not-share
    pairs; drop the last pickups of buses that are now over capacity; give
    every school left over (new or removed) its own bus. Assumes
    ``check_constraints`` has passed.
    """
    attendees = dict(zip(schools["school_id"], schools["attendees"]))
    row = {sid: k for k, sid in enumerate(schools["school_id"])}
    active = {s for s, a in attendees.items() if a > 0}
    largest = max(rates)
    groups = must_share_groups(sorted(active), constraints)
    linked = {s for g in groups if len(g) > 1 for s in g}
    solo = set(constraints.solo)
    forbidden = {frozenset(p) for p in constraints.cannot_share}

    kept_routes: list[list[str]] = []
    placed: set[str] = set()
    for stops in previous:
        route = [s for s in stops if s in active and s not in placed and s not in linked]
        if len(route) > 1:
            route = [s for s in route if s not in solo]
        clean: list[str] = []
        for s in route:
            if all(frozenset((s, k)) not in forbidden for k in clean):
                clean.append(s)
        while sum(attendees[s] for s in clean) > largest:
            clean.pop()
        if clean:
            kept_routes.append(clean)
            placed.update(clean)

    previously = {s for stops in previous for s in stops}
    routes = [[row[s] for s in r] for r in kept_routes]
    for g in groups:
        if len(g) > 1:
            routes.append(order_stops([row[s] for s in g], matrix))
    leftovers = sorted(active - placed - linked)
    routes += [[row[s]] for s in leftovers]

    kept = len(placed)
    moved = sorted((active & previously) - placed)
    new = sorted(active - previously)
    note = f"Warm start from the previous plan: {kept} of {len(active)} schools kept their previous bus"
    if moved:
        note += f"; re-placed: {', '.join(moved)}"
    if new:
        note += f"; new to the plan: {', '.join(new)}"
    return routes, note + "."


def order_stops(stops: list[int], matrix: np.ndarray) -> list[int]:
    """Pickup order with the shortest route (exact up to 8 stops, else greedy)."""
    if len(stops) <= 1:
        return list(stops)
    if len(stops) <= 8:
        return list(min(itertools.permutations(stops), key=lambda p: route_metric(list(p), matrix)))
    event = matrix.shape[0] - 1
    remaining = list(stops)
    current = max(remaining, key=lambda s: matrix[s, event])  # start farthest from the event
    order = [current]
    remaining.remove(current)
    while remaining:
        current = min(remaining, key=lambda s: matrix[current, s])
        order.append(current)
        remaining.remove(current)
    return order


def _finish(solution: Solution, schools, rates, constraints, matrix) -> Solution:
    """Number buses, run independent validation, mark INVALID on failure."""
    solution.routes.sort(key=lambda r: (-r.bus_size, r.stops[0] if r.stops else ""))
    for k, route in enumerate(solution.routes, start=1):
        route.bus_id = f"Bus {k:02d}"
    solution.total_cost = float(sum(r.cost for r in solution.routes))
    check_matrix = None if solution.model == "fixed" else matrix
    solution.violations = validate_solution(solution, schools, rates, constraints, check_matrix)
    if solution.violations:
        solution.status = INVALID
    return solution


def _excluded(schools: pd.DataFrame) -> list[str]:
    return schools.loc[schools["attendees"] == 0, "school_id"].tolist()


# --------------------------------------------------------------------------- #
# Baseline: every school on its own cheapest suitable bus
# --------------------------------------------------------------------------- #


def solve_baseline(
    schools: pd.DataFrame, rates: dict[int, float], model: str, matrix: np.ndarray, metric_unit: str = "miles"
) -> Solution:
    """Each school rents its own cheapest bus that fits, driving straight to the event.

    ``matrix`` gives the route metric (miles for fixed/distance, hours for time).
    For the fixed model the metric is display-only.
    """
    sol = Solution(model=model, status=OPTIMAL, method="Baseline: one bus per school",
                   metric_unit=metric_unit, excluded_schools=_excluded(schools))
    event = matrix.shape[0] - 1
    for i, row in enumerate(schools.itertuples()):
        if row.attendees == 0:
            continue
        fits = [s for s in rates if s >= row.attendees]
        if not fits:
            sol.status = INFEASIBLE
            sol.notes.append(f"{row.school_id} ({row.attendees}) fits on no available bus.")
            continue
        metric = float(matrix[i, event])
        unit_cost = (lambda s: rates[s]) if model == "fixed" else (lambda s: rates[s] * metric)
        size = min(fits, key=lambda s: (unit_cost(s), s))
        sol.routes.append(BusRoute("", size, [row.school_id], int(row.attendees), metric, unit_cost(size)))
    if sol.status == INFEASIBLE:
        sol.routes = []
        return sol
    return _finish(sol, schools, rates, None, matrix)


# --------------------------------------------------------------------------- #
# Model 1: fixed cost per bus (exact CP-SAT)
# --------------------------------------------------------------------------- #


def solve_fixed(
    schools: pd.DataFrame,
    rates: dict[int, float],
    event_lat: float,
    event_lon: float,
    constraints: Constraints | None = None,
    time_limit: float = 30.0,
    prefer_nearby: bool = True,
    warm_start: list[list[str]] | None = None,
) -> Solution:
    """Minimise total fixed rental cost. Exact; status says if optimality is proven.

    ``warm_start``: stop lists of a previous plan, repaired and passed to
    CP-SAT as a solution hint. Hints guide the search; they never change the
    optimum or the status rules.

    Formulation (symmetry-free "leader" model): schools are sorted by size,
    largest first. Bus slot j may only exist if school j rides it and is its
    lowest-index school (its leader), so x[i, j] is defined only for j <= i.
    Every partition of schools into buses has exactly one such labelling,
    which removes the permutation symmetry between identical buses.

        min   sum_j sum_t cost_t * z[j, t]
        s.t.  sum_{j<=i} x[i, j] = 1                       every school on one bus
              sum_t z[j, t] = x[j, j]                      slot j used iff led by j
              x[i, j] <= x[j, j]
              sum_{i>=j} a_i x[i, j] <= sum_t cap_t z[j, t]  capacity
    """
    constraints = constraints or Constraints()
    start = time.perf_counter()
    points = event_points(schools, event_lat, event_lon)
    miles = straight_line_matrix(points)
    sol = Solution(model="fixed", status=NO_SOLUTION, time_limit_s=time_limit,
                   method="CP-SAT exact bin packing (symmetry-free leader model)",
                   excluded_schools=_excluded(schools))

    conflicts = check_constraints(schools, rates, constraints)
    too_big = [f"{r.school_id} ({r.attendees}) exceeds the largest bus ({max(rates)})."
               for r in schools.itertuples() if r.attendees > max(rates)]
    if too_big or conflicts:
        sol.status = INFEASIBLE
        sol.notes = too_big + conflicts
        sol.runtime_s = time.perf_counter() - start
        return sol

    full_index = {sid: k for k, sid in enumerate(schools["school_id"])}
    active = schools[schools["attendees"] > 0].sort_values(["attendees", "school_id"], ascending=[False, True])
    ids = active["school_id"].tolist()
    att = active["attendees"].astype(int).tolist()
    n = len(ids)
    if n == 0:
        sol.status = OPTIMAL
        sol.notes.append("No school needs transport.")
        sol.runtime_s = time.perf_counter() - start
        return sol
    pos = {sid: k for k, sid in enumerate(ids)}
    sizes = sorted(rates)

    m = cp_model.CpModel()
    x = {(i, j): m.NewBoolVar(f"x_{i}_{j}") for i in range(n) for j in range(i + 1)}
    z = {(j, t): m.NewBoolVar(f"z_{j}_{t}") for j in range(n) for t in sizes}

    def xv(i, j):
        return x.get((i, j), 0)

    for i in range(n):
        m.AddExactlyOne(x[i, j] for j in range(i + 1))
    for j in range(n):
        m.Add(sum(z[j, t] for t in sizes) == x[j, j])
        for i in range(j + 1, n):
            m.AddImplication(x[i, j], x[j, j])
        m.Add(sum(att[i] * x[i, j] for i in range(j, n)) <= sum(t * z[j, t] for t in sizes))
        for t in sizes:  # a bus type smaller than its leader is never usable
            if t < att[j]:
                m.Add(z[j, t] == 0)

    # Redundant cut: bus counts per size must seat everyone. Lets CP-SAT use
    # integer knapsack reasoning on counts, which tightens the lower bound.
    count = {t: m.NewIntVar(0, n, f"count_{t}") for t in sizes}
    for t in sizes:
        m.Add(count[t] == sum(z[j, t] for j in range(n)))
    m.Add(sum(t * count[t] for t in sizes) >= sum(att))

    for a, b in constraints.must_share:
        ia, ib = pos[a], pos[b]
        for j in range(max(ia, ib) + 1):
            m.Add(xv(ia, j) == xv(ib, j))
    for a, b in constraints.cannot_share:
        ia, ib = pos[a], pos[b]
        for j in range(min(ia, ib) + 1):
            m.Add(xv(ia, j) + xv(ib, j) <= 1)
    for s in constraints.solo:
        i = pos[s]
        m.Add(x[i, i] == 1)
        for k in range(i + 1, n):
            m.Add(x[k, i] == 0)

    cost_expr = sum(int(round(rates[t] * COST_SCALE)) * z[j, t] for j in range(n) for t in sizes)
    lower = math.ceil(round(seat_cover_lower_bound(sum(att), rates) * COST_SCALE, 6))
    m.Add(cost_expr >= lower)  # valid cut, see seat_cover_lower_bound
    m.Minimize(cost_expr)

    class _StopAtBound(cp_model.CpSolverSolutionCallback):
        """A plan whose cost equals the seat-cover lower bound is optimal: stop there."""

        def __init__(self):
            super().__init__()
            self.hit = False

        def on_solution_callback(self):
            if self.ObjectiveValue() <= lower + 1e-6:
                self.hit = True
                self.StopSearch()

    stage1_limit = max(1.0, time_limit * (0.7 if prefer_nearby else 1.0))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = stage1_limit
    solver.parameters.num_search_workers = 8
    solver.parameters.random_seed = SEED
    if warm_start is not None:
        groups, warm_note = repair_plan(warm_start, schools, rates, constraints, miles)
        for group in groups:
            members = sorted(pos[schools["school_id"].iloc[k]] for k in group)
            leader, load = members[0], sum(att[i] for i in members)
            size = min((t for t in sizes if t >= load), key=lambda t: (rates[t], t))
            for i in members:
                for j in range(i + 1):
                    m.AddHint(x[i, j], j == leader)
            for t in sizes:
                m.AddHint(z[leader, t], t == size)
        leaders = {min(pos[schools["school_id"].iloc[k]] for k in g) for g in groups}
        for j in set(range(n)) - leaders:
            for t in sizes:
                m.AddHint(z[j, t], False)
        sol.notes.append(warm_note.replace("Warm start", "Warm start (solver hint)"))
    stopper = _StopAtBound()
    status = solver.Solve(m, stopper)

    if status == cp_model.INFEASIBLE:
        sol.status = INFEASIBLE
        sol.notes.append("CP-SAT proved that no assignment satisfies all constraints.")
        sol.runtime_s = time.perf_counter() - start
        return sol
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        sol.notes.append(f"CP-SAT stopped without a plan ({solver.StatusName(status)}). Increase the time limit.")
        sol.runtime_s = time.perf_counter() - start
        return sol

    best_cost = int(round(solver.ObjectiveValue()))
    if status == cp_model.OPTIMAL:
        sol.status = OPTIMAL
    elif best_cost <= lower:
        sol.status = OPTIMAL
        sol.notes.append(
            f"Optimality proven: cost {best_cost / COST_SCALE:.2f} equals the seat-cover lower bound "
            "(the cheapest set of buses that seats every attendee, ignoring the no-split rule)."
        )
    else:
        sol.status = FEASIBLE
        bound = max(solver.BestObjectiveBound(), lower) / COST_SCALE
        sol.notes.append(f"Time limit reached. Best lower bound on cost: {bound:.2f}.")
    assignment = {i: next(j for j in range(i + 1) if solver.Value(x[i, j])) for i in range(n)}
    types = {j: t for (j, t), var in z.items() if solver.Value(var)}

    if prefer_nearby and n > 1:
        # Stage 2: keep cost <= stage-1 cost, minimise the miles between each
        # school and its bus leader, so shared buses group nearby schools.
        m.Add(cost_expr <= best_cost)
        spread = sum(
            int(round(miles[full_index[ids[i]], full_index[ids[j]]] * 100)) * x[i, j]
            for (i, j) in x if i != j
        )
        m.Minimize(spread)
        m.ClearHints()
        for (i, j), var in x.items():
            m.AddHint(var, assignment[i] == j)
        solver2 = cp_model.CpSolver()
        solver2.parameters.max_time_in_seconds = max(1.0, time_limit - (time.perf_counter() - start))
        solver2.parameters.num_search_workers = 8
        solver2.parameters.random_seed = SEED
        status2 = solver2.Solve(m)
        if status2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            assignment = {i: next(j for j in range(i + 1) if solver2.Value(x[i, j])) for i in range(n)}
            types = {j: t for (j, t), var in z.items() if solver2.Value(var)}
            sol.notes.append(
                "Tie-break stage (group nearby schools at equal cost): "
                + ("optimal." if status2 == cp_model.OPTIMAL else "best found within time limit.")
            )
        else:
            sol.notes.append("Tie-break stage found nothing in time; stage-1 grouping kept.")

    for j, size in types.items():
        members = [ids[i] for i in range(n) if assignment[i] == j]
        order = order_stops([full_index[s] for s in members], miles)
        stops = [schools["school_id"].iloc[k] for k in order]
        sol.routes.append(BusRoute("", size, stops, sum(att[pos[s]] for s in members),
                                   route_metric(order, miles), rates[size]))
    sol.notes.append("Pickup order and miles are for display only: the fixed model's cost ignores location.")
    sol.runtime_s = time.perf_counter() - start
    return _finish(sol, schools, rates, constraints, miles)


# --------------------------------------------------------------------------- #
# Models 2 and 3: per-mile or per-hour routing
# --------------------------------------------------------------------------- #


def solve_routing(
    schools: pd.DataFrame,
    rates: dict[int, float],
    matrix: np.ndarray,
    model: str,
    constraints: Constraints | None = None,
    time_limit: float = 30.0,
    metric_unit: str = "miles",
    matrix_source: str = "straight line (haversine)",
    warm_start: list[list[str]] | None = None,
) -> Solution:
    """Heuristic routing (OR-Tools guided local search). Never claims optimality.

    ``matrix`` covers all schools in dataset order plus the event (last row).
    Costs: rate(bus size) x matrix leg, summed over each bus's legs.
    ``warm_start``: stop lists of a previous plan. It is repaired to fit the
    current inputs and used as the starting plan, so the result never costs
    more than that repaired plan.
    """
    constraints = constraints or Constraints()
    start = time.perf_counter()
    sol = Solution(model=model, status=NO_SOLUTION, time_limit_s=time_limit, metric_unit=metric_unit,
                   matrix_source=matrix_source, excluded_schools=_excluded(schools),
                   method="OR-Tools Routing: guided local search from a one-bus-per-group start")

    conflicts = check_constraints(schools, rates, constraints)
    too_big = [f"{r.school_id} ({r.attendees}) exceeds the largest bus ({max(rates)})."
               for r in schools.itertuples() if r.attendees > max(rates)]
    if too_big or conflicts:
        sol.status = INFEASIBLE
        sol.notes = too_big + conflicts
        sol.runtime_s = time.perf_counter() - start
        return sol
    if matrix.shape != (len(schools) + 1, len(schools) + 1):
        raise ValueError("Matrix must cover every school plus the event.")

    full_ids = schools["school_id"].tolist()
    act = [k for k, a in enumerate(schools["attendees"]) if a > 0]
    n = len(act)
    if n == 0:
        sol.status = OPTIMAL
        sol.notes.append("No school needs transport.")
        return sol
    demand = [0] + [int(schools["attendees"].iloc[k]) for k in act] + [0]
    event_full = len(schools)
    end_node = n + 1
    node_of = {full_ids[k]: i + 1 for i, k in enumerate(act)}

    sizes = sorted(rates)
    vehicle_size: list[int] = []
    for t in sizes:
        vehicle_size += [t] * sum(1 for d in demand[1:-1] if d <= t)
    V = len(vehicle_size)

    manager = pywrapcp.RoutingIndexManager(n + 2, V, [0] * V, [end_node] * V)
    routing = pywrapcp.RoutingModel(manager)

    def leg(a: int, b: int) -> float:
        if a == 0 or a == end_node or b == 0:
            return 0.0  # virtual start, or arcs never used in a real route
        to = event_full if b == end_node else act[b - 1]
        return float(matrix[act[a - 1], to])

    callbacks = {}
    for t in sizes:
        rate = rates[t]

        def cost_cb(i, j, rate=rate):
            return int(round(rate * leg(manager.IndexToNode(i), manager.IndexToNode(j)) * ROUTING_SCALE))

        callbacks[t] = routing.RegisterTransitCallback(cost_cb)
    for v, t in enumerate(vehicle_size):
        routing.SetArcCostEvaluatorOfVehicle(callbacks[t], v)

    demand_cb = routing.RegisterUnaryTransitCallback(lambda i: demand[manager.IndexToNode(i)])
    routing.AddDimensionWithVehicleCapacity(demand_cb, 0, vehicle_size, True, "Load")

    cp_solver = routing.solver()
    vvar = {sid: routing.VehicleVar(manager.NodeToIndex(node)) for sid, node in node_of.items()}
    for a, b in constraints.must_share:
        cp_solver.Add(vvar[a] == vvar[b])
    for a, b in constraints.cannot_share:
        cp_solver.Add(vvar[a] != vvar[b])
    for s in constraints.solo:
        for other in node_of:
            if other != s:
                cp_solver.Add(vvar[other] != vvar[s])

    # Initial plan: the repaired previous plan, or each must-share group on its own bus.
    if warm_start is not None:
        start_groups, warm_note = repair_plan(warm_start, schools, rates, constraints, matrix)
        sol.notes.append(warm_note)
        sol.method = "OR-Tools Routing: guided local search from the previous plan (warm start)"
    else:
        start_groups = [order_stops([full_ids.index(s) for s in g], matrix)
                        for g in must_share_groups([full_ids[k] for k in act], constraints)]
    free = {t: [v for v, s in enumerate(vehicle_size) if s == t] for t in sizes}
    initial: list[list[int]] = [[] for _ in range(V)]
    start_cost = 0.0
    for order in start_groups:
        load = sum(int(schools["attendees"].iloc[k]) for k in order)
        metric = route_metric(order, matrix)
        t = min((s for s in sizes if s >= load and free[s]), key=lambda s: (rates[s] * metric, s))
        initial[free[t].pop(0)] = [node_of[full_ids[k]] for k in order]
        start_cost += rates[t] * metric

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromMilliseconds(int(max(1.0, time_limit) * 1000))

    routing.CloseModelWithParameters(params)
    start_assignment = routing.ReadAssignmentFromRoutes(initial, True)
    if start_assignment is not None:
        sol.notes.append(f"Search started from a valid plan costing {start_cost:,.2f}; it can only improve on it.")
        result = routing.SolveFromAssignmentWithParameters(start_assignment, params)
    else:
        sol.notes.append("Initial one-bus-per-group plan was rejected; solving from scratch.")
        result = routing.SolveWithParameters(params)

    sol.runtime_s = time.perf_counter() - start
    if result is None:
        sol.notes.append("Routing search found no plan within the time limit.")
        return sol

    for v in range(V):
        idx = routing.Start(v)
        nodes = []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if 1 <= node <= n:
                nodes.append(node)
            idx = result.Value(routing.NextVar(idx))
        if not nodes:
            continue
        stops_full = [act[i - 1] for i in nodes]
        metric = route_metric(stops_full, matrix)
        size = vehicle_size[v]
        sol.routes.append(BusRoute("", size, [full_ids[k] for k in stops_full],
                                   sum(demand[i] for i in nodes), metric, rates[size] * metric))

    sol.status = FEASIBLE
    sol.notes.append(
        "Routing is heuristic: the plan is valid and was improved for the full time limit, "
        "but it is not proven to be the global minimum."
    )
    solver_cost = result.ObjectiveValue() / ROUTING_SCALE
    sol = _finish(sol, schools, rates, constraints, matrix)
    if abs(solver_cost - sol.total_cost) > 0.01 * max(1, len(sol.routes)):
        sol.notes.append(f"Solver objective {solver_cost:.2f} differs from recalculated {sol.total_cost:.2f} (integer rounding).")
    return sol
