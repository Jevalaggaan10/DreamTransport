#!/usr/bin/env python3
"""
select_hubs.py

Select K hub stops from an existing SQLite transport DB using a
passenger-weighted k-median (k-medoids) objective based on distances.

Objective (primary): minimize sum_i w_i * dist(i, nearest_hub(i))
  - w_i = daily_passengers if available else 1
  - dist taken from table `stop_distances` if present (keys are unordered pairs)
  - otherwise, computed on-the-fly via haversine using stop coordinates

Secondary constraints (optional):
  - min separation between hubs (km)

Outputs: prints chosen hub stop_ids and names, and writes selected hubs
to selected_hubs.csv and selected_hubs.json in the current directory.

Usage:
  python select_hubs.py --db "transport (3).db" --k 6
  python select_hubs.py --db transport.db --k 6 --min_hub_sep_km 0.5
"""

import argparse
import csv
import json
import math
import os
import sqlite3
from typing import Dict, List, Optional, Tuple, Set


def parse_float(v: Optional[object]) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        try:
            return float(v)
        except Exception:
            return None
    s = str(v).strip().replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None


LAT_KEYS = ["lat", "latitude", "stop_lat", "y", "lat_deg", "lat_decimal"]
LON_KEYS = ["lon", "lng", "longitude", "stop_lon", "x", "long", "lon_deg", "lon_decimal"]


def extract_lat_lon(rec: Dict[str, object]) -> Tuple[Optional[float], Optional[float]]:
    lower_to_key = {str(k).lower(): k for k in rec.keys()}
    lat, lon = None, None
    for lk in LAT_KEYS:
        if lk in lower_to_key:
            lat = parse_float(rec[lower_to_key[lk]])
            if lat is not None:
                break
    for ok in LON_KEYS:
        if ok in lower_to_key:
            lon = parse_float(rec[lower_to_key[ok]])
            if lon is not None:
                break
    if lat is None or lon is None:
        return None, None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None, None
    return lat, lon


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return R * c


def load_stops(db_path: str) -> Dict[str, Dict[str, object]]:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("SELECT * FROM stops")
    cols = [d[0] for d in cur.description]
    stops: Dict[str, Dict[str, object]] = {}
    for row in cur.fetchall():
        rec = dict(zip(cols, row))
        stop_id = str(rec.get("stop_id") or rec.get("id") or rec.get(cols[0]))
        stops[stop_id] = rec
    con.close()
    return stops


def load_route_intersections(db_path: str) -> Dict[str, int]:
    """Approximate stop centrality by counting distinct routes that serve each stop.

    Looks for table `route_stops(route_id, stop_id)` and counts unique route_id per stop_id.
    Returns a dict stop_id -> route_count. If table missing, returns empty dict.
    """
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='route_stops'")
        if not cur.fetchone():
            con.close()
            return {}
        cur.execute("SELECT stop_id, COUNT(DISTINCT route_id) AS c FROM route_stops GROUP BY stop_id")
        out = {str(r[0]): int(r[1]) for r in cur.fetchall()}
        con.close()
        return out
    except Exception:
        return {}


def load_distance_table(db_path: str, table_candidates: List[str] = None) -> Dict[Tuple[str, str], float]:
    table_candidates = table_candidates or ["stop_distances", "distances", "stop_pair_distances"]
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    # find first table that exists with required columns
    table = None
    for t in table_candidates:
        try:
            cur.execute(f"PRAGMA table_info({t})")
            cols = [r[1].lower() for r in cur.fetchall()]
            if not cols:
                continue
            if {"stop_id_a", "stop_id_b", "distance_km"}.issubset(set(cols)):
                table = t
                break
        except Exception:
            continue
    distances: Dict[Tuple[str, str], float] = {}
    if table:
        cur.execute(f"SELECT stop_id_a, stop_id_b, distance_km FROM {table}")
        for a, b, d in cur.fetchall():
            a, b = str(a), str(b)
            key = (a, b) if a < b else (b, a)
            try:
                distances[key] = float(d)
            except Exception:
                continue
    con.close()
    return distances


def build_candidate_set(stops: Dict[str, Dict[str, object]], distances: Dict[Tuple[str, str], float]) -> List[str]:
    # Candidates are stops that either have coordinates or appear in the distance table
    has_coord: Set[str] = set()
    for sid, rec in stops.items():
        lat, lon = extract_lat_lon(rec)
        if lat is not None and lon is not None:
            has_coord.add(sid)
    in_dist: Set[str] = set()
    for a, b in distances.keys():
        in_dist.add(str(a))
        in_dist.add(str(b))
    candidates = sorted(has_coord | in_dist)
    return candidates


def get_weight(rec: Dict[str, object]) -> float:
    w = rec.get("daily_passengers")
    try:
        w = float(w) if w is not None else 0.0
    except Exception:
        w = 0.0
    return w if w > 0 else 1.0


def build_hub_score_weights(db_path: str, stops: Dict[str, Dict[str, object]], candidates: List[str]) -> Dict[str, float]:
    """Composite hub desirability score used to bias k-medoids toward useful hubs.

    Components (normalized to [0,1] and combined):
      - passenger load at the stop (daily_passengers)
      - route intersections (num distinct routes at the stop)

    Returns a multiplicative bonus factor per candidate (>= 1.0) to slightly
    bias selection toward more central/high-load stops while preserving the
    primary distance objective.
    """
    # Collect raw signals
    raw_pass: Dict[str, float] = {}
    for sid in candidates:
        raw_pass[sid] = get_weight(stops.get(sid, {}))
    intersections = load_route_intersections(db_path)
    raw_int: Dict[str, float] = {sid: float(intersections.get(sid, 0)) for sid in candidates}

    # Normalize to [0,1]
    def normalize(vals: Dict[str, float]) -> Dict[str, float]:
        if not vals:
            return {k: 0.0 for k in candidates}
        m = max(vals.values())
        if m <= 0:
            return {k: 0.0 for k in candidates}
        return {k: (v / m) for k, v in vals.items()}

    p_norm = normalize(raw_pass)
    i_norm = normalize(raw_int)

    # Combine with small weights to avoid overpowering distance objective
    # bonus in [1.0, 1.0 + beta]
    beta = 0.3
    combined = {sid: 1.0 + beta * (0.7 * p_norm[sid] + 0.3 * i_norm[sid]) for sid in candidates}
    return combined


def make_distance_fn(stops: Dict[str, Dict[str, object]], distances: Dict[Tuple[str, str], float]):
    coords: Dict[str, Tuple[float, float]] = {}
    for sid, rec in stops.items():
        lat, lon = extract_lat_lon(rec)
        if lat is not None and lon is not None:
            coords[str(sid)] = (lat, lon)

    def d(a: str, b: str) -> float:
        if a == b:
            return 0.0
        key = (a, b) if a < b else (b, a)
        if key in distances:
            return distances[key]
        pa = coords.get(a)
        pb = coords.get(b)
        if pa is None or pb is None:
            return float("inf")
        return haversine_km(pa[0], pa[1], pb[0], pb[1])

    return d


def initial_greedy_medoids(candidates: List[str], k: int, d_fn, weights: Dict[str, float], hub_bonus: Dict[str, float]) -> List[str]:
    # Start with the best single medoid, then add greedily by max reduction
    # Compute cost if single medoid j: sum_i w_i * d(i, j)
    best_first = None
    best_cost = float("inf")
    for j in candidates:
        cost = 0.0
        for i in candidates:
            cost += weights[i] * d_fn(i, j)
        # apply hub desirability bonus (lower effective cost for good hubs)
        cost /= max(1e-9, hub_bonus.get(j, 1.0))
        if cost < best_cost:
            best_cost = cost
            best_first = j
    hubs = [best_first] if best_first is not None else []
    # Add next hubs greedily
    while len(hubs) < k and len(hubs) < len(candidates):
        best_j = None
        best_delta = float("inf")
        # current assignment cost
        for j in candidates:
            if j in hubs:
                continue
            # cost with hubs + {j}
            cost = 0.0
            for i in candidates:
                dist_to_set = min(d_fn(i, h) for h in hubs + [j])
                cost += weights[i] * dist_to_set
            cost /= max(1e-9, hub_bonus.get(j, 1.0))
            if cost < best_delta:
                best_delta = cost
                best_j = j
        if best_j is None:
            break
        hubs.append(best_j)
    return hubs


def total_cost(candidates: List[str], hubs: List[str], d_fn, weights: Dict[str, float]) -> float:
    cost = 0.0
    for i in candidates:
        dist_to_set = min(d_fn(i, h) for h in hubs)
        cost += weights[i] * dist_to_set
    return cost


def pam_local_swaps(candidates: List[str], hubs: List[str], d_fn, weights: Dict[str, float], hub_bonus: Dict[str, float], max_iter: int = 20) -> List[str]:
    # Partitioning Around Medoids (swap heuristic)
    current = hubs[:]
    # incorporate hub bonus as a small incentive in cost
    def cost_with_bonus(hset: List[str]) -> float:
        base = total_cost(candidates, hset, d_fn, weights)
        bonus_term = sum(hub_bonus.get(h, 1.0) for h in hset) / max(1, len(hset))
        # lower cost if hubs have better bonus
        return base / max(1e-9, bonus_term)

    current_cost = cost_with_bonus(current)
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        for m_idx, m in enumerate(current):
            for h in candidates:
                if h in current:
                    continue
                trial = current[:]
                trial[m_idx] = h
                trial_cost = cost_with_bonus(trial)
                if trial_cost + 1e-9 < current_cost:
                    current = trial
                    current_cost = trial_cost
                    improved = True
        # loop until no improving swap
    return current


def enforce_min_separation(hubs: List[str], d_fn, min_sep_km: float) -> List[str]:
    if min_sep_km <= 0:
        return hubs
    pruned: List[str] = []
    for h in hubs:
        too_close = any(d_fn(h, x) < min_sep_km for x in pruned)
        if not too_close:
            pruned.append(h)
    return pruned


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to SQLite database")
    ap.add_argument("--k", type=int, default=6, help="Number of hubs to select")
    ap.add_argument("--min_hub_sep_km", type=float, default=0.7, help="Minimum separation between hubs (km)")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        print(f"Database not found: {args.db}")
        raise SystemExit(1)

    stops = load_stops(args.db)
    distances = load_distance_table(args.db)
    candidates = build_candidate_set(stops, distances)
    if not candidates:
        print("No candidate stops with coordinates or distance entries.")
        raise SystemExit(1)

    weights = {sid: get_weight(stops[sid]) for sid in candidates}
    d_fn = make_distance_fn(stops, distances)

    # Initial greedy medoids, then refine with PAM swaps
    hub_bonus = build_hub_score_weights(args.db, stops, candidates)
    init_hubs = initial_greedy_medoids(candidates, args.k, d_fn, weights, hub_bonus)
    hubs = pam_local_swaps(candidates, init_hubs, d_fn, weights, hub_bonus)
    hubs = enforce_min_separation(hubs, d_fn, args.min_hub_sep_km)
    if len(hubs) > args.k:
        hubs = hubs[: args.k]

    # Prepare output rows
    rows = []
    for h in hubs:
        rec = stops.get(h, {})
        rows.append({
            "stop_id": h,
            "stop_name": rec.get("stop_name"),
            "daily_passengers": rec.get("daily_passengers"),
        })

    # Save outputs
    with open("selected_hubs.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    with open("selected_hubs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stop_id", "stop_name", "daily_passengers"])
        for r in rows:
            w.writerow([r["stop_id"], r["stop_name"], r["daily_passengers"]])

    # Print concise summary
    print("Chosen hubs (stop_id, stop_name):")
    for r in rows:
        print(f"  {r['stop_id']}: {r['stop_name']}")


if __name__ == "__main__":
    main()


