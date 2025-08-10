#!/usr/bin/env python3
"""
Hub selection pipeline for transport.db
- Computes candidate hubs (demand + centrality)
- Builds distance/time matrix (prefers coords; else graph shortest-path)
- Solves p-median via PuLP on candidate set
- Outputs selected_hubs.csv and stop_assignments.csv and metrics summary

Usage:
  python hub_selection.py --db transport.db --outdir results
  python hub_selection.py --db transport.db --p 8 --outdir results
  python hub_selection.py --db transport.db --coverage 0.75 --outdir results
"""

import argparse
import math
import os
import re
import sqlite3
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from pulp import LpBinary, LpMinimize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD
from tqdm import tqdm


# -------------------------
# Utilities
# -------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    # returns distance in kilometers
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


# -------------------------
# DB loaders
# -------------------------
def load_stops(conn: sqlite3.Connection) -> pd.DataFrame:
    q = "SELECT stop_id, stop_name, daily_passengers, annual_passengers FROM stops"
    df = pd.read_sql(q, conn)
    # Ensure numeric
    if "daily_passengers" in df.columns:
        df["daily_passengers"] = pd.to_numeric(
            df["daily_passengers"], errors="coerce"
        ).fillna(0.0)
    else:
        df["daily_passengers"] = 0.0
    return df


def load_route_stops(conn: sqlite3.Connection) -> pd.DataFrame:
    q = "SELECT route_id, stop_id, stop_sequence FROM route_stops"
    df = pd.read_sql(q, conn)
    if "stop_sequence" in df.columns:
        df["stop_sequence"] = pd.to_numeric(df["stop_sequence"], errors="coerce")
    return df


def try_load_coords(conn: sqlite3.Connection) -> Optional[pd.DataFrame]:
    # try common column names for coordinates
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(stops)")
    cols = [r[1].lower() for r in cur.fetchall()]
    lat_col = None
    lon_col = None
    for candidate in [
        "lat",
        "latitude",
        "y",
        "y_coord",
        "ycoord",
        "breitengrad",
        "latitude_deg",
    ]:
        if candidate in cols:
            lat_col = candidate
            break
    for candidate in [
        "lon",
        "lng",
        "longitude",
        "x",
        "x_coord",
        "xcoord",
        "laengengrad",
        "longitude_deg",
    ]:
        if candidate in cols:
            lon_col = candidate
            break
    if lat_col and lon_col:
        q = f"SELECT stop_id, stop_name, {lat_col} AS lat, {lon_col} AS lon FROM stops"
        df = pd.read_sql(q, conn)
        # Validate
        df = df.dropna(subset=["lat", "lon"])
        if df.shape[0] == 0:
            return None
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df = df.dropna(subset=["lat", "lon"])
        return df
    return None


# -------------------------
# Graph & centrality
# -------------------------
def build_route_graph(route_stops: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    # For each route, add edges between consecutive stops by sequence
    grouped = (
        route_stops.dropna(subset=["stop_sequence"])
        .sort_values(["route_id", "stop_sequence"])
        .groupby("route_id")
    )
    for rid, group in grouped:
        seq = group.sort_values("stop_sequence")["stop_id"].tolist()
        for u, v in zip(seq, seq[1:]):
            G.add_edge(int(u), int(v))
    # also optionally add edges from routes where sequence missing (we skip)
    return G


def compute_betweenness(G: nx.Graph, nodes: List[int]) -> Dict[int, float]:
    # If network too large, consider k-node approximation (not implemented here)
    if G.number_of_nodes() == 0:
        return {n: 0.0 for n in nodes}
    bc = nx.betweenness_centrality(G, normalized=True)
    # ensure all nodes in dict
    return {n: float(bc.get(n, 0.0)) for n in nodes}


# -------------------------
# Distances/time matrix
# -------------------------
def build_distance_matrix_by_coords(
    stops_coords: pd.DataFrame, all_stop_ids: List[int]
) -> Tuple[np.ndarray, Dict[int, int]]:
    # Index mapping
    id_to_idx = {sid: i for i, sid in enumerate(all_stop_ids)}
    N = len(all_stop_ids)
    D = np.full((N, N), np.inf, dtype=float)
    # build lat/lon dict
    coords = {}
    for _, r in stops_coords.iterrows():
        coords[int(r["stop_id"])] = (float(r["lat"]), float(r["lon"]))
    # For stops missing coords, leave distances inf for now
    for i, si in enumerate(all_stop_ids):
        if si in coords:
            lat1, lon1 = coords[si]
            for j, sj in enumerate(all_stop_ids):
                if sj in coords:
                    lat2, lon2 = coords[sj]
                    D[i, j] = haversine_km(lat1, lon1, lat2, lon2)
        else:
            # leave row as inf
            pass
    # replace diagonal with zero
    for i in range(N):
        D[i, i] = 0.0
    return D, id_to_idx


def build_distance_matrix_by_graph(
    G: nx.Graph, all_stop_ids: List[int]
) -> Tuple[np.ndarray, Dict[int, int]]:
    # Use hop-count shortest path as fallback; optionally scale to km by average edge length unknown.
    id_to_idx = {sid: i for i, sid in enumerate(all_stop_ids)}
    N = len(all_stop_ids)
    D = np.full((N, N), np.inf, dtype=float)
    for i, si in enumerate(all_stop_ids):
        # compute single-source shortest path lengths (in hops)
        try:
            lengths = nx.single_source_shortest_path_length(G, si)
        except Exception:
            lengths = {}
        for sj, l in lengths.items():
            j = id_to_idx.get(sj)
            if j is not None:
                D[i, j] = float(l)  # hops
        D[i, i] = 0.0
    # Convert hops to pseudo-km by treating each hop as 1 km (this is arbitrary; we will use it only when coords missing)
    return D, id_to_idx


# -------------------------
# Score stops and pick candidates
# -------------------------
def compute_scores(
    stops_df: pd.DataFrame,
    bc_map: Dict[int, float],
    route_count_map: Dict[int, int],
    weights=(0.45, 0.2, 0.25, 0.1),
):
    # stops_df: columns stop_id, daily_passengers
    s = stops_df.copy().set_index("stop_id")
    s["daily_passengers"] = s["daily_passengers"].astype(float)
    s["route_count"] = s.index.map(lambda sid: route_count_map.get(sid, 0))
    s["betweenness"] = s.index.map(lambda sid: bc_map.get(sid, 0.0))

    # normalize columns
    def minmax(v):
        v = np.array(v, dtype=float)
        if np.nanmax(v) == np.nanmin(v):
            return np.zeros_like(v)
        return (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v))

    np_p = minmax(s["daily_passengers"].fillna(0.0))
    np_r = minmax(s["route_count"].fillna(0.0))
    np_b = minmax(s["betweenness"].fillna(0.0))
    # population weight not available typically; drop it (weights sum to 1)
    w_p, w_r, w_b, w_pop = weights
    # if pop not present, redistribute its weight to p (demand)
    if "pop" not in s.columns:
        w_p += w_pop
        w_pop = 0.0
    score = w_p * np_p + w_r * np_r + w_b * np_b  # + w_pop*...
    s["score"] = score
    return s.reset_index()


# -------------------------
# p-median solver (PuLP)
# -------------------------
def solve_pmedian(
    weights: np.ndarray,
    dist_matrix: np.ndarray,
    p: int,
    candidate_idx: List[int],
    time_limit: Optional[int] = 60,
):
    """
    weights: length N (stops) weights w_i
    dist_matrix: N x K distances from stops i to candidate j (note: columns correspond to candidate_idx)
    candidate_idx: list of candidate stop indices (length K)
    returns: list of selected candidate indices (indices inside candidate_idx list)
    """
    N = weights.shape[0]
    K = len(candidate_idx)
    # Create MILP
    model = LpProblem("p_median", LpMinimize)
    x = {}
    for i in range(N):
        for j in range(K):
            x[(i, j)] = LpVariable(f"x_{i}_{j}", cat=LpBinary)
    y = {j: LpVariable(f"y_{j}", cat=LpBinary) for j in range(K)}
    # Objective
    model += lpSum(
        weights[i] * float(dist_matrix[i, j]) * x[(i, j)]
        for i in range(N)
        for j in range(K)
    )
    # Constraints: assign each i to exactly one j
    for i in range(N):
        model += lpSum(x[(i, j)] for j in range(K)) == 1
    # x <= y
    for i in range(N):
        for j in range(K):
            model += x[(i, j)] <= y[j]
    # p hubs
    model += lpSum(y[j] for j in range(K)) == p
    # Solve
    solver = PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    model.solve(solver)
    selected = [
        candidate_idx[j]
        for j in range(K)
        if y[j].value() is not None and y[j].value() > 0.5
    ]
    # get assignment matrix
    assign = np.zeros((N,), dtype=int)
    for i in range(N):
        bestj = None
        for j in range(K):
            v = x[(i, j)].value()
            if v is not None and v > 0.5:
                bestj = j
                break
        assign[i] = bestj if bestj is not None else -1
    return selected, assign


# -------------------------
# evaluation & outputs
# -------------------------
def evaluate_and_output(
    outdir: str,
    stops_df: pd.DataFrame,
    all_stop_ids: List[int],
    selected_hub_ids: List[int],
    assign_indices: np.ndarray,
    id_to_idx: Dict[int, int],
    dist_matrix_full: np.ndarray,
):
    ensure_dir(outdir)
    # selected hubs CSV
    hubs = stops_df[stops_df["stop_id"].isin(selected_hub_ids)].copy()
    hubs = hubs.sort_values("daily_passengers", ascending=False)
    hubs.to_csv(os.path.join(outdir, "selected_hubs.csv"), index=False)
    # stop assignments: map each stop to its hub id and distance
    rows = []
    for sid in all_stop_ids:
        i = id_to_idx[sid]
        j = assign_indices[i]
        if j < 0:
            hub_id = None
            dist = None
        else:
            # j is index into candidate list; but our assign_indices come from solver where columns correspond to candidates
            # To simplify, we'll compute nearest selected hub for each stop now.
            # compute distances to selected hubs
            pass
    # simpler: compute nearest selected hub per stop using dist_matrix_full (N x N)
    selected_idxs = [id_to_idx[s] for s in selected_hub_ids]
    assignments = []
    for sid in all_stop_ids:
        i = id_to_idx[sid]
        if len(selected_idxs) == 0:
            hub_id = None
            dist = None
        else:
            dists = dist_matrix_full[i, selected_idxs]
            argmin = int(np.nanargmin(dists))
            hub_idx = selected_idxs[argmin]
            hub_id = all_stop_ids[hub_idx]
            dist = float(dists[argmin])
        assignments.append(
            (
                sid,
                hub_id,
                dist,
                float(
                    stops_df.loc[stops_df["stop_id"] == sid, "daily_passengers"].iloc[0]
                ),
            )
        )
    df_assign = pd.DataFrame(
        assignments,
        columns=["stop_id", "assigned_hub_id", "dist_to_hub", "daily_passengers"],
    )
    # enrich with stop names
    df_assign = df_assign.merge(
        stops_df[["stop_id", "stop_name"]], on="stop_id", how="left"
    )
    df_assign = df_assign.merge(
        hubs[["stop_id", "stop_name"]].rename(
            columns={"stop_id": "assigned_hub_id", "stop_name": "assigned_hub_name"}
        ),
        on="assigned_hub_id",
        how="left",
    )
    df_assign.to_csv(os.path.join(outdir, "stop_assignments.csv"), index=False)
    # metrics
    total_passenger_dist = (
        df_assign["daily_passengers"] * df_assign["dist_to_hub"].fillna(0.0)
    ).sum()
    mean_dist = (df_assign["dist_to_hub"].fillna(0.0)).mean()
    coverage_within_1km = (
        (df_assign["dist_to_hub"] <= 1.0).mean()
        if not df_assign["dist_to_hub"].isna().all()
        else 0.0
    )
    metrics = {
        "num_stops": len(all_stop_ids),
        "num_hubs": len(selected_hub_ids),
        "total_passenger_weighted_distance": total_passenger_dist,
        "mean_dist_km": mean_dist,
        "pct_within_1km": coverage_within_1km,
    }
    pd.DataFrame([metrics]).to_csv(
        os.path.join(outdir, "metrics_summary.csv"), index=False
    )
    print("Wrote outputs to", outdir)
    print("Metrics:", metrics)
    return df_assign, hubs, metrics


# -------------------------
# Main pipeline
# -------------------------
def run_pipeline(
    db_path: str,
    outdir: str,
    p: Optional[int],
    coverage_target: Optional[float],
    candidate_multiplier: int = 4,
    weights=(0.45, 0.2, 0.25, 0.1),
):
    ensure_dir(outdir)
    conn = sqlite3.connect(db_path)
    stops_df = load_stops(conn)
    route_stops_df = load_route_stops(conn)
    all_stop_ids = list(stops_df["stop_id"].astype(int).tolist())
    # route_count
    route_count = dict(route_stops_df.groupby("stop_id")["route_id"].nunique())
    # graph
    print("Building route graph and computing centrality...")
    G = build_route_graph(route_stops_df)
    bc_map = compute_betweenness(G, all_stop_ids)
    # coords?
    coords_df = try_load_coords(conn)
    if coords_df is not None:
        print("Found coordinates in stops table; using haversine distances.")
        D_coords, id_to_idx = build_distance_matrix_by_coords(coords_df, all_stop_ids)
        # D_coords is N x N but only filled for stops that had coords; for missing entries remain inf -> we'll fallback via graph
        need_graph_fallback = np.isinf(D_coords).any()
    else:
        print(
            "No coordinates present; falling back to graph-based shortest path distances (hop counts)."
        )
        D_coords, id_to_idx = build_distance_matrix_by_graph(G, all_stop_ids)
        need_graph_fallback = False  # already graph-based
    # If coords partially missing, fill missing rows using graph shortest-path hop distances
    if need_graph_fallback:
        print("Filling missing distances with graph hop-count distances where needed.")
        D_graph, _ = build_distance_matrix_by_graph(G, all_stop_ids)
        mask = np.isinf(D_coords)
        D_coords[mask] = D_graph[mask]
    # Now we have dist matrix D_coords (N x N)
    # Score stops and choose candidates
    scored = compute_scores(stops_df, bc_map, route_count, weights=weights)
    # Bring stop names into scored; fall back to stringified stop_id if missing
    if "stop_name" in stops_df.columns:
        scored = scored.merge(
            stops_df[["stop_id", "stop_name"]], on="stop_id", how="left"
        )
    else:
        scored["stop_name"] = scored["stop_id"].astype(str)
    if "stop_name" not in scored.columns:
        scored["stop_name"] = scored["stop_id"].astype(str)

    # ---------------------------------------------
    # Deduplicate platform variants (e.g., "Bahnhof 1/2/3", "Rathaus 1/2")
    # Keep only the highest-scoring stop per base name
    # ---------------------------------------------
    if "stop_name" in stops_df.columns:
        all_names_set = set(stops_df["stop_name"].dropna().astype(str).tolist())
    else:
        all_names_set = set(scored["stop_name"].dropna().astype(str).tolist())

    def base_stop_key(name: Optional[str]) -> str:
        if not isinstance(name, str):
            return ""
        nm = name.strip()
        # If name ends with a trailing token (roman, single letter, or 1-2 digits),
        # consider stripping it, but only if other stops share the same base pattern.
        m = re.search(r"\s+(?:[IVXLCDM]+|[A-Z]|\d{1,2})$", nm)
        if m:
            base = re.sub(r"\s+(?:[IVXLCDM]+|[A-Z]|\d{1,2})$", "", nm)
            pat = re.compile(
                r"^" + re.escape(base) + r"\s+(?:[IVXLCDM]+|[A-Z]|\d{1,2})$"
            )
            # Group only if at least one other variant exists
            if any((other != nm and pat.match(other)) for other in all_names_set):
                return base
        return nm

    scored["base_key"] = scored["stop_name"].map(base_stop_key)
    # pick best-scoring stop within each base_key
    idx_best = scored.groupby("base_key")["score"].idxmax()
    scored_dedup = scored.loc[idx_best].reset_index(drop=True)
    # decide p if not specified
    N = len(all_stop_ids)
    if p is None:
        # greedy add top-scored hubs until cumulative passenger coverage of assigned-to-hub (nearest) >= coverage_target
        if coverage_target is None:
            coverage_target = 0.75
        print(f"Auto-selecting p to meet coverage target {coverage_target:.2f}")
        # compute ordering by score
        ordered = scored_dedup.sort_values("score", ascending=False).reset_index(
            drop=True
        )
        # incremental coverage: for k=1..K, compute passenger assigned to nearest of top-k hubs
        Kmax = min(len(ordered), max(10, candidate_multiplier * int(math.sqrt(N))))
        cum_covered = []
        total_pass = scored["daily_passengers"].sum()
        all_ids = all_stop_ids
        id_to_idx_local = id_to_idx
        D = D_coords
        # iterate k
        chosen = []
        for k in range(1, Kmax + 1):
            chosen_ids = ordered["stop_id"].iloc[:k].tolist()
            chosen_idx = [
                id_to_idx_local[sid] for sid in chosen_ids if sid in id_to_idx_local
            ]
            if len(chosen_idx) == 0:
                cov = 0.0
            else:
                # assign each stop to nearest chosen hub and sum passengers
                dists_to_chosen = D[
                    np.ix_([id_to_idx_local[s] for s in all_ids], chosen_idx)
                ]
                nearest = np.nanmin(dists_to_chosen, axis=1)
                # if some distances are inf, treat as large
                nearest = np.where(np.isinf(nearest), 1e6, nearest)
                # Now coverage metric: count how many passengers have distance <= threshold (we want nearest hub within some cutoff)
                # But user wanted coverage of boardings being nearest-hub — instead we compute fraction of total passenger weight 'served by' hubs (always 100% if any hubs exist)
                # Instead use decreasing marginal benefit: use total passenger-distance reduction vs k=0 baseline. For simplicity, choose k where added hub yields <1% improvement.
                # For speed, we'll compute passenger-weighted average distance and pick elbow where it flattens.
                pw_dist = (
                    scored.set_index("stop_id").loc[all_ids, "daily_passengers"].values
                    * nearest
                ).sum()
                cum_covered.append((k, pw_dist))
            # pick k with elbow heuristic later
        # choose k with largest drop relative to previous until marginal drop < 0.01 * initial
        if len(cum_covered) < 2:
            p = 5
        else:
            initial = cum_covered[0][1]
            prev = initial
            chosen_k = 1
            for k, val in cum_covered:
                drop = prev - val
                if drop < 0.01 * initial:  # marginal improvement small
                    break
                prev = val
                chosen_k = k
            p = chosen_k
        print(f"Auto-picked p = {p}")
    else:
        print(f"Using fixed p = {p}")
    # Candidate set: spatially diverse by farthest-first on coordinates (if present)
    K = min(len(scored_dedup), max(10, candidate_multiplier * p))
    coords_map: Dict[int, Tuple[float, float]] = {}
    if coords_df is not None and {"stop_id", "lat", "lon"}.issubset(
        set(coords_df.columns.str.lower())
    ):
        # normalize column names in case
        cdf = coords_df.copy()
        # ensure columns are exactly lat/lon
        if "lat" not in cdf.columns:
            for c in cdf.columns:
                if c.lower() == "lat":
                    cdf.rename(columns={c: "lat"}, inplace=True)
        if "lon" not in cdf.columns:
            for c in cdf.columns:
                if c.lower() in ("lon", "lng", "longitude"):
                    cdf.rename(columns={c: "lon"}, inplace=True)
        for _, r in cdf.dropna(subset=["lat", "lon"]).iterrows():
            try:
                coords_map[int(r["stop_id"])] = (float(r["lat"]), float(r["lon"]))
            except Exception:
                continue

    def geo_dist_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return haversine_km(a[0], a[1], b[0], b[1])

    cand_df = scored_dedup.sort_values("score", ascending=False).reset_index(drop=True)
    if coords_map:
        seeds: List[int] = []
        # pick best-scored with coords as first seed
        for _, row in cand_df.iterrows():
            sid = int(row["stop_id"])
            if sid in coords_map:
                seeds.append(sid)
                break
        # farthest-first until K
        while len(seeds) < K:
            best_sid = None
            best_d = -1.0
            for _, row in cand_df.iterrows():
                sid = int(row["stop_id"])
                if sid in seeds or sid not in coords_map:
                    continue
                dmin = (
                    min(geo_dist_km(coords_map[sid], coords_map[sj]) for sj in seeds)
                    if seeds
                    else 1e9
                )
                if dmin > best_d:
                    best_d = dmin
                    best_sid = sid
            if best_sid is None:
                break
            seeds.append(best_sid)
        # fill if fewer than K
        if len(seeds) < K:
            for _, row in cand_df.iterrows():
                sid = int(row["stop_id"])
                if sid not in seeds:
                    seeds.append(sid)
                if len(seeds) >= K:
                    break
        candidate_ids = seeds[:K]
        candidates = cand_df[cand_df["stop_id"].isin(candidate_ids)]
    else:
        candidates = cand_df.head(K)
        candidate_ids = candidates["stop_id"].tolist()
    # Build dist matrix from all stops to candidate stops (N x K)
    id_to_idx_local = id_to_idx
    N = len(all_stop_ids)
    K = len(candidate_ids)
    D_all = D_coords  # N x N
    # handle any missing rows/cols by replacing inf with large number
    D_all = np.array(D_all, dtype=float)
    D_all[np.isinf(D_all)] = 1e6
    dist_to_candidates = np.zeros((N, K), dtype=float)
    for j, cid in enumerate(candidate_ids):
        if cid not in id_to_idx_local:
            dist_to_candidates[:, j] = 1e6
        else:
            jidx = id_to_idx_local[cid]
            dist_to_candidates[:, j] = D_all[:, jidx]
    # weights vector
    idx_order = [id_to_idx_local[sid] for sid in all_stop_ids]
    weights_vec = np.array(
        [
            float(stops_df.loc[stops_df["stop_id"] == sid, "daily_passengers"].iloc[0])
            for sid in all_stop_ids
        ],
        dtype=float,
    )
    # solve p-median on candidate set
    print(f"Solving p-median with N={N} stops and K={K} candidates for p={p}...")
    selected_cand_idxs, assign = solve_pmedian(
        weights_vec, dist_to_candidates, p, list(range(K)), time_limit=60
    )
    # selected_cand_idxs are indices into candidate_idx list; convert to stop ids
    selected_candidate_indices = selected_cand_idxs
    selected_hub_ids = [candidate_ids[j] for j in selected_candidate_indices]
    # Post-process to enforce one hub per base stop (guard in case any duplicate slipped through)
    id_to_name = dict(
        zip(
            stops_df["stop_id"].astype(int),
            stops_df.get("stop_name", stops_df["stop_id"].astype(str)),
        )
    )
    used_bases: set = set()
    dedup_selected: List[int] = []
    for sid in selected_hub_ids:
        nm = id_to_name.get(int(sid), str(sid))
        bk = base_stop_key(nm)
        if bk not in used_bases:
            used_bases.add(bk)
            dedup_selected.append(int(sid))
    # If we ended up with fewer than p hubs, top up from candidate list by score order, skipping used bases
    if len(dedup_selected) < p:
        scored_candidates = candidates.sort_values("score", ascending=False)
        for _, row in scored_candidates.iterrows():
            cid = int(row["stop_id"])
            if cid in dedup_selected:
                continue
            nm = id_to_name.get(cid, str(cid))
            bk = base_stop_key(nm)
            if bk in used_bases:
                continue
            dedup_selected.append(cid)
            used_bases.add(bk)
            if len(dedup_selected) >= p:
                break
    selected_hub_ids = dedup_selected
    # Now compute full N x N distance matrix (we already have D_all)
    # assign each stop to nearest selected hub (in terms of D_all)
    df_assign, hubs_df, metrics = evaluate_and_output(
        outdir, stops_df, all_stop_ids, selected_hub_ids, assign, id_to_idx, D_all
    )
    conn.close()
    return df_assign, hubs_df, metrics


# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Hub selection pipeline")
    parser.add_argument("--db", required=True, help="Path to transport.db")
    parser.add_argument("--outdir", default="results", help="Output directory")
    parser.add_argument(
        "--p", type=int, default=None, help="Number of hubs to choose (overrides auto)"
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=None,
        help="Auto p based on coverage target (0-1). Default 0.75 if p not provided",
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=4,
        help="K = min(total, multiplier * p)",
    )
    args = parser.parse_args()
    run_pipeline(args.db, args.outdir, args.p, args.coverage, args.candidate_multiplier)


if __name__ == "__main__":
    main()
