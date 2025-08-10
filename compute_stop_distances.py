#!/usr/bin/env python3
"""
compute_stop_distances.py

Reads an SQLite database with a `stops` table, extracts latitude/longitude for
stops that have coordinates, computes great-circle (haversine) distances
between every pair, and saves the result efficiently:
  - Default: writes into the same SQLite DB as table `stop_distances`
    with PRIMARY KEY (stop_id_a, stop_id_b), canonicalized so A,B == B,A.
  - Optional: also save JSON/CSV files if flags are provided.

Usage:
  python compute_stop_distances.py --db "transport (3).db"
  python compute_stop_distances.py --db transport.db
"""

import argparse
import json
import math
import os
import sqlite3
from typing import Dict, List, Optional, Tuple


def parse_float(value: Optional[object]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except Exception:
            return None
    s = str(value).strip()
    if s == "":
        return None
    # normalize possible decimal comma
    s = s.replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None


LAT_CANDIDATE_KEYS = [
    "lat", "latitude", "stop_lat", "y", "lat_deg", "lat_decimal",
]
LON_CANDIDATE_KEYS = [
    "lon", "lng", "longitude", "stop_lon", "x", "long", "lon_deg", "lon_decimal",
]


def extract_lat_lon(rec: Dict[str, object]) -> Tuple[Optional[float], Optional[float]]:
    # build lowercase lookup
    lower_to_key = {str(k).lower(): k for k in rec.keys()}
    lat_val: Optional[float] = None
    lon_val: Optional[float] = None
    for lk in LAT_CANDIDATE_KEYS:
        if lk in lower_to_key:
            lat_val = parse_float(rec[lower_to_key[lk]])
            if lat_val is not None:
                break
    for ok in LON_CANDIDATE_KEYS:
        if ok in lower_to_key:
            lon_val = parse_float(rec[lower_to_key[ok]])
            if lon_val is not None:
                break
    if lat_val is None or lon_val is None:
        return None, None
    # sanity
    if not (-90.0 <= lat_val <= 90.0 and -180.0 <= lon_val <= 180.0):
        return None, None
    return lat_val, lon_val


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0088  # mean Earth radius (km)
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return R * c


def read_stops(db_path: str) -> Dict[str, Dict[str, object]]:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("SELECT * FROM stops")
    cols = [d[0] for d in cur.description]
    out: Dict[str, Dict[str, object]] = {}
    for row in cur.fetchall():
        rec = dict(zip(cols, row))
        stop_id = str(rec.get("stop_id") or rec.get("id") or rec.get(cols[0]))
        out[stop_id] = rec
    con.close()
    return out


def compute_pair_distances(stops: Dict[str, Dict[str, object]]) -> Dict[str, float]:
    # collect only stops with coords
    pts: List[Tuple[str, float, float]] = []
    for sid, rec in stops.items():
        lat, lon = extract_lat_lon(rec)
        if lat is None or lon is None:
            continue
        pts.append((sid, lat, lon))
    distances: Dict[str, float] = {}
    n = len(pts)
    for i in range(n):
        sid_i, la_i, lo_i = pts[i]
        for j in range(i + 1, n):
            sid_j, la_j, lo_j = pts[j]
            d_km = haversine_km(la_i, lo_i, la_j, lo_j)
            # canonical unordered key so A,B == B,A
            a, b = (sid_i, sid_j) if str(sid_i) < str(sid_j) else (sid_j, sid_i)
            key = f"{a}|{b}"
            distances[key] = d_km
    return distances


def save_json_csv(distances: Dict[str, float], out_json: str, out_csv: str) -> None:
    # JSON map
    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(distances, f, ensure_ascii=False, indent=2)
    # CSV rows
    if out_csv:
        import csv
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["stop_id_a", "stop_id_b", "distance_km"])
            for key, d in distances.items():
                a, b = key.split("|", 1)
                w.writerow([a, b, f"{d:.6f}"])


def save_sqlite(db_path: str, distances: Dict[str, float], table: str = "stop_distances") -> None:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            stop_id_a TEXT NOT NULL,
            stop_id_b TEXT NOT NULL,
            distance_km REAL NOT NULL,
            PRIMARY KEY (stop_id_a, stop_id_b)
        )
        """
    )
    # Insert in batches with UPSERT to avoid duplicates if rerun
    items = []
    for key, d in distances.items():
        a, b = key.split("|", 1)
        items.append((a, b, float(d)))
    cur.execute("BEGIN")
    cur.executemany(
        f"INSERT INTO {table} (stop_id_a, stop_id_b, distance_km) VALUES (?, ?, ?)\n"
        f"ON CONFLICT(stop_id_a, stop_id_b) DO UPDATE SET distance_km=excluded.distance_km",
        items,
    )
    con.commit()
    con.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="transport.db", help="Path to SQLite database with 'stops' table")
    ap.add_argument("--table", default="stop_distances", help="SQLite table name to write distances to")
    ap.add_argument("--write-json", action="store_true", help="Also write JSON map file")
    ap.add_argument("--write-csv", action="store_true", help="Also write CSV file")
    ap.add_argument("--json", default="stop_pair_distances_km.json", help="Output JSON filename")
    ap.add_argument("--csv", default="stop_pair_distances_km.csv", help="Output CSV filename")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        print(f"Database not found: {args.db}")
        raise SystemExit(1)
    stops = read_stops(args.db)
    distances = compute_pair_distances(stops)
    # Save to SQLite (default)
    save_sqlite(args.db, distances, table=args.table)
    # Optional JSON/CSV
    if args.write_json or args.write_csv:
        save_json_csv(distances, args.json if args.write_json else "", args.csv if args.write_csv else "")
    # brief summary
    coord_stops = len({p.split("|")[0] for p in distances.keys()} | {p.split("|")[1] for p in distances.keys()})
    print(f"Computed distances for {len(distances)} pairs across ~{coord_stops} stops with coordinates.")
    print(f"Saved into SQLite table '{args.table}' in {args.db}.")
    if args.write_json or args.write_csv:
        outs = []
        if args.write_json:
            outs.append(args.json)
        if args.write_csv:
            outs.append(args.csv)
        print("Also saved:", ", ".join(outs))


if __name__ == "__main__":
    main()


