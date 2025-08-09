import os
import re
import sys
import csv
import math
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd


def normalize_whitespace(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return re.sub(r"\s+", " ", str(value)).strip() or None


def parse_german_number(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "-"}:
        return None
    # Remove thousand separators and normalize decimal comma
    s = s.replace(" ", "").replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None


def parse_date(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "-"}:
        return None
    # Try multiple common formats (German dd.mm.yyyy; ISO; with time)
    fmts = [
        "%d.%m.%Y",
        "%Y-%m-%d",
        "%d.%m.%y",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%Y-%m-%d %H:%M:%S",
        "%d.%m.%Y %H:%M",
        "%Y-%m-%dT%H:%M:%S",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            continue
    # Last resort: pandas
    try:
        dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def normalize_time_to_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "-"}:
        return None
    # Replace German comma time if any
    s = s.replace(",", ":")
    # If contains date and time, extract time
    m = re.search(r"(\d{1,2}:\d{2})(?::\d{2})?", s)
    if m:
        h, mnt = m.group(1).split(":")
        hh = int(h)
        mm = int(mnt)
        return f"{hh:02d}:{mm:02d}:00"
    # If numeric minutes since midnight
    if s.isdigit():
        minutes = int(s)
        hh = (minutes // 60) % 24
        mm = minutes % 60
        return f"{hh:02d}:{mm:02d}:00"
    # Already like HH:MM:SS
    if re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", s):
        parts = s.split(":")
        if len(parts) == 2:
            return f"{int(parts[0]):02d}:{int(parts[1]):02d}:00"
        return f"{int(parts[0]):02d}:{int(parts[1]):02d}:{int(parts[2]):02d}"
    return s


def time_to_seconds(value: Optional[str]) -> Optional[int]:
    t = normalize_time_to_text(value)
    if t is None:
        return None
    m = re.fullmatch(r"(\d{2}):(\d{2}):\d{2}", t)
    if not m:
        return None
    hh, mm = int(m.group(1)), int(m.group(2))
    return hh * 3600 + mm * 60


def sniff_csv_dialect(file_path: str, encoding_candidates: List[str]) -> Tuple[str, csv.Dialect]:
    for enc in encoding_candidates:
        try:
            with open(file_path, "r", encoding=enc, errors="ignore") as f:
                sample = f.read(8192)
                if not sample:
                    continue
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
                    return enc, dialect
                except Exception:
                    # Fallback: try manual delimiters by counting
                    best = None
                    best_count = -1
                    for delim in [",", ";", "\t", "|"]:
                        c = sample.count(delim)
                        if c > best_count:
                            best = delim
                            best_count = c
                    class Simple(csv.Dialect):
                        delimiter = best or ","
                        quotechar = '"'
                        doublequote = True
                        skipinitialspace = True
                        lineterminator = "\n"
                        quoting = csv.QUOTE_MINIMAL
                    return enc, Simple
        except Exception:
            continue
    # Default
    class Default(csv.Dialect):
        delimiter = ","
        quotechar = '"'
        doublequote = True
        skipinitialspace = True
        lineterminator = "\n"
        quoting = csv.QUOTE_MINIMAL
    return "utf-8", Default


def find_header_row(lines: List[str], delimiter: str) -> int:
    # Look for a row that contains any of expected header keywords
    header_keywords = [
        "linie", "route", "lin", "haltestelle", "halt", "stop", "betriebstag", "datum",
        "soll", "ist", "versp", "abweich", "delay", "fahrtdatum", "fahrtnummer", "trip",
        "sequenz", "reihenfolge", "ankunft", "abfahrt"
    ]
    for idx, line in enumerate(lines):
        cells = [c.strip().lower() for c in line.split(delimiter)]
        matches = sum(any(hk in c for hk in header_keywords) for c in cells)
        if matches >= 2 and len(cells) >= 4:
            return idx
    # fallback: first non-empty line after possible metadata
    for idx, line in enumerate(lines):
        if line.strip() != "":
            return idx
    return 0


def read_punctuality_csv(file_path: str) -> pd.DataFrame:
    enc, dialect = sniff_csv_dialect(file_path, ["utf-8-sig", "utf-8", "cp1252", "latin1"])
    with open(file_path, "r", encoding=enc, errors="ignore") as f:
        raw = f.read().splitlines()
    header_idx = find_header_row(raw[:50], getattr(dialect, "delimiter", ","))
    df = pd.read_csv(
        file_path,
        encoding=enc,
        delimiter=getattr(dialect, "delimiter", ","),
        skiprows=header_idx,
        dtype=str,
        engine="python",
    )
    # Clean column names robustly (keine .strip() auf None)
    new_cols: List[str] = []
    for col in df.columns:
        if isinstance(col, str):
            norm = normalize_whitespace(col)
            new_cols.append(norm if isinstance(norm, str) else "")
        else:
            new_cols.append(str(col))
    df.columns = new_cols
    # Drop fully empty columns
    df = df.dropna(axis=1, how="all")
    return df


def find_column(candidates: List[str], cols: List[str]) -> Optional[str]:
    cols_norm: List[Tuple[Optional[str], str]] = []
    for c in cols:
        norm = normalize_whitespace(c)
        low = norm.lower() if isinstance(norm, str) else ""
        cols_norm.append((c if isinstance(c, str) else (str(c) if c is not None else None), low))
    for cand in candidates:
        cand_low = cand.lower()
        # exact
        for orig, low in cols_norm:
            if low == cand_low:
                return orig
        # contains
        for orig, low in cols_norm:
            if cand_low in low:
                return orig
    return None


def map_punctuality_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = list(df.columns)
    mapping = {
        "route_number": find_column([
            "Linie", "Line", "Route", "Liniennummer", "Liniennr", "Linien-Nr", "Linientext"
        ], cols),
        "stop_name": find_column([
            "Haltestelle", "Halt", "Stop", "Haltestellenname", "Stop Name"
        ], cols),
        "service_date": find_column([
            "Betriebstag", "Datum", "Fahrtdatum", "Service Date", "Date"
        ], cols),
        "scheduled_time": find_column([
            "Sollzeit", "Soll-Abfahrt", "Planzeit", "Geplante Zeit", "Abfahrtszeit (Soll)", "Fahrplanzeit", "Plan", "Soll"
        ], cols),
        "actual_time": find_column([
            "Istzeit", "Ist-Abfahrt", "Istzeitpunkt", "Tatsächliche Zeit", "Abfahrtszeit (Ist)", "Ist"
        ], cols),
        "delay_minutes": find_column([
            "Verspätung", "+/-", "Abweichung", "Delay", "Minuten"
        ], cols),
        "trip_id": find_column([
            "Fahrt", "Fahrtnummer", "Trip", "Trip-ID", "Umlauf"
        ], cols),
        "stop_sequence": find_column([
            "Haltestellenreihenfolge", "Reihenfolge", "Sequenz", "Stop Sequence", "Haltesequenz"
        ], cols),
    }
    return mapping


def detect_event_text_column(df: pd.DataFrame) -> Optional[str]:
    keywords = ["verspät", "verfrüh", "ankunft", "abfahrt", "pünkt"]
    for col in df.columns:
        series = df[col].astype(str).str.lower()
        sample = series.head(300)
        if sample.apply(lambda x: any(k in x for k in keywords)).any():
            return col
    return None


def extract_passenger_tables_from_pdf(pdf_path: str) -> pd.DataFrame:
    try:
        import pdfplumber  # type: ignore
    except Exception as e:
        print(f"pdfplumber nicht verfügbar: {e}")
        return pd.DataFrame(columns=["stop_name", "route_number", "daily_passengers", "annual_passengers"])

    all_rows: List[List[str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            try:
                tables = page.extract_tables()
            except Exception:
                tables = []
            for tbl in tables or []:
                # Normalize row lengths
                max_len = max((len(r) for r in tbl if r), default=0)
                for r in tbl:
                    if r is None:
                        continue
                    row = [(c if c is not None else "") for c in r]
                    if len(row) < max_len:
                        row += [""] * (max_len - len(row))
                    all_rows.append([normalize_whitespace(c) or "" for c in row])

    if not all_rows:
        return pd.DataFrame(columns=["stop_name", "route_number", "daily_passengers", "annual_passengers"])

    # Find header row by keywords
    header_keywords = ["linie", "haltestelle", "mo", "fr", "jahr", "summe", "gesamt", "durchschnitt"]
    header_idx = 0
    best_score = -1
    for idx, row in enumerate(all_rows[:50]):
        cells = [str(c).lower() for c in row]
        score = sum(any(hk in c for hk in header_keywords) for c in cells)
        if score > best_score:
            best_score = score
            header_idx = idx
    header = all_rows[header_idx]
    body = [r for i, r in enumerate(all_rows) if i > header_idx]

    def hdr_val(i: int) -> str:
        if i < len(header) and header[i] is not None:
            try:
                return str(header[i]).lower()
            except Exception:
                return ""
        return ""
    df = pd.DataFrame(body, columns=[f"col_{i}_{hdr_val(i)}" for i in range(len(header))])
    # Attempt to map columns
    stop_col = find_column(["Haltestelle", "Haltestellenname", "Stop"], list(df.columns) + header)
    line_col = find_column(["Linie", "Liniennummer", "Line", "Route"], list(df.columns) + header)
    daily_col = find_column(["Mo-Fr", "Mo–Fr", "Durchschnitt", "Ø", "Werktag"], list(df.columns) + header)
    annual_col = find_column(["Jahr", "Summe", "Gesamt", "Jahres", "p.a."], list(df.columns) + header)

    # If mapping failed, try heuristics by value patterns
    def choose_col_by_values(candidate_cols: List[str], pattern: str) -> Optional[str]:
        regex = re.compile(pattern)
        best = None
        best_hits = -1
        for c in candidate_cols:
            hits = 0
            for v in df[c].head(200).astype(str):
                if regex.search(v):
                    hits += 1
            if hits > best_hits:
                best_hits = hits
                best = c
        return best

    if stop_col is None:
        stop_col = choose_col_by_values(list(df.columns), r"[A-Za-zÄÖÜäöüß]"
                                        )
    if line_col is None:
        line_col = choose_col_by_values(list(df.columns), r"^(\d{1,3}[A-Za-z]?)$")
    if daily_col is None:
        daily_col = choose_col_by_values(list(df.columns), r"^\d{1,3}(\.|\s)?\d{0,3}(,\d+)?$")
    if annual_col is None:
        annual_col = choose_col_by_values(list(df.columns), r"^\d{1,3}(\.|\s)?\d{3}(\.|\s)?\d{0,3}$")

    # Build normalized DataFrame
    norm_rows = []
    for _, row in df.iterrows():
        stop_name = normalize_whitespace(row.get(stop_col)) if stop_col in df.columns else None
        route_number = normalize_whitespace(row.get(line_col)) if line_col in df.columns else None
        daily = parse_german_number(row.get(daily_col)) if daily_col in df.columns else None
        annual = parse_german_number(row.get(annual_col)) if annual_col in df.columns else None
        if stop_name is None or route_number is None:
            # Some PDFs might list stop once and multiple lines in columns; skip such complex forms
            continue
        norm_rows.append({
            "stop_name": stop_name,
            "route_number": route_number,
            "daily_passengers": daily,
            "annual_passengers": annual,
        })
    if not norm_rows:
        return pd.DataFrame(columns=["stop_name", "route_number", "daily_passengers", "annual_passengers"])
    out = pd.DataFrame(norm_rows).drop_duplicates()
    return out


def build_database(
    csv_path: str,
    pdf_path: str,
    db_path: str = "transport.db",
    schema_out: str = "schema.sql",
) -> None:
    # Read inputs
    print(f"Lese CSV: {csv_path}")
    df_csv_raw = read_punctuality_csv(csv_path)
    csv_map = map_punctuality_columns(df_csv_raw)
    # Fallback-Heuristiken für spezielle Pünktlichkeitsanalyse-CSV (Positionsbasiert)
    if (csv_map.get("stop_name") is None) or (csv_map.get("service_date") is None) or (csv_map.get("delay_minutes") is None):
        event_col = detect_event_text_column(df_csv_raw)
        try:
            ncols = df_csv_raw.shape[1]
        except Exception:
            ncols = 0
        # service_date als erste Spalte annehmen
        if csv_map.get("service_date") is None and ncols >= 1:
            df_csv_raw["__service_date"] = df_csv_raw.iloc[:, 0]
            csv_map["service_date"] = "__service_date"
        # stop_name: Spalte links neben Ereignistext verwenden (Ziel-Haltestelle)
        if csv_map.get("stop_name") is None and event_col is not None:
            try:
                idx = df_csv_raw.columns.get_loc(event_col)
                if idx - 1 >= 0:
                    left_col = df_csv_raw.columns[idx - 1]
                    df_csv_raw["__stop_name"] = df_csv_raw[left_col]
                    csv_map["stop_name"] = "__stop_name"
            except Exception:
                pass
        # Falls route_number fehlt, aber typische "Linie"-Werte in Spalte 1 stehen
        if csv_map.get("route_number") is None and ncols >= 2:
            df_csv_raw["__route_number"] = df_csv_raw.iloc[:, 1]
            csv_map["route_number"] = "__route_number"

    missing = [k for k, v in csv_map.items() if v is None and k in {"route_number", "stop_name", "service_date"}]
    if missing:
        print(f"Warnung: folgende Pflichtspalten wurden nicht eindeutig gefunden: {missing}")

    # Normalize CSV
    df_csv = pd.DataFrame({
        "route_number": df_csv_raw.get(csv_map["route_number"]) if csv_map["route_number"] else None,
        "stop_name": df_csv_raw.get(csv_map["stop_name"]) if csv_map["stop_name"] else None,
        "service_date": df_csv_raw.get(csv_map["service_date"]) if csv_map["service_date"] else None,
        "scheduled_time": df_csv_raw.get(csv_map["scheduled_time"]) if csv_map["scheduled_time"] else None,
        "actual_time": df_csv_raw.get(csv_map["actual_time"]) if csv_map["actual_time"] else None,
        "delay_minutes": df_csv_raw.get(csv_map["delay_minutes"]) if csv_map["delay_minutes"] else None,
        "trip_id": df_csv_raw.get(csv_map["trip_id"]) if csv_map["trip_id"] else None,
        "stop_sequence": df_csv_raw.get(csv_map["stop_sequence"]) if csv_map["stop_sequence"] else None,
    })
    # Clean
    for col in ["route_number", "stop_name"]:
        df_csv[col] = df_csv[col].map(normalize_whitespace)
    df_csv["service_date"] = df_csv["service_date"].map(parse_date)
    df_csv["scheduled_time"] = df_csv["scheduled_time"].map(normalize_time_to_text)
    df_csv["actual_time"] = df_csv["actual_time"].map(normalize_time_to_text)
    # Delay: falls vorhanden, als Zahl interpretieren; sonst aus Soll/Ist berechnen
    if "delay_minutes" in df_csv.columns and df_csv["delay_minutes"].notna().any():
        # Kann als "HH:MM(:SS)" vorliegen -> in Minuten umrechnen
        def delay_to_minutes(v: Optional[str]) -> Optional[float]:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return None
            s = str(v).strip()
            if s == "":
                return None
            # try german number like "1,5"
            num = parse_german_number(s)
            if num is not None:
                return num
            m = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", s)
            if m:
                hh = int(m.group(1))
                mm = int(m.group(2))
                ss = int(m.group(3)) if m.group(3) else 0
                minutes = hh * 60 + mm + ss / 60.0
                # Ohne Kontext: positiv interpretieren; Vorzeichen wird ggf. durch Text nicht bekannt
                return minutes
            return None

        df_csv["delay_minutes"] = df_csv["delay_minutes"].map(delay_to_minutes)
    else:
        # Berechne Differenz (Ist - Soll) in Minuten
        def diff_minutes(row: pd.Series) -> Optional[float]:
            t_plan = time_to_seconds(row.get("scheduled_time"))
            t_act = time_to_seconds(row.get("actual_time"))
            if t_plan is None or t_act is None:
                return None
            diff = (t_act - t_plan) / 60.0
            return diff
        df_csv["delay_minutes"] = df_csv.apply(diff_minutes, axis=1)
    if "stop_sequence" in df_csv:
        df_csv["stop_sequence"] = pd.to_numeric(df_csv["stop_sequence"], errors="coerce")

    # Drop rows without essential identifiers
    df_csv = df_csv.dropna(subset=["route_number", "stop_name", "service_date"], how="any").reset_index(drop=True)

    # PDF passengers
    print(f"Lese PDF: {pdf_path}")
    df_pdf = extract_passenger_tables_from_pdf(pdf_path)
    if not df_pdf.empty:
        for col in ["route_number", "stop_name"]:
            df_pdf[col] = df_pdf[col].map(normalize_whitespace)
        df_pdf = df_pdf.dropna(subset=["route_number", "stop_name"], how="any").reset_index(drop=True)

    # Create SQLite DB
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    create_statements = {
        "stops": (
            """
            CREATE TABLE IF NOT EXISTS stops (
                stop_id INTEGER PRIMARY KEY AUTOINCREMENT,
                stop_name TEXT,
                daily_passengers REAL,
                annual_passengers REAL
            );
            """
        ),
        "routes": (
            """
            CREATE TABLE IF NOT EXISTS routes (
                route_id INTEGER PRIMARY KEY AUTOINCREMENT,
                route_number TEXT,
                route_name TEXT
            );
            """
        ),
        "route_stops": (
            """
            CREATE TABLE IF NOT EXISTS route_stops (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                route_id INTEGER REFERENCES routes(route_id),
                stop_id INTEGER REFERENCES stops(stop_id),
                stop_sequence INTEGER
            );
            """
        ),
        "performance_metrics": (
            """
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                route_id INTEGER REFERENCES routes(route_id),
                stop_id INTEGER REFERENCES stops(stop_id),
                service_date DATE,
                scheduled_time TEXT,
                actual_time TEXT,
                delay_minutes REAL
            );
            """
        ),
    }

    for name, stmt in create_statements.items():
        cur.executescript(stmt)
    conn.commit()

    # Build dictionaries for stops and routes
    stop_to_id: Dict[str, int] = {}
    route_to_id: Dict[str, int] = {}

    # Stops from CSV
    unique_stops = sorted(set([s for s in df_csv["stop_name"].dropna().astype(str)]))

    # Merge passenger info per stop from PDF (taking any route; we store passenger counts on stop level)
    stop_passenger_map: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    if not df_pdf.empty:
        # Aggregate by stop (if multiple lines have different values, pick max of annual and daily)
        grp = df_pdf.groupby("stop_name").agg({
            "daily_passengers": lambda x: float(pd.Series(x).dropna().max()) if pd.Series(x).dropna().size > 0 else None,
            "annual_passengers": lambda x: float(pd.Series(x).dropna().max()) if pd.Series(x).dropna().size > 0 else None,
        })
        for stop_name, row in grp.iterrows():
            stop_passenger_map[stop_name] = (row.get("daily_passengers"), row.get("annual_passengers"))

    # Also include stops present in PDF but not in CSV
    if not df_pdf.empty:
        for s in df_pdf["stop_name"].dropna().astype(str).unique():
            if s not in unique_stops:
                unique_stops.append(s)
    unique_stops = sorted(set(unique_stops))

    for stop_name in unique_stops:
        daily, annual = stop_passenger_map.get(stop_name, (None, None))
        cur.execute(
            "INSERT INTO stops (stop_name, daily_passengers, annual_passengers) VALUES (?, ?, ?)",
            (stop_name, daily, annual),
        )
        stop_to_id[stop_name] = cur.lastrowid

    # Routes from both CSV and PDF
    unique_routes = set()
    if not df_csv.empty and "route_number" in df_csv:
        unique_routes.update(df_csv["route_number"].dropna().astype(str).unique())
    if not df_pdf.empty and "route_number" in df_pdf:
        unique_routes.update(df_pdf["route_number"].dropna().astype(str).unique())

    for route_number in sorted(unique_routes):
        cur.execute(
            "INSERT INTO routes (route_number, route_name) VALUES (?, ?)",
            (route_number, None),
        )
        route_to_id[route_number] = cur.lastrowid

    conn.commit()

    # route_stops: determine stop_sequence
    # Prefer explicit sequence from CSV
    route_stop_sequence: Dict[Tuple[str, str], int] = {}
    if "stop_sequence" in df_csv.columns and df_csv["stop_sequence"].notna().any():
        tmp = df_csv.dropna(subset=["route_number", "stop_name", "stop_sequence"]).copy()
        tmp["stop_sequence"] = pd.to_numeric(tmp["stop_sequence"], errors="coerce")
        tmp = tmp.dropna(subset=["stop_sequence"]).groupby(["route_number", "stop_name"])['stop_sequence'].median().reset_index()
        for _, row in tmp.iterrows():
            route_stop_sequence[(row["route_number"], row["stop_name"])]= int(round(float(row["stop_sequence"])) )

    # If not available, infer by median time-of-day per stop within each route
    missing_pairs = set((r, s) for r in unique_routes for s in unique_stops if (r, s) not in route_stop_sequence)
    if missing_pairs:
        time_basis = df_csv.copy()
        # choose scheduled_time, else actual_time
        time_basis["time_seconds"] = time_basis.apply(
            lambda r: time_to_seconds(r["scheduled_time"]) if pd.notna(r.get("scheduled_time")) else time_to_seconds(r.get("actual_time")),
            axis=1,
        )
        time_basis = time_basis.dropna(subset=["route_number", "stop_name", "time_seconds"])  # type: ignore
        if not time_basis.empty:
            tmp = time_basis.groupby(["route_number", "stop_name"])['time_seconds'].median().reset_index()
            # Rank within each route
            tmp["stop_sequence"] = tmp.groupby("route_number")["time_seconds"].rank(method="dense").astype(int)
            for _, row in tmp.iterrows():
                key = (row["route_number"], row["stop_name"])  # type: ignore
                route_stop_sequence.setdefault(key, int(row["stop_sequence"]))

    # Insert route_stops entries
    inserted_pairs = set()
    # Build mapping of stops present per route from CSV
    pairs_from_csv = df_csv.dropna(subset=["route_number", "stop_name"]).copy()
    if not pairs_from_csv.empty:
        for _, row in pairs_from_csv[["route_number", "stop_name"]].drop_duplicates().iterrows():
            route_number = row["route_number"]
            stop_name = row["stop_name"]
            if route_number not in route_to_id or stop_name not in stop_to_id:
                continue
            seq = route_stop_sequence.get((route_number, stop_name))
            if seq is None:
                # place at end if unknown
                seq = 9999
            cur.execute(
                "INSERT INTO route_stops (route_id, stop_id, stop_sequence) VALUES (?, ?, ?)",
                (route_to_id[route_number], stop_to_id[stop_name], int(seq)),
            )
            inserted_pairs.add((route_number, stop_name))

    # performance_metrics from CSV
    batch = []
    for _, row in df_csv.iterrows():
        route_number = row.get("route_number")
        stop_name = row.get("stop_name")
        if route_number not in route_to_id or stop_name not in stop_to_id:
            continue
        batch.append((
            route_to_id[route_number],
            stop_to_id[stop_name],
            row.get("service_date"),
            row.get("scheduled_time"),
            row.get("actual_time"),
            float(row.get("delay_minutes")) if pd.notna(row.get("delay_minutes")) else None,
        ))
    cur.executemany(
        "INSERT INTO performance_metrics (route_id, stop_id, service_date, scheduled_time, actual_time, delay_minutes) VALUES (?, ?, ?, ?, ?, ?)",
        batch,
    )
    conn.commit()

    # Vacuum/Analyze for good measure
    try:
        cur.execute("VACUUM")
        cur.execute("ANALYZE")
    except Exception:
        pass

    # Export schema
    with open(schema_out, "w", encoding="utf-8") as f:
        for row in cur.execute("SELECT sql FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"):
            if row[0]:
                f.write(row[0].strip() + ";\n\n")

    # Print counts and first 5 rows of each table
    for table in ["stops", "routes", "route_stops", "performance_metrics"]:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        print(f"{table}: {count} Zeilen")
        cur.execute(f"SELECT * FROM {table} LIMIT 5")
        rows = cur.fetchall()
        col_names = [d[0] for d in cur.description]
        print(json.dumps({"columns": col_names, "rows": rows}, ensure_ascii=False, indent=2))

    conn.close()
    print(f"Fertig. DB: {db_path}, Schema: {schema_out}")


def main():
    # Default paths based on workspace
    default_csv = "20250805_Neumarkt 01.01.2025 - 31.07.2025.xlsx - Pünktlichkeitsanalyse.csv"
    default_pdf = "2024 Haltestellenbezogene Fahrgastzahlen.pdf"

    csv_path = sys.argv[1] if len(sys.argv) > 1 else default_csv
    pdf_path = sys.argv[2] if len(sys.argv) > 2 else default_pdf
    db_path = sys.argv[3] if len(sys.argv) > 3 else "transport.db"
    schema_out = sys.argv[4] if len(sys.argv) > 4 else "schema.sql"

    if not os.path.exists(csv_path):
        print(f"CSV nicht gefunden: {csv_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(pdf_path):
        print(f"PDF nicht gefunden (fahre ohne Fahrgastzahlen fort): {pdf_path}")
    build_database(csv_path, pdf_path, db_path, schema_out)


if __name__ == "__main__":
    main()


