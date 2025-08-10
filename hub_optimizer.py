
#!/usr/bin/env python3
"""
hub_optimizer.py

Hub-and-spoke local-improvement optimizer (greedy or batch).
Designed to run on the provided SQLite DB (transport.db) and use only
metrics computable from the database tables: stops, route_stops, routes.

Usage example (from same folder as transport.db):
    python3 hub_optimizer.py --hubs 76,99,119,135,142,145 --mode greedy --topN 5

The script outputs CSV reports into ./hub_optimizer_outputs and prints summaries.
"""

import sqlite3, os, csv, argparse, math, copy, json
from collections import defaultdict, deque
from statistics import mean, pstdev

# ----------------------- Utility functions -----------------------
def read_db(db_path):
    conn = sqlite3.connect(db_path)
    stops = { }
    cur = conn.cursor()
    # read stops
    cur.execute("SELECT * FROM stops")
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    for r in rows:
        rec = dict(zip(cols, r))
        sid = str(rec.get('stop_id') or rec.get('id') or rec.get(cols[0]))
        stops[sid] = rec
    # read route_stops
    cur.execute("SELECT * FROM route_stops")
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    route_stops = []
    for r in rows:
        rec = dict(zip(cols, r))
        route_stops.append(rec)
    # read routes (if exists)
    routes = []
    try:
        cur.execute("SELECT * FROM routes")
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        for r in rows:
            routes.append(dict(zip(cols, r)))
    except Exception:
        routes = []
    conn.close()
    return stops, route_stops, routes

def build_graph(route_stops, route_id_col='route_id', stop_id_col='stop_id', seq_col=None):
    G = defaultdict(set)
    route_sequences = defaultdict(list)
    # group by route
    groups = defaultdict(list)
    for r in route_stops:
        rid = str(r.get(route_id_col) or r.get(list(r.keys())[0]))
        groups[rid].append(r)
    for rid, rows in groups.items():
        if seq_col and seq_col in rows[0]:
            rows = sorted(rows, key=lambda x: x.get(seq_col))
        seq = [str(r.get(stop_id_col)) for r in rows]
        route_sequences[rid] = seq
        for i in range(len(seq)-1):
            a,b = seq[i], seq[i+1]
            G[a].add(b); G[b].add(a)
    return G, route_sequences

def build_graph_from_route_sequences(route_sequences):
    """Build an undirected adjacency graph from in-memory route sequences.

    This is used to evaluate candidate plans whose sequences are not persisted
    to the database, so that graph-dependent metrics can reflect the proposal.
    """
    G = defaultdict(set)
    for seq in route_sequences.values():
        if not seq:
            continue
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            G[a].add(b)
            G[b].add(a)
    return G

def bfs_distances(G, start):
    q = deque([start])
    dist = {start:0}
    parent = {start: None}
    while q:
        u = q.popleft()
        for v in G[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                parent[v] = u
                q.append(v)
    return dist, parent

def shortest_path(G, u, v):
    if u==v: return [u]
    q = deque([u]); prev = {u:None}; visited={u}
    while q:
        x = q.popleft()
        for nb in G[x]:
            if nb not in visited:
                visited.add(nb); prev[nb]=x; q.append(nb)
                if nb == v:
                    path = [v]; cur = v
                    while prev[cur] is not None:
                        cur = prev[cur]; path.append(cur)
                    path.reverse(); return path
    return None

# ----------------------- Metrics (fully computable) -----------------------
def compute_pwavg_hops(stops, G, stop_to_hub, demand_key='daily_passengers'):
    total_weight = 0.0
    weighted_sum = 0.0
    for s, (h, hops) in stop_to_hub.items():
        p = stops.get(s, {}).get(demand_key)
        try:
            p = float(p) if p is not None else 0.0
        except Exception:
            p = 0.0
        if p <= 0:
            p = 1.0  # fallback uniform
        weighted_sum += p * hops
        total_weight += p
    if total_weight == 0: return 0.0
    return weighted_sum / total_weight

def compute_avg_route_length(stops, route_sequences, demand_key='daily_passengers'):
    lens = []
    weights = []
    for rid, seq in route_sequences.items():
        length = len(seq)
        total_p = 0.0
        for s in seq:
            p = stops.get(s, {}).get(demand_key)
            try: p = float(p) if p is not None else 0.0
            except: p = 0.0
            if p <= 0: p = 1.0
            total_p += p
        lens.append(length); weights.append(total_p)
    if sum(weights) == 0:
        return mean(lens) if lens else 0.0
    return sum(l * w for l,w in zip(lens,weights)) / sum(weights)

def compute_hub_load_cv(stops, stop_to_hub, demand_key='daily_passengers'):
    hub_loads = defaultdict(float)
    for s, (h, hops) in stop_to_hub.items():
        p = stops.get(s, {}).get(demand_key)
        try: p = float(p) if p is not None else 0.0
        except: p = 0.0
        if p <= 0: p = 1.0
        hub_loads[h] += p
    vals = list(hub_loads.values())
    if not vals: return 0.0
    mu = mean(vals)
    if mu == 0: return 0.0
    sd = pstdev(vals) if len(vals)>1 else 0.0
    return sd / mu

def compute_max_hops(stop_to_hub):
    return max(hops for (_,hops) in stop_to_hub.values()) if stop_to_hub else 0

def compute_directness(G, route_sequences):
    vals = []
    for rid, seq in route_sequences.items():
        if len(seq) < 2:
            vals.append(1.0); continue
        origin, end = seq[0], seq[-1]
        sp = shortest_path(G, origin, end)
        if sp is None:
            vals.append(0.0); continue
        shortest = len(sp)-1
        route_hops = max(1, sum(1 for i in range(len(seq)-1)))
        vals.append(shortest / route_hops)
    return mean(vals) if vals else 1.0

def route_complexity_score(G, seq):
    if not seq: return 0.0
    cnt = 0
    for s in seq:
        if len(G[s]) > 2: cnt += 1
    return cnt / len(seq)

# ----------------------- Similarity & change magnitude -----------------------
def lcs_len(a, b):
    n, m = len(a), len(b)
    if n==0 or m==0: return 0
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n-1, -1, -1):
        for j in range(m-1, -1, -1):
            if a[i] == b[j]:
                dp[i][j] = 1 + dp[i+1][j+1]
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j+1])
    return dp[0][0]

def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb: return 1.0
    return len(sa & sb) / len(sa | sb)

def route_similarity(a, b):
    if not a and not b: return 1.0
    l = lcs_len(a,b)
    lcs_norm = l / max(1, max(len(a), len(b)))
    j = jaccard(a,b)
    return 0.7 * lcs_norm + 0.3 * j

# ----------------------- Plan evaluation -----------------------
def assign_stops_to_hubs(G, stops, hubs):
    hub_set = set(str(h) for h in hubs)
    hub_dists = {h: bfs_distances(G, h)[0] for h in hub_set}
    mapping = {}
    for s in stops.keys():
        best_h, best_d = None, None
        for h in hub_set:
            d = hub_dists[h].get(s, float('inf'))
            if best_h is None or d < best_d:
                best_h, best_d = h, d
            elif d == best_d:
                pd = stops.get(h, {}).get('daily_passengers', 0) or 0
                best_pd = stops.get(best_h, {}).get('daily_passengers', 0) or 0
                if pd > best_pd: best_h = h
        mapping[s] = (best_h, best_d if best_d != float('inf') else 9999)
    return mapping

def compute_all_metrics(stops, G, route_sequences, hubs, demand_key='daily_passengers'):
    stop_to_hub = assign_stops_to_hubs(G, stops, hubs)
    pwavg = compute_pwavg_hops(stops, G, stop_to_hub, demand_key=demand_key)
    avg_len = compute_avg_route_length(stops, route_sequences, demand_key=demand_key)
    hub_cv = compute_hub_load_cv(stops, stop_to_hub, demand_key=demand_key)
    max_h = compute_max_hops(stop_to_hub)
    direct = compute_directness(G, route_sequences)
    complexities = [route_complexity_score(G, seq) for seq in route_sequences.values()]
    avg_complex = mean(complexities) if complexities else 0.0
    return {'PWAvgHops': pwavg, 'AvgRouteLen': avg_len, 'HubLoadCV': hub_cv, 'MaxHops': max_h, 'Directness': direct, 'AvgComplexity': avg_complex, 'stop_to_hub': stop_to_hub}

# ----------------------- Candidate generation -----------------------
def gen_candidates_split(route_sequences, min_split_len=6):
    candidates = []
    for rid, seq in route_sequences.items():
        if len(seq) >= min_split_len:
            for cut in range(2, len(seq)-2):
                seq1 = seq[:cut+1]
                seq2 = seq[cut:]
                cand = {'type':'split', 'orig_route': rid, 'new_seqs': [seq1, seq2]}
                candidates.append(cand)
    return candidates

def gen_candidates_attach_hub(route_sequences, G, hubs, max_insert=4):
    candidates = []
    for rid, seq in route_sequences.items():
        if not seq: continue
        rep = seq[len(seq)//2]
        for h in hubs:
            h = str(h)
            path = shortest_path(G, rep, h)
            if path and 1 < len(path) <= max_insert:
                idx = seq.index(rep)
                new_seq = seq[:idx+1] + path[1:] + seq[idx+1:]
                if new_seq != seq:
                    candidates.append({'type':'attach_hub', 'orig_route': rid, 'new_seqs': [new_seq]})
    return candidates

def gen_candidates_two_opt(route_sequences):
    """Generate simple 2-opt style candidates: reverse a middle segment to reduce detours.

    Keeps the same set of stops; aims to shorten route length in graph terms and improve directness.
    """
    candidates = []
    for rid, seq in route_sequences.items():
        n = len(seq)
        if n < 6:
            continue
        # try a small set of segment reversals
        for i in range(1, n - 3):
            for j in range(i + 2, min(n - 1, i + 5)):
                new_seq = seq[:i] + list(reversed(seq[i:j])) + seq[j:]
                if new_seq != seq:
                    candidates.append({'type': 'two_opt', 'orig_route': rid, 'new_seqs': [new_seq]})
    return candidates

def generate_candidates(route_sequences, G, hubs):
    c1 = gen_candidates_split(route_sequences)
    c2 = gen_candidates_attach_hub(route_sequences, G, hubs)
    c3 = gen_candidates_two_opt(route_sequences)
    seen = set()
    allc = []
    for c in c1 + c2 + c3:
        key = json.dumps(c.get('new_seqs'))
        if key in seen: continue
        seen.add(key); allc.append(c)
    return allc

def apply_candidate(route_sequences, cand):
    new_routes = copy.deepcopy(route_sequences)
    if cand['type'] == 'split':
        orig = cand['orig_route']
        seq1, seq2 = cand['new_seqs']
        if orig in new_routes: del new_routes[orig]
        new_routes[f"{orig}_a"] = seq1
        new_routes[f"{orig}_b"] = seq2
    elif cand['type'] == 'attach_hub':
        orig = cand['orig_route']
        seq_new = cand['new_seqs'][0]
        new_routes[orig] = seq_new
    return new_routes

# ----------------------- Evaluation & selection -----------------------
def evaluate_candidate(baseline_metrics, baseline_routes, cand_routes, stops, G, hubs, weights, base_threshold=10, alpha_percent=40, change_alpha=0.7):
    # Recompute graph based on candidate sequences so graph-dependent metrics can change
    cand_G = build_graph_from_route_sequences(cand_routes)
    cand_metrics = compute_all_metrics(stops, cand_G, cand_routes, hubs)
    weights = weights or {'PWAvgHops':0.4,'AvgRouteLen':0.25,'HubLoadCV':0.15,'MaxHops':0.10,'Directness':0.10}
    improvement_ratio = 0.0
    deltas = {}
    for k, w in weights.items():
        m0 = baseline_metrics[k]; mc = cand_metrics[k]
        if m0 == 0:
            delta = 0.0
        else:
            if k == 'Directness':
                delta = (mc - m0) / m0
            else:
                delta = (m0 - mc) / m0
        deltas[k] = delta
        improvement_ratio += w * delta
    improvement_percent = 100.0 * improvement_ratio
    # change magnitude
    avg_change_costs = []
    changed_count = 0
    orig_rids = list(baseline_routes.keys())
    new_rids = list(cand_routes.keys())
    for orig in orig_rids:
        orig_seq = baseline_routes[orig]
        best_sim = -1.0
        for nr in new_rids:
            sim = route_similarity(orig_seq, cand_routes[nr])
            if sim > best_sim: best_sim = sim
        change_cost = 1.0 - best_sim
        avg_change_costs.append(change_cost)
        if change_cost > 1e-6: changed_count += 1
    avg_route_change_cost = mean(avg_change_costs) if avg_change_costs else 0.0
    fraction_changed = changed_count / max(1, len(orig_rids))
    change_magnitude = change_alpha * avg_route_change_cost + (1 - change_alpha) * fraction_changed
    required = base_threshold + alpha_percent * change_magnitude
    accepted = improvement_percent >= required
    return {'improvement_percent': improvement_percent, 'deltas': deltas, 'avg_route_change_cost': avg_route_change_cost,
            'fraction_changed': fraction_changed, 'change_magnitude': change_magnitude, 'required_percent': required,
            'accepted': accepted, 'cand_metrics': cand_metrics}

# ----------------------- Main optimizer -----------------------
def run_optimizer(db_path, hubs, mode='greedy', topN=5, max_iters=10, outputs_dir='./hub_optimizer_outputs',
                  base_threshold=(-1), alpha_percent=30, change_alpha=0.7, weights=None, demand_key='daily_passengers',
                  enforce_route_count_leq_original=True):
    os.makedirs(outputs_dir, exist_ok=True)
    stops, route_stops, routes = read_db(db_path)
    G, route_sequences = build_graph(route_stops)
    original_route_count = len(route_sequences)
    print(f"Loaded DB. original routes: {original_route_count}, stops: {len(stops)}")
    hubs = [str(h) for h in hubs]
    baseline_metrics = compute_all_metrics(stops, G, route_sequences, hubs, demand_key=demand_key)
    # Hard constraint baseline: exact set of served stops must be preserved
    baseline_served_stops = set(s for seq in route_sequences.values() for s in seq)
    weights = weights or {'PWAvgHops':0.4,'AvgRouteLen':0.25,'HubLoadCV':0.15,'MaxHops':0.10,'Directness':0.10}
    applied = []
    candidates_report = []
    current_routes = copy.deepcopy(route_sequences)
    for iteration in range(max_iters):
        print(f"Iteration {iteration+1} -- generating candidates")
        cands = generate_candidates(current_routes, G, hubs)
        print(f"  generated {len(cands)} candidates")
        evaluated = []
        for cand in cands:
            cand_routes = apply_candidate(current_routes, cand)
            if enforce_route_count_leq_original and len(cand_routes) > original_route_count:
                continue
            # Hard constraint: preserve served stops exactly (no new stops, none removed)
            cand_served_stops = set(s for seq in cand_routes.values() for s in seq)
            if cand_served_stops != baseline_served_stops:
                continue
            ev = evaluate_candidate(baseline_metrics, route_sequences, cand_routes, stops, G, hubs,
                                    weights, base_threshold=base_threshold, alpha_percent=alpha_percent, change_alpha=change_alpha)
            evaluated.append((cand, ev, cand_routes))
        accepted = [ (c,e,r) for (c,e,r) in evaluated if e['accepted'] ]
        if not accepted:
            print("No accepted candidates this iteration.")
            break
        accepted_sorted = sorted(accepted, key=lambda x: x[1]['improvement_percent'], reverse=True)
        if mode == 'batch':
            chosen = accepted_sorted[:topN]
            for c,e,r in chosen:
                candidates_report.append({'candidate': c, 'eval': e})
            print(f"Batch mode selected top {len(chosen)} candidates.")
            break
        else:
            best_c, best_eval, best_routes = accepted_sorted[0]
            print(f"Applying best candidate: type={best_c['type']}, improvement={best_eval['improvement_percent']:.2f}%, change_mag={best_eval['change_magnitude']:.3f}")
            current_routes = best_routes
            baseline_metrics = best_eval['cand_metrics']
            applied.append({'candidate': best_c, 'eval': best_eval})
            candidates_report.append({'candidate': best_c, 'eval': best_eval})
            if len(applied) >= topN:
                print("Reached topN applied changes, stopping.")
                break
    with open(os.path.join(outputs_dir, 'applied_changes.json'), 'w') as f:
        json.dump(applied, f, default=str, indent=2)
    with open(os.path.join(outputs_dir, 'candidates_report.json'), 'w') as f:
        json.dump(candidates_report, f, default=str, indent=2)
    with open(os.path.join(outputs_dir, 'final_routes.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['route_id','sequence'])
        for rid, seq in current_routes.items():
            writer.writerow([rid, ",".join(seq)])
    print("Reports written to", outputs_dir)
    return {'applied': applied, 'candidates_report': candidates_report, 'final_routes': current_routes}

# ----------------------- CLI -----------------------
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--db', default='transport.db')
    p.add_argument('--hubs', required=True, help='comma-separated hub stop_ids')
    p.add_argument('--mode', default='greedy', choices=['greedy','batch'])
    p.add_argument('--topN', type=int, default=5)
    p.add_argument('--max_iters', type=int, default=10)
    p.add_argument('--out', default='./hub_optimizer_outputs')
    p.add_argument('--allow_more_routes', action='store_true', help='Allow candidates that increase total route count (enables split proposals)')
    args = p.parse_args()
    hubs = [h.strip() for h in args.hubs.split(',') if h.strip()!='']
    enforce = not args.allow_more_routes
    run_optimizer(args.db, hubs, mode=args.mode, topN=args.topN, max_iters=args.max_iters, outputs_dir=args.out, enforce_route_count_leq_original=enforce)
