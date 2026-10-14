"""Part A (reproduction of Bai et al. 2604.22750) and Part B (lemons test). See PLAN.md.

Run once with the agentlogs venv:
  ~/projects/cost-predictor-spike/agentlogs/.venv/bin/python analyze.py          # registered run
  ~/projects/cost-predictor-spike/agentlogs/.venv/bin/python analyze.py --smoke  # synthetic data only
"""
import glob, gzip, json, os, pickle, random, re, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OWN = os.path.expanduser("~/projects/fixed-price-inference")
OWN_SETS = ("SWE-smith", "SWE-rebench-OH", "SWE-agent")
SEED = 20261014
LAMBDAS = (0.0, 0.1, 0.3)
MIN_TASKS_GROUP = 50          # groups with fewer multi-run tasks are not scored in Part B
MIN_CELL = 5                  # B2 feature-cell minimum (fit tasks)
SONNET45 = ("claude-sonnet-4.5_4runs", "claude-sonnet-4.5_new_v2_4runs")
DIFF3 = {"<15 min fix": 0, "15 min - 1 hour": 1, "1-4 hours": 2, ">4 hours": 2}


# ---------------------------------------------------------------- loading
def run_no(run_dir):
    m = re.search(r"run_?(\d+)$", run_dir)
    return int(m.group(1)) if m else None


def load_bai():
    """-> {archive: {(inst, k): {"tok":, "cost":, "res": bool|None}}}"""
    out = {}
    for f in sorted(glob.glob(os.path.join(DATA, "bai_calls_*.jsonl.gz"))):
        arch = os.path.basename(f)[len("bai_calls_"):-len(".jsonl.gz")]
        if not os.path.exists(f + ".done"):
            print("WARNING incomplete archive skipped:", arch); continue
        runs = defaultdict(lambda: {"tok": 0, "cost": 0.0, "calls": 0})
        with gzip.open(f, "rt") as fh:
            for line in fh:
                c = json.loads(line)
                k = run_no(c["run"])
                r = runs[(c["inst"], k)]
                r["tok"] += c["p"] + c["o"]; r["cost"] += c["cost"]; r["calls"] += 1
        res = {}
        for rf in glob.glob(os.path.join(DATA, f"bai_report_{arch}_*.json")):
            k = run_no(rf[:-5])
            d = json.load(open(rf))
            res[k] = (set(d.get("resolved_ids", [])), set(d.get("submitted_ids", [])) | set(d.get("completed_ids", [])))
        for (inst, k), r in runs.items():
            r["res"] = (inst in res[k][0]) if k in res else None
        out[arch] = dict(runs)
    return out


def load_meta():
    d = json.load(open(os.path.join(DATA, "verified_meta.json")))
    return {r["instance_id"]: r for r in d["rows"]}


def load_own(ds):
    rows = pickle.load(open(os.path.join(OWN, f"cache_{ds}.pkl"), "rb"))
    return [(g, i, float(sum(s)), float(p)) for g, i, s, p, o in rows]


def repo_of(inst):
    return re.sub(r"-\d+$", "", inst.split(".")[0])


# ---------------------------------------------------------------- stats
def kendall_tau_b(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n, conc, disc, tx, ty = len(x), 0, 0, 0, 0
    for i in range(0, n - 1, 2000):
        xi, yi = x[i:i + 2000, None], y[i:i + 2000, None]
        dx, dy = np.sign(x[None, :] - xi), np.sign(y[None, :] - yi)
        mask = np.arange(n)[None, :] > np.arange(i, min(i + 2000, n))[:, None]
        p = (dx * dy)[mask]
        conc += int((p > 0).sum()); disc += int((p < 0).sum())
        tx += int(((dx == 0) & (dy != 0))[mask].sum()); ty += int(((dy == 0) & (dx != 0))[mask].sum())
    return (conc - disc) / np.sqrt((conc + disc + tx) * (conc + disc + ty))


def icc1(groups):
    """groups: list of arrays (log cost per run of one task, len >= 2). One-way random effects ICC(1)."""
    k = len(groups); ns = np.array([len(g) for g in groups]); N = ns.sum()
    allv = np.concatenate(groups); gm = allv.mean()
    means = np.array([g.mean() for g in groups])
    ssb = (ns * (means - gm) ** 2).sum(); ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - (ns ** 2).sum() / N) / (k - 1)
    return float((msb - msw) / (msb + (n0 - 1) * msw))


def icc_ci(groups, rng, nb=500):
    v = [icc1([groups[j] for j in rng.integers(0, len(groups), len(groups))]) for _ in range(nb)]
    return [float(np.quantile(v, .05)), float(np.quantile(v, .95))]


def public_r2(tasks, feats, rng_seed=SEED):
    """tasks: {inst: mean log cost}; feats: {inst: tuple of feature keys, most to least specific}.
    Table model fit on 70% of tasks (by instance id), out-of-sample R^2 on 30%."""
    ids = sorted(tasks); random.Random(rng_seed).shuffle(ids)
    fit, test = ids[:int(.7 * len(ids))], ids[int(.7 * len(ids)):]
    gm = float(np.mean([tasks[i] for i in fit]))
    levels = len(next(iter(feats.values())))
    tab = [defaultdict(list) for _ in range(levels)]
    for i in fit:
        for L in range(levels):
            tab[L][feats[i][L]].append(tasks[i])

    def pred(i):
        for L in range(levels):
            v = tab[L].get(feats[i][L])
            if v and len(v) >= MIN_CELL:
                return float(np.mean(v))
        return gm
    y = np.array([tasks[i] for i in test]); yh = np.array([pred(i) for i in test])
    return float(1 - ((y - yh) ** 2).sum() / ((y - gm) ** 2).sum()), len(fit), len(test)


def market(task_runs, lam, oracle=False):
    """task_runs: list of arrays of linear cost, ordered; run 0 = the buyer's earlier run (k=1).
    Current run = each of runs 1.. (equal weight). Returns uptake share of tasks, p*/pooled mean."""
    est = np.array([r[1:].mean() if oracle else r[0] for r in task_runs])
    act = np.array([r[1:].mean() for r in task_runs])     # expected actual cost per task over current runs
    pooled = act.mean()
    p = (1 + lam) * pooled
    for _ in range(200):
        take = p <= (1 + lam) * est
        if not take.any():
            return 0.0, None
        newp = act[take].mean()
        if abs(newp - p) <= 1e-9 * pooled:
            break
        p = newp
    take = p <= (1 + lam) * est
    return float(take.mean()), float(p / pooled)


# ---------------------------------------------------------------- Part A
def part_a(bai, meta, rng):
    full = {}
    for arch, runs in bai.items():
        by = defaultdict(dict)
        for (inst, k), r in runs.items():
            if r["calls"] > 0 and k is not None:
                by[inst][k] = r
        four = {i: v for i, v in by.items() if len(v) == 4}
        full[arch] = four
    complete = sorted(a for a, v in full.items() if len(v) >= 450)
    out = {"archives": {a: len(v) for a, v in full.items()}, "complete": complete}
    s45 = [a for a in SONNET45 if a in complete]
    base = [a for a in complete if a not in SONNET45]
    variants = [base + [a] for a in s45] or [base]
    out["variants"] = variants
    res = []
    for models in variants:
        r = {"models": models}
        ratios, cells = [], []        # cells: (model, inst, mean tok)
        for a in models:
            for inst, v in full[a].items():
                t = np.array([v[k]["tok"] for k in sorted(v)], float)
                if t.min() > 0:
                    ratios.append(t.max() / t.min())
                cells.append((a, inst, t.mean()))
        ratios = np.array(ratios)
        r["A1_median_maxmin"] = float(np.median(ratios)); r["A1"] = 1.5 <= r["A1_median_maxmin"] <= 2.5
        r["A2_max_maxmin"] = float(ratios.max()); r["A2"] = 20 <= r["A2_max_maxmin"] <= 45
        r["A2_share_above_10x"] = float((ratios > 10).mean()); r["n_cells"] = int(len(ratios))
        cm = [(DIFF3[meta[i]["difficulty"]], t, a) for a, i, t in cells if i in meta]
        d = np.array([c[0] for c in cm]); t = np.array([c[1] for c in cm])
        r["A3_tau_b"] = float(kendall_tau_b(d, t)); r["A3"] = abs(r["A3_tau_b"] - 0.32) <= 0.05
        r["A3_per_model"] = {a: float(kendall_tau_b(d[[c[2] == a for c in cm]], t[[c[2] == a for c in cm]])) for a in models}
        easy, hard = t[d == 0], t[d == 2]
        r["A4_easy_above_hard_mean"] = float((easy > hard.mean()).mean())
        r["A4_hard_below_easy_mean"] = float((hard < easy.mean()).mean())
        r["A4"] = abs(r["A4_easy_above_hard_mean"] - .067) <= .03 and abs(r["A4_hard_below_easy_mean"] - .111) <= .03
        diffs, per_model, insts = {}, {}, []
        for a in models:
            dm = []
            for inst, v in full[a].items():
                ks = sorted(v, key=lambda k: (v[k]["tok"], k))
                lo, hi = v[ks[0]]["res"], v[ks[-1]]["res"]
                if lo is None or hi is None:
                    continue
                dm.append(int(hi) - int(lo)); diffs[(a, inst)] = int(hi) - int(lo)
            per_model[a] = float(np.mean(dm)) if dm else None
        r["A5_per_model_max_minus_min"] = per_model
        neg = sum(1 for v in per_model.values() if v is not None and v < 0)
        byinst = defaultdict(list)
        for (a, i), v in diffs.items():
            byinst[i].append(v)
        ids = sorted(byinst)
        boot = []
        for _ in range(2000):
            pick = rng.integers(0, len(ids), len(ids))
            vals = [x for j in pick for x in byinst[ids[j]]]
            boot.append(np.mean(vals))
        r["A5_pooled"] = float(np.mean(list(diffs.values())))
        r["A5_ci95"] = [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]
        r["A5_models_negative"] = neg
        r["A5"] = neg >= 6 and r["A5_ci95"][1] < 0
        res.append(r)
    out["variant_results"] = res
    # A6
    a6 = {}
    gpt5 = next((a for a in full if a.startswith("gpt-5_")), None)
    kimi = next((a for a in full if a.startswith("kimi")), None)
    for s in s45:
        common = set(full[gpt5]) & set(full[kimi]) & set(full[s]) if gpt5 and kimi else set()
        if not common:
            a6[s] = None; continue
        m = {a: float(np.mean([full[a][i][k]["tok"] for i in common for k in full[a][i]])) for a in (gpt5, kimi, s)}
        a6[s] = {"tasks": len(common), "mean_tok": m, "kimi_minus_gpt5": m[kimi] - m[gpt5],
                 "sonnet_minus_gpt5": m[s] - m[gpt5],
                 "pass": m[kimi] - m[gpt5] > 1.5e6 and m[s] - m[gpt5] > 1.5e6}
    out["A6"] = a6
    claims = {}
    for c in ("A1", "A2", "A3", "A4", "A5"):
        claims[c] = all(r[c] for r in res) if res else None
    claims["A6"] = (all(v and v["pass"] for v in a6.values()) if a6 else None)
    out["claims"] = claims
    ev = [v for v in claims.values() if v is not None]
    out["reproduced_count"] = f"{sum(ev)}/{len(ev)}"
    out["overall_reproduced"] = sum(ev) >= 5 and len(ev) == 6
    return out


# ---------------------------------------------------------------- Part B
def part_b_group(task_runs_lin, feats, rng):
    """task_runs_lin: {inst: array of linear costs (ordered, buyer's earlier run first)}."""
    multi = {i: r for i, r in task_runs_lin.items() if len(r) >= 2 and (r > 0).all()}
    if len(multi) < MIN_TASKS_GROUP:
        return None
    logs = [np.log(r) for r in multi.values()]
    out = {"tasks": len(multi), "runs": int(sum(len(r) for r in multi.values()))}
    out["icc"] = icc1(logs); out["icc_ci90"] = icc_ci(logs, rng)
    r2, nf, nt = public_r2({i: float(np.log(r).mean()) for i, r in multi.items()}, {i: feats[i] for i in multi})
    out["public_r2"] = r2
    out["private_share"] = out["icc"] * (1 - min(max(r2, 0.0), 1.0))
    runs = list(multi.values())
    for lam in LAMBDAS:
        u, p = market(runs, lam); uo, po = market(runs, lam, oracle=True)
        out[f"lam{lam}"] = {"uptake": u, "p_over_mean": p, "oracle_uptake": uo, "oracle_p_over_mean": po}
    return out


def part_b(bai, meta, rng, complete):
    out = {"bai": {}, "own": {}}
    for arch in complete:
        runs = bai[arch]
        zero = sum(1 for r in runs.values() if r["calls"] > 0 and r["cost"] <= 0) / max(1, sum(1 for r in runs.values() if r["calls"] > 0))
        field = "tok" if zero > 0.05 else "cost"
        by = defaultdict(dict)
        for (inst, k), r in runs.items():
            if r["calls"] > 0 and k is not None:
                by[inst][k] = r[field]
        tr = {i: np.array([v[k] for k in sorted(v)], float) for i, v in by.items()}
        ps = {i: meta[i]["ps_chars"] for i in tr if i in meta}
        cuts = np.quantile(list(ps.values()), np.linspace(.1, .9, 9))
        feats = {}
        for i in tr:
            m = meta.get(i, {"repo": "?", "difficulty": "?", "ps_chars": 0})
            dec = int(np.searchsorted(cuts, m["ps_chars"]))
            dif = DIFF3.get(m["difficulty"], -1)
            feats[i] = ((m["repo"], dif, dec), (m["repo"], dif), (m["repo"],))
        g = part_b_group(tr, feats, rng)
        if g:
            g["cost_field"] = field; g["zero_cost_share"] = zero
        out["bai"][arch] = g
    for ds in OWN_SETS:
        rows = load_own(ds)
        by = defaultdict(lambda: defaultdict(list)); pl = defaultdict(lambda: defaultdict(list))
        for g, i, c, p in rows:
            by[g][i].append(c); pl[g][i].append(p)
        for g in sorted(by):
            tr = {i: np.array(v, float) for i, v in by[g].items()}
            pm = {i: float(np.mean(pl[g][i])) for i in tr}
            cuts = np.quantile(list(pm.values()), np.linspace(.1, .9, 9))
            feats = {i: ((repo_of(i), int(np.searchsorted(cuts, pm[i]))), (repo_of(i),), (int(np.searchsorted(cuts, pm[i])),)) for i in tr}
            r = part_b_group(tr, feats, rng)
            if r:
                out["own"][f"{ds} | {g}"] = r
    def score(groups):
        gs = [v for v in groups.values() if v]
        if not gs:
            return None
        icc = float(np.median([v["icc"] for v in gs])); up = float(np.median([v["lam0.1"]["uptake"] for v in gs]))
        return {"groups": len(gs), "median_icc": icc, "H_B1": icc >= 0.5,
                "median_uptake_lam0.1": up, "H_B2": up < 0.5,
                "collapsed_groups_lam0.1": sum(1 for v in gs if v["lam0.1"]["uptake"] < 0.2)}
    out["score_bai"] = score(out["bai"]); out["score_own"] = score(out["own"])
    return out


# ---------------------------------------------------------------- smoke
def smoke():
    rng = np.random.default_rng(SEED)
    # ICC recovery: between-task sd 1, within sd 0.5 -> ICC 0.8
    groups = [rng.normal(rng.normal(0, 1), .5, 4) for _ in range(2000)]
    print("smoke ICC (expect ~0.80):", round(icc1(groups), 3))
    print("smoke tau_b perfect (expect 1):", kendall_tau_b([1, 2, 3, 4], [1, 2, 3, 5]),
          "reversed (expect -1):", kendall_tau_b([1, 2, 3], [3, 2, 1]))
    # market: no private info (all tasks identical) -> full uptake at lam>0, p*=1
    same = [np.exp(rng.normal(0, .5, 4)) for _ in range(500)]
    print("smoke market no-private-info lam0.1 (expect high uptake, p~1):", market(same, .1))
    hetero = [np.exp(rng.normal(m, .2, 4)) for m in rng.normal(0, 1.5, 500)]
    print("smoke market strong-private-info lam0.1 (expect low uptake, p>>1):", market(hetero, .1))
    feats = {i: (("r", i % 3), ("r",)) for i in range(300)}
    tasks = {i: float(i % 3) + rng.normal(0, .1) for i in range(300)}
    print("smoke public R2 (expect ~0.98):", round(public_r2(tasks, feats)[0], 3))
    print("smoke repo_of:", repo_of("getmoto__moto.694ce1f4.pr_5752"), repo_of("django__django-11099"),
          repo_of("scikit-learn__scikit-learn-10844"))


def main():
    if "--smoke" in sys.argv:
        smoke(); return
    rng = np.random.default_rng(SEED)
    meta = load_meta(); bai = load_bai()
    A = part_a(bai, meta, rng)
    B = part_b(bai, meta, rng, A["complete"])
    json.dump({"A": A, "B": B}, open(os.path.join(HERE, "results.json"), "w"), indent=1, default=str)
    print(json.dumps({"A_claims": A["claims"], "A_reproduced": A["reproduced_count"], "A_overall": A["overall_reproduced"],
                      "B_bai": B["score_bai"], "B_own": B["score_own"]}, indent=1))


if __name__ == "__main__":
    main()
