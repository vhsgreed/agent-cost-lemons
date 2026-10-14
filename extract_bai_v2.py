"""Extractor for the one Bai et al. archive in OpenHands-SDK format (claude-sonnet-4.5_new_v2_4runs).

Per run x instance, per-call usage lives in conversations/<inst>/workspace/conversations/<id>/base_state.json
(.stats.usage_to_metrics.default.token_usages[] and .costs[]); the same file may also sit inside a nested
conversations/<inst>.tar.gz. Resolved labels: logs/run_evaluation/<run>/OpenHands/<inst>/report.json.
Writes the same row schema as extract_bai.py. Prints counts only.
"""
import gzip, io, json, os, re, tarfile, time, urllib.request

REPO = "loong0814/openhands_trajectories"
REV = "fa9cbb063f770df596da95af24f7af3b8f595778"
NAME = "claude-sonnet-4.5_new_v2_4runs"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")
BASE = re.compile(r"/(?P<run>[^/]+)/conversations/(?P<inst>[^/]+)/workspace/conversations/[^/]+/base_state\.json$")
NESTED = re.compile(r"/(?P<run>[^/]+)/conversations/(?P<inst>[^/]+)\.tar\.gz$")
REPORT = re.compile(r"/run_evaluation/(?P<run>[^/]+)/OpenHands/(?P<inst>[^/]+)/report\.json$")


def rows_from_state(d, run, inst):
    m = ((d.get("stats") or {}).get("usage_to_metrics") or {}).get("default") or {}
    tus, costs = m.get("token_usages") or [], m.get("costs") or []
    out = []
    for j, u in enumerate(tus):
        out.append({"run": run, "inst": inst, "ts": float(j),
                    "p": u.get("prompt_tokens", 0) or 0, "o": u.get("completion_tokens", 0) or 0,
                    "cached": 0, "cw": u.get("cache_write_tokens", 0) or 0, "cr": u.get("cache_read_tokens", 0) or 0,
                    "reason": u.get("reasoning_tokens", 0) or 0,
                    "cost": (costs[j].get("cost", 0.0) if j < len(costs) else 0.0) or 0.0})
    return out, len(costs)


def main():
    dst = os.path.join(OUT, f"bai_calls_{NAME}.jsonl.gz")
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/{NAME}.tar.gz"
    t0, done, n, mism, bad_nested = time.time(), set(), 0, 0, 0
    resolved = {}
    with urllib.request.urlopen(url, timeout=120) as resp, gzip.open(dst, "wt") as out, \
            tarfile.open(fileobj=resp, mode="r|gz") as tar:
        for m in tar:
            if not m.isfile():
                continue
            b, nt, rp = BASE.search(m.name), NESTED.search(m.name), REPORT.search(m.name)
            if rp:
                d = json.loads(tar.extractfile(m).read())
                for inst, v in d.items():
                    resolved.setdefault(rp["run"], {})[inst] = bool(v.get("resolved"))
                continue
            states = []
            if b:
                states.append((b["run"], b["inst"], tar.extractfile(m).read()))
            elif nt:
                try:
                    inner = tarfile.open(fileobj=io.BytesIO(tar.extractfile(m).read()), mode="r:*")
                    for im in inner:
                        if im.isfile() and im.name.endswith("base_state.json"):
                            states.append((nt["run"], nt["inst"], inner.extractfile(im).read()))
                except (tarfile.TarError, EOFError, OSError):
                    bad_nested += 1
            for run, inst, raw in states:
                if (run, inst) in done:
                    continue
                done.add((run, inst))
                rows, nc = rows_from_state(json.loads(raw), run, inst)
                mism += nc != len(rows)
                for r in rows:
                    out.write(json.dumps(r) + "\n")
                n += len(rows)
                if len(done) % 200 == 0:
                    print(f"{NAME}: {len(done)} runs, {n} calls, {time.time()-t0:.0f}s", flush=True)
    for run, d in resolved.items():
        json.dump({"resolved_ids": sorted(i for i, v in d.items() if v), "completed_ids": sorted(d)},
                  open(os.path.join(OUT, f"bai_report_{NAME}_{run}.json"), "w"))
    open(dst + ".done", "w").write(json.dumps({"runs": len(done), "calls": n, "cost_len_mismatch": mism,
                                                "unreadable_nested_tars": bad_nested,
                                                "secs": round(time.time() - t0)}))
    print(f"{NAME}: DONE {len(done)} runs, {n} calls, {mism} cost/usage length mismatches", flush=True)


if __name__ == "__main__":
    main()
