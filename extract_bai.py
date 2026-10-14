"""Stream Bai et al. (arXiv 2604.22750) OpenHands trajectories from HF and keep only per-call usage.

Writes data/bai_calls_<archive>.jsonl.gz (one row per LLM call) and data/bai_report_<archive>_<run>.json
(SWE-bench resolved ids). Prints only progress counts, never token or cost values: the analysis is
pre-registered in PLAN.md and must not be seen before it is frozen.

Usage: python3 extract_bai.py <archive-name-without-.tar.gz>
"""
import gzip, json, os, re, sys, tarfile, time, urllib.request

REPO = "loong0814/openhands_trajectories"
REV = "fa9cbb063f770df596da95af24f7af3b8f595778"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")
CALL = re.compile(r"^[^/]+/(?P<run>[^/]+)/llm_completions/(?P<inst>[^/]+)/[^/]*?-(?P<ts>\d+\.\d+)\.json$")
REPORT = re.compile(r"^[^/]+/(?P<run>[^/]+)/report\.json$")


def main(name):
    os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, f"bai_calls_{name}.jsonl.gz")
    if os.path.exists(dst + ".done"):
        print(name, "already done"); return
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/{name}.tar.gz"
    t0, n, bad = time.time(), 0, 0
    with urllib.request.urlopen(url, timeout=120) as resp, gzip.open(dst, "wt") as out, \
            tarfile.open(fileobj=resp, mode="r|gz") as tar:
        for m in tar:
            if not m.isfile():
                continue
            c = CALL.match(m.name)
            r = None if c else REPORT.match(m.name)
            if not (c or r):
                continue
            raw = tar.extractfile(m).read()
            if r:
                with open(os.path.join(OUT, f"bai_report_{name}_{r['run']}.json"), "wb") as f:
                    f.write(raw)
                continue
            try:
                d = json.loads(raw)
            except ValueError:
                bad += 1; continue
            u = (d.get("response") or {}).get("usage") or {}
            pd_ = u.get("prompt_tokens_details") or {}
            cd = u.get("completion_tokens_details") or {}
            out.write(json.dumps({
                "run": c["run"], "inst": c["inst"], "ts": float(c["ts"]),
                "p": u.get("prompt_tokens", 0) or 0, "o": u.get("completion_tokens", 0) or 0,
                "cached": pd_.get("cached_tokens", 0) or 0,
                "cw": u.get("cache_creation_input_tokens", 0) or 0,
                "cr": u.get("cache_read_input_tokens", 0) or 0,
                "reason": cd.get("reasoning_tokens", 0) or 0,
                "cost": d.get("cost", 0.0) or 0.0,
            }) + "\n")
            n += 1
            if n % 20000 == 0:
                print(f"{name}: {n} calls, {time.time()-t0:.0f}s", flush=True)
    open(dst + ".done", "w").write(json.dumps({"calls": n, "unparseable": bad, "secs": round(time.time() - t0)}))
    print(f"{name}: DONE {n} calls, {bad} unparseable, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
