# agent-cost-lemons

Pre-registered reproduction of Bai et al., "How Do AI Agents Spend Your Money?"
(arXiv 2604.22750v3) from their released trajectories, plus a test of buyer-side adverse
selection ("lemons") in fixed-price pricing of agent tasks.

- `PLAN.md`: the registration, change log and results (registered verdicts and
  exploratory checks, labelled separately).
- `extract_bai.py`, `extract_bai_v2.py`, `fetch_all.sh`: stream the trajectories from
  Hugging Face (`loong0814/openhands_trajectories` @ `fa9cbb06`) and keep only per-call
  usage and cost. Raw data is not redistributed here.
- `analyze.py`: the single registered run (`--smoke` = synthetic data only).
- `diag.py`: exploratory checks made after the run.
- `results.json`, `results.out`: aggregate outputs.

Part B also uses three SWE trajectory sets prepared in a separate project; those caches
are not included.

Code: MIT. Analysis: Karl Sundström with Hermes Agent. Contact via issues.
