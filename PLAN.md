# Agent cost: a reproduction of Bai et al. and a lemons test

Registered 2026-10-14, before any token, cost or resolve value from the trajectories below
was computed or seen. Owner Karl Sundström; analysis Hermes. Destination agentrecourse.org
Journal (pre-registered research). Everything below is frozen except through the change log.

## Why

Bai et al., "How Do AI Agents Spend Your Money?" (arXiv 2604.22750v3, 2026) is the main
empirical paper on agent token cost. They released the raw trajectories. Part A checks
whether their headline numbers come out of their own data with independently written code
(a *reproduction*: same data, new code). Part B asks the question their paper and the
literature sweep of 2026-10-14 leave open: how much of an agent task's cost a buyer can
know in advance that a fixed-price seller cannot, and whether that alone breaks fixed-price
selling (Akerlof 1970, applied to agent cost).

## Data

- Bai et al. trajectories: HF `loong0814/openhands_trajectories` @
  `fa9cbb063f770df596da95af24f7af3b8f595778` (no license on the card; tables of aggregates
  only are published; raw rows stay local). 9 full archives (the `_mini` archives are
  ignored). Per LLM call we keep only `usage` fields and `cost` (`extract_bai.py`).
  Resolved labels from each run's `report.json`.
- Task metadata: `princeton-nlp/SWE-bench_Verified` @
  `c104f840cc67f8b6eec6f759ebc8b2693d585d4a` (500 tasks; difficulty 194 / 261 / 42 / 3).
- Part B also uses the three pinned cost-predictor sets (SWE-smith, SWE-rebench-OH,
  SWE-agent; see ~/projects/cost-predictor-spike/agentlogs/PLAN.md), cost = estimated
  tokens (chars / 4, never converted to dollars).

Prior contact, disclosed: one `gpt-5_4runs_mini` call file was opened during the survey to
read its schema; one call's `cost` and `usage` printed (0.0118 USD, 17,431 prompt, 946
completion tokens). No distribution computed. The paper's figures were read (they are the
claims under test). The SWE-bench difficulty counts above were printed. The three
cost-predictor sets were used in rounds 1-8 and in the R study; their between-task variance
share was never computed.

## Units

- Run = (model archive, run_k, instance). Total tokens = sum over calls of prompt +
  completion tokens (the paper's definition, including cached input). Cost = sum of the
  per-call `cost` field.
- Model = archive. The paper lists 8 models; the release has 9 archives, two of them Sonnet
  4.5 (`claude-sonnet-4.5_4runs`, `claude-sonnet-4.5_new_v2_4runs`). Rule, from counts only:
  an archive is *complete* if it has >= 450 instances with all 4 runs. If both Sonnet 4.5
  archives are complete, every Sonnet 4.5 claim is computed on both and counts as
  reproduced only if it holds on both. Configs differ by archive (e.g. maxiter 100 for
  GPT-5, 500 for GPT-5.2); reported, not adjusted.
- Runs with zero calls are dropped and counted.

## Part A: reproduction (each claim scored REPRODUCED / NOT REPRODUCED)

Claims are quoted from v3. Tolerances are set now because the paper gives no CIs.

- **A1 "the most expensive runs double ... the least expensive"** (same agent, same task).
  Statistic: median over (model, task) of max / min total tokens across its 4 runs.
  REPRODUCED if the pooled median is in [1.5, 2.5].
- **A2 "runs on the same task can differ by up to 30x in total tokens".** Statistic: max
  over (model, task) of the same ratio. REPRODUCED if in [20, 45]. Also reported: share of
  (model, task) cells above 10x, to show whether "up to 30x" is typical or an outlier.
- **A3 Kendall tau_b = 0.32 between human difficulty and total tokens**, difficulty merged to
  3 levels (>4 h into 1-4 h). Statistic: tau_b of difficulty vs per-(model, task) mean total
  tokens over 4 runs, pooled over models. REPRODUCED if within 0.05 of 0.32. Per-model tau_b
  reported.
- **A4 overlap: 6.7% of "<15 min" tasks use more total tokens than the average ">1 hour"
  task, 11.1% of ">1 hour" tasks use fewer than the average "<15 min" task.** Computed on
  the same per-(model, task) means, pooled. REPRODUCED if both within 3 percentage points.
- **A5 "higher token usage does not translate into higher accuracy".** Within each task the
  4 runs are ranked by total tokens (MinCost ... MaxCost; ties by run number). Statistic per
  model: resolve rate of MaxCost minus MinCost. REPRODUCED if negative in >= 6 of the
  complete models (sign test) and the pooled difference's 95% bootstrap CI over tasks
  (2,000 resamples, seed 20261014) is below 0.
- **A6 "Kimi-K2 and Claude-Sonnet-4.5, on average, consume over 1.5 million more tokens
  than GPT-5".** Mean total tokens per run over tasks present in all three. REPRODUCED if
  both differences are > 1.5M.

The paper's result counts as reproduced overall if at least 5 of 6 claims are. Any claim
that cannot be computed (e.g. missing archive) is reported NOT EVALUABLE, not as a fail.

## Part B: the lemons test (the gap)

A buyer who has run a task before knows something about its cost; a fixed-price seller
pricing from a public description does not.

- **B1 private share.** Within each group (Bai: model; own sets: dataset x group as in the
  R study), the between-task share of variance of log cost, estimated by one-way random
  effects ANOVA (ICC(1)), using tasks with >= 2 runs. Reported with a 90% bootstrap CI
  over tasks (500 resamples). This is the share of cost variation that is about the task,
  not luck.
- **B2 public share.** The part of the between-task variance a seller can see: R^2 of
  per-task mean log cost on public features (repository, problem-statement length decile,
  difficulty where present), fit on 70% of tasks, scored on 30% (split by instance_id,
  seed 20261014). Private share = B1 minus (B1 x B2 test R^2).
- **B3 the market.** Per group, a seller offers one fixed price p per task (public features
  ignored: the flat case). A buyer knows a noisy estimate of their task's cost: the cost of
  ONE earlier run of the same task (k = 1, a repeat customer; for Bai the earliest run by
  run number, the 3 others are the possible current runs). The buyer takes the fixed price
  if p <= (1 + lambda) x estimate, lambda in {0, 0.1, 0.3} (willingness to pay for
  insurance). The seller must break even on those who take it: equilibrium p* = the
  highest p with p = mean actual cost of takers, found by iterating from p = (1 + lambda)
  x pooled mean cost until it stops moving (max 200 steps). Reported: uptake share of tasks
  at p*, p* / pooled mean, and the same with the oracle buyer (estimate = mean of the other
  runs).
- **Hypotheses, scored on Bai complete models and on the own sets separately:**
  - **H-B1:** median ICC across groups >= 0.5 (most cost variation is knowable per task).
  - **H-B2:** at lambda = 0.1, k = 1, median uptake at p* < 50% (adverse selection alone
    removes most of the market). **Collapse** is reported where uptake < 20%.
  - Expectation written now: H-B1 PASS (Bai's own figure shows large between-task spread);
    H-B2 uncertain, the 2x within-task noise blurs what a buyer knows from one run.

Part B is descriptive-plus-simulation; it does not model sellers who price on public
features (that is the R study's I1, which gained nothing) or buyers who game a seller over
time.

## Run rule

`analyze.py` is committed before its single run. Smoke = the same code on synthetic data
only. If the run crashes after printing results, only reporting code may change; rerun with
the same seeds and disclose. Results append below; nothing above changes except via the
change log.

## Change log

- 2026-10-14: registered.

## Publishing

1. Journal article on agentrecourse.org with PLAN.md, code and aggregate tables linked.
2. Open an issue on `LongjuBai/agent_token_consumption_analysis` with the Part A table
   before publishing, so the authors can answer, and quote their answer if any.
3. Optional: a ReScience C submission (peer-reviewed reproductions, open review on GitHub)
   if Part A finds a substantive discrepancy; Zenodo DOI for the code either way.
