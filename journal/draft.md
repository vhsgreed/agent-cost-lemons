# Can you sell an AI agent's work at a fixed price? Pooling handles luck, not what the buyer knows

**Version:** 0.1 (draft), 2026-10-15.
**Author:** Karl Sundström (Agent Recourse), corresponding and accountable for all claims.
**Contributions:** Study design, analysis code, statistics and drafting by an AI agent (Claude Opus, running in Hermes Agent) under the author's direction. The author posed the research questions (whether fixed-price agent work can be sold, and whether the gap is worth a study or a reproduction), rejected the first framing as textbook, edited the text and approved every claim. Errors are the author's responsibility to correct.

**TL;DR:** A seller who charges a fixed price per agent task, instead of per token, is writing an insurance policy. Two pre-registered studies, one on 40,421 agent runs and one on about 1.05 million model calls, show what that policy costs. Luck is cheap to insure: on a batch of 100 tasks a 9 to 31% markup breaks even 95% of the time, and reading the prompt does not lower it. What the buyer knows is not cheap. About half of the variation in a task's cost belongs to the task itself, a seller cannot see it from the task description, and a buyer who has run the task once can pick fixed price only when it is a bargain. That alone pushes the break-even fixed price to 1.4 to 1.9 times average cost and removes most buyers. Along the way we reproduced the main cost paper in this area from its released data: five of its six headline figures come out, one ("up to 30x") does not, and we have asked the authors.

---

## Abstract

Agentic coding tasks vary widely in cost, so a provider who sells them at a fixed price takes on risk. We measure that risk in two pre-registered studies. Study 1 (three public software-engineering trajectory sets, 40,421 test runs) asks what markup a fixed-price seller needs to break even with 95% probability. One task needs 100 to 275%; a batch of 100 needs 9 to 31% (registered bar of 30% met in 2 of 3 datasets); 1,000 tasks need 3 to 10%. Pricing by prompt length does not reduce the markup in any dataset (0 of 3). Study 2 first reproduces Bai et al. (arXiv 2604.22750) from their released trajectories (8 models x 500 SWE-bench Verified tasks x 4 runs, about 1.05 million calls) with independent code. Under our registered rules 2 of 6 claims reproduce; reading the unit of analysis and the wording from the paper's body, 5 of 6 do, and "up to 30x" is not reproduced (we find 59x to 1,380x depending on filtering). Study 2 then tests adverse selection. The share of log-cost variance between tasks is 0.50 (median over 8 models) and 0.62 (14 groups in our own sets); public task features explain at most 18% of it out of sample on the Bai data. In a simulated market where buyers know the cost of one earlier run of their task, the break-even fixed price rises to 1.94x (Bai) and 1.40x (own sets) average cost and uptake falls to 12% and 27%; with runs shuffled across tasks the price stays at 1.00x. With perfect knowledge of their task the market unravels (about 1% uptake). Pooling handles the luck in agent cost; it does not handle what buyers know.

## 1. The question

Most agent work is sold per token: the buyer pays for what the model reads and writes, and carries the risk that a task runs long. Some vendors now sell per outcome instead, a fixed price per resolved ticket or completed task. Dholakia describes the vendor in that arrangement as one who "is, in effect, writing an insurance policy for its customer", and argues that vendors "must screen customers, customize the price, verify outcomes, control costs".[10] Anthropic's 2025 weekly limits on Claude Code subscriptions, a flat-price product, were announced as "affecting less than 5% of users", and named "running Claude 24/7 in the background" among the usage patterns behind them.[11] A small share of heavy users choosing the flat price is the classic shape of adverse selection.[4]

The question is how large that risk is for agent tasks, and which part of it a fixed-price seller can manage.

Our earlier article on agent cost found that cost is hard to predict before a run starts.[12] This piece asks the commercial follow-up: if a seller cannot predict cost, can they still charge a fixed price? An insurer does not need to predict each claim, only the average over many. That works for luck. It fails when the buyer knows more about their own risk than the insurer does.[4]

## 2. Related work

Work on LLM pricing so far protects the buyer from the provider. Pay-per-token pricing "creates a financial incentive for providers to strategize and misreport" token counts;[5] an auditor that "sequentially queries a provider" can detect it;[6] providers "may covertly substitute cheaper alternatives" for the model the user pays for.[7] On mechanism design, inference auctions let users "bid for faster service",[8] and a model of tiered hardware gives "theoretical grounding for flat per-token API pricing".[9]

The closest empirical work is Bai et al., who ran eight models four times each on SWE-bench Verified and released the trajectories.[1][2] They report that the most expensive run of a task typically doubles the cheapest, that runs "can differ by up to 30" times in total tokens, and that human difficulty ratings only weakly track cost.[1] We use their data for both a reproduction and the lemons test.

We found no study that measures the opposite asymmetry: how much a buyer knows about the cost of their own task that a seller pricing from the task description does not.

## 3. Study 1: the markup for luck

**Registration.** Plan, code and pass bars were committed before the test data was read (2026-10-13). Data: three public trajectory sets from our cost-prediction work (SWE-smith, SWE-rebench-OpenHands, SWE-agent), with cost as estimated tokens processed and a 70/30 split by task.[12] A seller prices each task either flat (the group's mean cost, I0) or by prompt-length decile (I1). The statistic is m95(N): the markup over forecast cost at which a batch of N test tasks breaks even with 95% probability.

| Dataset | Test runs | Flat m95(100) | Prompt-aware m95(100) | Ratio | Single task m95(1) |
|---|---|---|---|---|---|
| SWE-smith | 7,256 | 0.201 [0.139, 0.239] | 0.193 [0.142, 0.228] | 0.96 [0.82, 1.12] | 2.43 |
| SWE-rebench-OH | 9,900 | 0.093 [0.068, 0.118] | 0.091 [0.065, 0.112] | 0.98 [0.86, 1.10] | 0.99 |
| SWE-agent | 23,265 | 0.304 [0.241, 0.359] | 0.312 [0.245, 0.369] | 1.02 [0.89, 1.16] | 2.75 |

90% confidence intervals over test tasks.

- **H1, reading the prompt cuts the markup: FAIL, 0 of 3.** Prompt length and cost correlate at 0.03 to 0.09 (Spearman).
- **H2, a 100-task batch needs at most 30%: PASS, 2 of 3.** SWE-agent needs 31% (upper bound 37%).
- At 1,000 tasks the markup is 3 to 10%.

This is the law of large numbers and nothing more. We report it because the size matters for anyone pricing agent work, not because it is new. It also assumes buyers know nothing about their tasks, so these are the lowest markups a seller could charge.

## 4. Study 2a: reproducing Bai et al.

**How a reproduction works.** A reproduction uses the authors' data and new code; a replication uses new data. We registered six of the paper's headline claims, a statistic for each and a tolerance (the paper gives no confidence intervals), before computing any value from the trajectories, and set "reproduced overall" at 5 of 6.[13] Data: the released archives at revision fa9cbb06,[2] task metadata from SWE-bench Verified.[14] One release archive (a second Sonnet 4.5 run set) has a fourth run for only 177 tasks, so under the registered completeness rule we used the other.

| Claim | Paper | Ours (registered) | Verdict | Ours (exploratory) |
|---|---|---|---|---|
| Most expensive run about doubles the cheapest | ~2x | 2.16x (median) | REPRODUCED | |
| Runs differ by up to 30x | 30x | 1,380x | NOT REPRODUCED | 59x excluding runs of 10 or fewer calls |
| Difficulty vs tokens, Kendall tau_b | 0.32 | 0.21 | NOT REPRODUCED | 0.318 with one value per task |
| Overlap of easy and hard tasks | 6.7% / 11.1% | 12.1% / 31.2% | NOT REPRODUCED | 7.8% / 11.4% with one value per task |
| More tokens, no more accuracy | | MaxCost +3.8 points over MinCost | NOT REPRODUCED | rises then flat: 58.5 / 63.5 / 63.3 / 62.3% |
| Kimi-K2 and Sonnet 4.5 use 1.5M+ more tokens than GPT-5 | >1.5M | +2.25M / +3.16M | REPRODUCED | |

**Registered result: 2 of 6, not reproduced.** We report it as registered. Three of the four misses are ours, not the paper's:

- *Difficulty and overlap.* The paper does not say whether each point is a (model, task) pair or a task. We guessed pairs. Computed with one value per task (the mean over the eight models, 493 tasks), all three figures fall within our tolerances.
- *Accuracy.* We encoded the abstract's sentence, "higher token usage does not translate into higher accuracy".[1] The body says accuracy "increases modestly from MinCost to LowerCost, but then saturates in higher-cost settings",[1] and the data shows exactly that. The abstract states the finding more strongly than the body and the data.

The fourth miss is real. The largest ratio between two runs of one task is 1,380x: on one matplotlib task Gemini 3 Pro used 8,624 tokens in a run that stopped after one call, and 11.9 million in a run of 212 calls. Runs that stop after one or two calls drive the extremes. Removing every case whose cheapest run has 10 or fewer calls still leaves 59x. "Up to 30x" therefore needs a filter the paper does not describe. We asked the authors on 2026-10-14;[3] [REPLY PENDING: quote their answer here, or state that none came by publication].

The two claims that matter for pricing, the roughly 2x spread between identical runs and the large differences between models, reproduce cleanly.

## 5. Study 2b: what the buyer knows

**Registration.** Same registration as 2a, committed before any value was seen.[13] One hypothesis was replaced before the run: a synthetic smoke test showed that "uptake below 50%" would pass on pure noise, so the bar became a price bar, checked against a shuffled null. The change is in the public change log.

**Method, in three steps.**
1. *How much of the cost is about the task?* For each model (Bai) or group (own sets), the share of the variance of log cost that lies between tasks rather than between runs of the same task (intraclass correlation, ICC).
2. *How much of that can a seller see?* How well public features (repository, difficulty rating, length of the task description) predict a task's mean cost on held-out tasks (R^2).
3. *What happens to a fixed price?* A seller offers one price p per task. A buyer knows the cost of one earlier run of the same task (a repeat customer, k = 1) and takes the fixed price if p is at most 1.1 times that estimate (willing to pay 10% for insurance). The seller must break even on the buyers who take it. We find the price where that holds, p*, and report p* over average cost and the share of tasks that take it. Control: the same market with runs shuffled across tasks, so the buyer's estimate carries no task information.

Bai cost is the per-call dollar cost field, except for Qwen3-Coder (cost missing for more than 5% of runs, tokens used, per the registered rule). Own sets use estimated tokens.

**Results.**

| | Groups | Between-task share (median ICC) | Seller-visible (R^2) | Break-even price / mean cost | Shuffled control | Uptake (shuffled) | Groups below 20% uptake |
|---|---|---|---|---|---|---|---|
| Bai, 8 models | 8 | 0.50 (0.32 to 0.66) | -0.02 to 0.18 | 1.94 (1.34 to 2.19) | 0.97 to 1.09 | 12% (40%) | 7 of 8 |
| Own sets | 14 | 0.62 (0.25 to 0.73) | -0.13 to 0.39 | 1.40 (1.20 to 2.90) | 0.78 to 1.05 | 27% (45%) | 4 of 14 |

- **H-B1, at least half of cost variation is between tasks: PASS** on both. On Bai it passes by 0.001, so the honest reading is "about half", not "most".
- **H-B2, the break-even fixed price is at least 1.25x average cost: PASS** on both, and the shuffled control stays near 1.00x, so the rise comes from task knowledge, not from noise in the buyer's estimate.
- **With perfect knowledge** (the buyer's estimate is the mean of the other runs) uptake falls to 0.2 to 1.4% on Bai at 2.5 to 7.2x average cost. The market for fixed-price tasks unravels in the way Akerlof described for used cars.[4]
- A higher willingness to pay for insurance helps a little: at 30% (lambda 0.3), Bai uptake rises to 15 to 48% and the price falls to 1.21 to 1.91x.

## 6. What this means

**For sellers.** Fixed-price agent work is viable against luck and fragile against knowledge. Volume makes the luck cheap (Study 1). It does nothing about a buyer who has run the task before, and repeat buyers are exactly the customers a seller wants. The levers are the ones insurers use and the ones Dholakia lists:[10] price from the buyer's history rather than from the task description, cap the exposure (fixed price up to a token ceiling, per-token above it), or sell per token and leave the risk with the buyer. Per-token pricing, which most of the market uses, is the stable outcome this data predicts.

**For buyers.** If you rerun similar tasks, you know which ones are expensive, and a flat price is worth most to you exactly on those. Expect flat-price offers to come with limits, as Claude Code's did.[11]

**For the cost-prediction literature.** Bai et al.'s central findings hold on their own data. The one figure that does not is the most quoted one. "Up to 30x" depends on how runs that stop after one call are treated, and the abstract's accuracy sentence overstates the body. Neither changes the paper's conclusions.

## 7. Limits

- Our market has one seller, no competition and no learning over time. Real sellers can screen buyers; real buyers can game a seller over many rounds.
- The buyer's information is one earlier run of the same task. Real buyers rerun similar tasks, not identical ones, so they may know less (or, from their own codebase, more).
- Benchmark tasks are not commercial tasks. The three own sets measure cost as estimated tokens, not dollars.
- Study 1 and Study 2 use different cost measures and different data; they agree in direction, not in magnitude.
- The reproduction tolerances were set by us, since the paper gives no intervals. Our exploratory re-analyses were done after seeing the registered result and are labelled as such.

## 8. Data and code

Registration, change log, analysis code and aggregate results: github.com/vhsgreed/agent-cost-lemons.[13] Raw trajectories are not redistributed; the extraction scripts fetch them from the authors' Hugging Face release at the pinned revision.[2] Study 1's registration and code: [PENDING: publish ~/projects/fixed-price-inference alongside].

## Sources

[1] https://arxiv.org/abs/2604.22750: Bai et al. 2026, How Do AI Agents Spend Your Money?
[2] https://huggingface.co/datasets/loong0814/openhands_trajectories: Bai et al. released trajectories (HF)
[3] https://github.com/LongjuBai/agent_token_consumption_analysis/issues/2: Our issue to the authors
[4] https://doi.org/10.2307/1879431: Akerlof 1970, The Market for Lemons
[5] https://arxiv.org/abs/2505.21627: Velasco et al., Is Your LLM Overcharging You?
[6] https://arxiv.org/abs/2510.05181: Velasco et al., Auditing Pay-Per-Token in LLMs
[7] https://arxiv.org/abs/2504.04715: Cai et al., Auditing Model Substitution in LLM APIs
[8] https://arxiv.org/abs/2609.40070: Harris et al., Inference Auctions
[9] https://arxiv.org/abs/2609.40098: McDougall & Sankaralingam, Pricing Time, Not Just Tokens
[10] https://thepricingconundrum.substack.com/p/outcome-based-pricing-as-insurance: Dholakia, Outcome-Based Pricing as Insurance
[11] https://news.ycombinator.com/item?id=44713757: Claude Code weekly rate limits (Anthropic announcement, HN)
[12] https://agentrecourse.org/journal/agent-run-cost: Sundström 2026, What an AI coding-agent run will cost (v1.5)
[13] https://github.com/vhsgreed/agent-cost-lemons: This study: code, registration, aggregates
[14] https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified: SWE-bench Verified
