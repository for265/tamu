# Architecture: Short Deck PLO5 bot (`agent2.py`)

Goal: beat the opposing bot in a duplicate match of two 1000-hand halves, under the judge's rules. This document records what the research says, what our game's constraints allow, what `agent2.py` does today, and what I recommend building next.

**How to read the sources.** Claims marked *(searched)* come from the web search results I got while writing this. They are mostly secondary summaries, and I did not read the full papers. Claims marked *(background)* are from my own prior knowledge and were not re-verified here. Where a search came back empty, I say so.

---

## 1. The game, as `game.py` defines it

| Fact | Value | Why it matters |
|---|---|---|
| Deck / hand | 36 cards, 5 hole cards, exactly 2 hole + 3 board must be used | Equity is expensive: 100 five-card lookups per hand evaluation |
| Stacks | 80 chips = 40 big blinds, reset every hand | Short, shallow game. Action tree is small |
| Actions | fold, check/call, raise half-pot, raise pot (max 4 raises per street) | Tiny discrete action space. Ideal for tabular CFR |
| Time | 8 s thinking time per half (about 8 ms per decision at 1000 hands), 1 ms free per request | The hard constraint. Online search of any size is impossible |
| Compute | about 1 CPU, no network, no GPU, numpy and CPU PyTorch available | Anything heavy must be done offline |
| Shipping | repo files must each be under 100 MB, image adds at most 3 GB, server must start in 15 s | We may ship a precomputed strategy table or a small network |
| Memory | wiped every half, each half is a fresh process | Opponent learning has 1000 hands to pay off, then is lost |
| Format | duplicate: both halves use the same deals with seats swapped | Card luck largely cancels, so skill edges show up in the result |
| Verdict | mean bb per deal with a 95% interval, chop if the interval straddles 0 | We need a statistically clear win, not just a positive mean |

**The central insight:** the time bank forbids online computation but the rules allow arbitrarily heavy offline work. The best architecture moves almost all the work offline and keeps the in-match decision to a table or network lookup (microseconds), leaving the time bank for cheap opponent adaptation.

---

## 2. What the research says

### 2.1 Equilibrium solvers: CFR and its descendants
- DeepStack, Libratus and Pluribus all build on counterfactual regret minimization (CFR) *(searched)*. DeepStack re-solves at every decision with a neural network estimating values at the depth limit. Libratus uses an offline blueprint plus nested subgame solving. Pluribus used Monte Carlo CFR, abstraction and depth-limited search *(searched)*. Basic CFR is too costly on full no-limit games, which is why these systems add abstraction, subgame solving or learned values *(searched)*.
- Deep CFR (Brown, Lerer, Gross, Sandholm, ICML 2019) replaces hand abstraction with neural networks that approximate CFR and is proven to converge to an ε-Nash equilibrium in two-player zero-sum games. A reviewer-style summary reports it matching a 30,000-bucket abstraction on heads-up limit hold'em with 1-2 orders of magnitude fewer samples *(searched; that figure is from a summary site, not the paper)*.
- ReBeL combines self-play RL and search, converging to Nash in two-player zero-sum games with much less poker-specific knowledge *(searched)*.
- Monte Carlo CFR with external sampling (Lanctot et al., 2009) traverses only part of the tree per iteration *(searched, via the Deep CFR paper's description)*. CFR+ and Linear CFR are faster-converging variants *(searched: CFR+ keeps regrets non-negative; Pluribus used Linear CFR)*.
- My search for Supremus / DeepStack improvements returned nothing, so I make no claim about them.

**What transfers to us.** The heavy machinery (DeepStack and ReBeL style online search, subgame solving) exists to cope with huge no-limit trees under a time budget. Our game has 4 actions, 4 raises per street, 40 bb stacks. A fully offline blueprint is realistic here and online search is not. That points at tabular or neural CFR trained offline, not at DeepStack/ReBeL.

### 2.2 Safe opponent exploitation
- Ganzfried and Sandholm (2011) build an opponent model from observed action frequencies combined with a precomputed equilibrium strategy and show it can exploit after few interactions *(searched)*. Their 2015 "Safe Opponent Exploitation" trades exploitation against a safety allowance in repeated zero-sum play, but relies on a whole-game LP that scales poorly *(searched)*.
- Safe Exploitation Search (Liu et al., NeurIPS 2022) and OX-Search (Ge et al., ICML 2024) adapt online and stay no more exploitable than the blueprint *(searched)*.
- Two 2026 preprints on confidence-scheduled restricted responses and on exploitation under showdown censoring appeared in the results. They are very recent and likely not peer reviewed, and I did not read them. I treat their titles as pointers only, not evidence.
- I found nothing on exploitation under a tight per-decision time bank *(searched, empty)*.

**What transfers to us.** The principle is the one that matters: *start from an equilibrium (unexploitable) strategy and deviate toward the opponent's measured weaknesses only as far as the evidence supports.* With only 1000 hands per half and sparse showdown data, deviations must be small and confidence-weighted.

### 2.3 PLO specifically
- I found no peer-reviewed PLO-specific CFR or neural-network bot paper *(searched)*. Available PLO material is mostly vendor pages, so treat it as unverified.
- Those pages make a consistent technical point: PLO equity cannot reuse a Hold'em best-five-of-seven evaluator because exactly two hole cards must be used, and practical bots split work between offline solved range tables and per-decision Monte Carlo against an inferred range *(searched, vendor claims)*. That matches our `omaha_score`.
- Bucketing (grouping similar hands by strategic value) is the standard abstraction technique in the poker AI literature *(searched, via a survey summary)*.

**What transfers to us.** There is no off-the-shelf PLO5 solution to copy. We must build our own abstraction, and the engine's `Game` class is already a complete simulator for it.

---

## 3. Where `agent2.py` is today

Current design: per decision, Monte Carlo equity (8 to about 100 samples), bluff probability from showdown history, call or raise against pot odds, adaptive time budget (about 4.8 s of the 8 s bank per half).

Local results (1000 hands, one-sided, in-process, not the real judge): +1.5 to +2.8 bb per hand against four simple fixed policies (random, calling station, always half-pot, always pot). These policies are weak. This result says the bot is not broken. It does not say it beats a competent opponent.

Weaknesses I know about:
1. **Noisy equity.** About 8 to 100 samples gives a standard error of roughly 5 to 15 points of equity, which is large relative to the 3-point margins in the decision rules.
2. **Biased bluff model.** The bluff stat only counts showdowns. Bluffs that we fold to never reach showdown, so the model systematically underestimates bluffing. This is the showdown-censoring problem *(searched, as a topic)*.
3. **Heuristic thresholds** (0.62, 0.78, 0.85, ...) were set by hand, not fit or tested.
4. **Exploitable by design.** A fixed equity-versus-pot-odds policy has predictable frequencies. A good opponent can exploit it, and nothing guards against that.
5. **Weak testing.** No test against a strong opponent, no confidence intervals, no duplicate evaluation.

---

## 4. Options

| Option | Idea | Pros | Cons |
|---|---|---|---|
| A. Keep refining the heuristic | Better thresholds, more samples via faster equity | Cheap, no new infrastructure | Capped quality, exploitable, always time-starved |
| B. Offline blueprint, table lookup (**recommended core**) | Bucket hands by equity, solve the abstract game with external-sampling MCCFR / Linear CFR offline, ship the strategy table | Near-unexploitable, microsecond decisions, fits the 8 ms bank, game tree is small | Needs a good hand abstraction, build time, unverifiable until tested |
| C. Deep CFR / learned policy network | Train value or policy networks with self-play; ship weights; run a numpy or Torch forward pass | No hand-made abstraction, generalizes across states | Training instability, much more compute, harder to debug, more risk for the gain |
| D. DeepStack/ReBeL-style online search | Re-solve at each decision with learned leaf values | State of the art in no-limit | Impossible within 8 ms per decision on one CPU |
| E. Pure exploitation / opponent-model best response | Fit a model, best-respond | Big gains vs weak, static bots | Exploitable by adaptive bots, little data per half |

---

## 5. Recommendation

**Build B, add a small and safe version of E on top, and keep today's heuristic as the fallback.**

### 5.1 Layers

```
                 +--------------------------------------------+
 obs ----------> | 1. State encoder (fast)                    |
                 |    hand bucket (equity class) + board +    |
                 |    street + pot/stack + action history     |
                 +----------------------+---------------------+
                                        v
                 +--------------------------------------------+
                 | 2. Blueprint lookup (offline MCCFR table)  |  <- equilibrium action probabilities
                 +----------------------+---------------------+
                                        v
                 +--------------------------------------------+
                 | 3. Safe exploit adjustment                 |  <- opponent stats (this half only)
                 |    shift probabilities by a bounded amount |     bound grows with evidence
                 +----------------------+---------------------+
                                        v
                 +--------------------------------------------+
                 | 4. Sample action from the mixed strategy   |
                 +--------------------------------------------+
   Fallback: if a state is missing from the table or an error occurs,
   use the equity-vs-pot-odds heuristic that is in agent2.py now.
```

### 5.2 Offline pipeline (not shipped, lives outside `agent2.py`)
1. **Simulator:** reuse `game.Game` and `evaluator.py` directly. The engine is the ground truth for rules.
2. **Hand abstraction:** assign each (hole cards, board) to a bucket by equity versus a random hand (and, if time allows, by an equity histogram so draws separate from made hands). Precompute and cache. Preflop with 5 cards has a modest number of strategically distinct classes; suit-isomorphism cuts it further.
3. **Solver:** external-sampling MCCFR with Linear CFR or CFR+ weighting over the abstract game (buckets x betting history). The betting tree is tiny (four actions, at most four raises per street).
4. **Check solution quality:** measure exploitability of the abstract strategy (or at least head-to-head strength vs. earlier checkpoints), and play it in the real engine.
5. **Export:** quantize the table (probabilities as uint8), store as a compact `.npy`/`.npz` file well under 100 MB, load at startup inside the 15 s window.

### 5.3 In-match opponent adaptation (layer 3)
- Track, per street and per facing-bet-size class: fold-to-raise, call, raise frequencies.
- Estimate with Bayesian shrinkage toward the equilibrium frequencies (so little data means almost no change).
- Cap the shift: for example, move at most `min(0.25, n_obs / (n_obs + 200))` of the way from the blueprint probability toward the best response. This gives the "no worse than the blueprint, within a small allowance" property that the safe-exploitation papers formalize *(searched; our cap is a pragmatic stand-in, not their algorithm)*.
- Correct the censoring problem: count folds, not just showdowns, when estimating how often the opponent bets with weak hands. Where a quantity is unobservable, widen the prior.

### 5.4 Time management
Blueprint lookup and the adjustment are microseconds. The remaining bank can pay for a short, bounded equity calculation only where the table has no entry. Keep the existing adaptive budget (cap at about 60% of the 8 s bank) as the safety net.

---

## 6. Evaluation plan (needed before trusting any version)

1. **Duplicate format always.** Use `judge_engine.py` as intended (both halves, same deals, swapped seats) so luck cancels.
2. **Report mean bb per deal with a 95% interval,** matching the judge's verdict rule. A positive mean inside the interval is a chop, not a win.
3. **Opponent ladder:** the four `baseline-bots`, then the current `agent2.py` heuristic, then each blueprint checkpoint. Each new version must beat the previous one with a clear interval.
4. **Exploitability check:** play the blueprint against a best-response or exploiter bot of our own to see how far it can be pushed.
5. **Time check:** log time used per half and the slowest single decision. It must stay under 8 s total and under 1 s per action.
6. **Robustness:** run the real server through `tools/check_build.sh` (needs Docker) and `pytest`. I have not run either yet because the sandbox here lacks `uvicorn` and `fastapi`.

---

## 7. Plan and risks

| Step | Output | Risk |
|---|---|---|
| 1. Evaluation harness with duplicate play and confidence intervals | Trustworthy numbers | Low |
| 2. Fix the heuristic's known issues (faster equity, uncensored bluff stats) | Better fallback and baseline | Low |
| 3. Hand abstraction + MCCFR on a simplified tree | First blueprint | Medium: abstraction quality drives everything |
| 4. Full-tree blueprint, export, lookup in `agent2.py` | Table-driven bot | Medium: table size and load time |
| 5. Safe adaptation layer | Gains versus exploitable opponents | Medium: easy to overfit 1000 hands |

Honest unknowns:
- I could not find published PLO5 or short-deck PLO results, so I cannot say how large the gain from a blueprint will be here. The literature supports the approach in general, not our exact game.
- Abstraction quality is the main failure point. A coarse bucketing can make the "equilibrium" strategy exploitable in the real game.
- The opponent is unknown. If it is a fixed-policy bot, step 5 (exploitation) matters more than the blueprint. If it is a strong solver bot, the blueprint matters more. Building both covers both cases.

---

## 8. Sources

- [AI in Human-computer Gaming: Techniques, Challenges and Opportunities (survey)](https://arxiv.org/pdf/2111.07631)
- [ReBeL: Combining Deep Reinforcement Learning and Search for Imperfect-Information Games](https://arxiv.org/pdf/2007.13544)
- [Deep Counterfactual Regret Minimization](https://arxiv.org/pdf/1811.00164)
- [OpenHoldem: A Benchmark for Large-Scale Imperfect-Information Game Research](https://arxiv.org/pdf/2012.06168)
- [Counterfactual regret minimization for poker AI (int8.io)](https://int8.io/counterfactual-regret-minimization-for-poker-ai/)
- [Safe opponent exploitation and subgame refinement (NeurIPS 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/file/b12a1d1014e952e676f5d6931d03241a-Paper-Conference.pdf)
- [OX-Search / Subgame refinement for exploitation (ICML 2024)](https://personal.ntu.edu.sg/boan/papers/ICML24_Subgame.pdf)
- [Safe Opponent Exploitation, Ganzfried and Sandholm (arXiv 1603.03491)](https://arxiv.org/pdf/1603.03491v5)
- [Equity Calculation in Pot-Limit Omaha (vendor page, unverified)](https://plopokerbot-com.pages.dev/equity-calculation/)
- [A Survey on Game Theory Optimal Poker (summary)](https://www.themoonlight.io/en/review/a-survey-on-game-theory-optimal-poker)
- 2026 preprints on safe exploitation and showdown censoring: [Agents That Certify Their Own Exploits](https://arxiv.org/pdf/2607.28520), [Safe Observation Capacity under Showdown Censoring](https://arxiv.org/pdf/2608.09954). Not read, not peer reviewed.
