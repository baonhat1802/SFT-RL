# Capstone — mentor updates (Nguyen Lam Phu Quy, on-policy guidance workstream)

Newest first. Each update states what was run, what was found, and what comes next. Numbers sit in tables; the text gives the reading.

---

## 2026-11-26 — Why Meta-OPSD collapsed, faster training code, one idea rejected before training

**Setting for all results below.** Qwen2.5-3B (base) with LoRA (rank 32), 8,519 curated MATH questions with reference solutions, binary
reward from the team verifier, 2,048-token answer budget, one A100 40GB (team GCP project). These numbers are not comparable with the
team baseline table, which uses Qwen3-0.6B.

### 1. Diagnosis of the Meta-OPSD collapse

Meta-OPSD (previous update) trained the model to benefit from one self-distillation step, where the teacher is the same model conditioned
on the reference solution y*. Both variants collapsed, and the training log showed a negative cosine (−0.27) between the teacher gradient
and the reward gradient. A dedicated diagnostic (no training) measured the first-order transfer of each guidance signal: the cosine between
the guidance gradient on one half of the questions and the GRPO reward gradient on the other half, at the initial model, over three
batches of 256 questions.

| Guidance signal (gradient on questions A) | Cosine with reward gradient on questions B |
|---|---|
| **Reference: GRPO gradient on A** | **+0.15** (range 0.07 to 0.29) |
| Concise teacher on all-incorrect groups (Meta-OPSD setting) | +0.06 |
| Teacher with the full reference prompt | +0.16 (positive in 3 of 3 batches) |
| Teacher on incorrect solutions of mixed groups | +0.17 |
| Teacher on correct solutions | +0.05 |
| Teacher given only the final answer | +0.01 |
| Per-token teacher weights on GRPO (RLSD) | +0.08 (plain GRPO on the same samples: +0.10) |

| Measurement point | After a teacher step | After a GRPO step (control) |
|---|---|---|
| Cosine at θ (correct) | +0.06 | +0.15 |
| Cosine at θ′, as logged during Meta-OPSD | −0.26 | −0.10 |

**Findings.**
- The −0.27 in the Meta-OPSD log is a measurement artefact. Measuring after an inner step lowers the cosine for any step direction,
  including the GRPO gradient itself. At the initial model, the teacher gradient is weakly aligned with the reward, not opposed to it.
- The collapse therefore comes from the update machinery: the teacher step had a fixed norm larger than the GRPO step, the meta-gradient
  was evaluated at a shifted point, and the finite-difference Hessian term was noise-dominated.
- Two design choices weakened the teacher signal: the concise teacher prompt (+0.06 against +0.16 for the full prompt) and applying the
  teacher to all-incorrect groups (+0.06 against +0.17 for incorrect solutions in mixed groups).
- Per-token teacher weighting (RLSD) adds no transfer at the initial model.
- Gradient estimates from one rollout batch are very noisy (the GRPO-to-GRPO cosine itself ranges from 0.07 to 0.29). Methods that steer
  each update by a per-batch gradient estimate inherit this noise.

### 2. Faster training code

Profiling showed that forward and backward passes took 57–70% of each rollout. The new implementation packs samples by length under a
token budget, computes log-probabilities in chunks with activation checkpointing instead of keeping full-vocabulary fp32 logits, and frees
the vLLM memory during training.

| | Old path | New path |
|---|---|---|
| Forward + backward, 64 samples up to 2,048 tokens | 35.1 s, 15.4 GB peak | 17.8 s, 9.3 GB peak |
| Full GRPO rollout (128 questions × 8 samples) | 332 s | 247 s |
| Log-probability difference, one sample at a time | — | ≤ 2e-6 |

Results are numerically equivalent to the old path. A full one-epoch GRPO run drops from about 6.1 to about 4.5 hours.

### 3. New idea rejected by a pre-screen

Following the request for a more novel direction than mixing GRPO with distillation, a literature check found that most variants of a
reference-conditioned teacher already exist (contrastive teachers, memory-conditioned teachers, multi-teacher consensus, sub-question
curricula, RLSD). One direction remained: use y* only to locate where an incorrect solution went wrong, then let the model itself
re-generate from that point, so that y* chooses where to explore but never what to write. A pre-screen tested this before any training,
on 262 questions where all 8 samples were incorrect, with 8 continuations per branch point.

| Branch point | Questions rescued (≥ 1 correct continuation) |
|---|---|
| Resample from the start | 23.7% |
| Random position at the depth the teacher chose | 24.0% |
| Teacher: strongest disagreement | 21.4% |
| Teacher: first disagreement | 18.3% |
| Random position | 15.3% |
| Highest student entropy | 11.1% |

**Findings.** The teacher does not locate the error: no teacher-based branch point beats resampling from the start, and the first
disagreement does worse than a random position at the same depth. Deeper branch points do worse, so for this model a prefix of an incorrect
solution is a burden rather than a foothold. The idea was dropped without a training run. A side result: about one quarter of the
"all-incorrect" questions are solved by eight more samples from scratch, so many zero-gradient groups reflect too few samples rather than
missing knowledge.

### 4. Does a base-model teacher use the reference solution at all?

Six ways of presenting y* to the teacher were compared on the same student samples (two batches of 256 questions): the concise and the
full instruction prompts used so far, the final answer only, y* as the model's own earlier answer in a multi-turn chat, y* in the system
turn, and y* placed at the start of the model's own answer.

| Teacher context | Mean |log p_teacher − log p_student| per token | Tokens changed by > 1 nat | Transfer cosine (reference GRPO: 0.03 / 0.12) |
|---|---|---|---|
| Concise instruction | 0.09 | 1.4% | +0.05 / +0.16 (all-wrong), +0.09 / +0.22 (mixed) |
| Full instruction | 0.08 | 1.2% | +0.03 / +0.20, +0.02 / +0.23 |
| Final answer only | 0.04 | 0.3% | about 0 |
| Earlier answer in a multi-turn chat | 0.08 | 1.1% | mixed sign |
| System turn | 0.07 | 0.9% | negative |
| Start of its own answer | 0.10 | 1.9% | ≤ 0 |

**Findings.** For every format, the base model's teacher distribution barely moves when y* is shown (about 1% of tokens change by more
than one nat). The formats that move the teacher most (y* at the start of its own answer) transfer worst, so a stronger reaction to y* is
not a better training signal. The instruction-style prompts already in use are the best of the six.

### 5. Teacher with y* in the weights instead of the context (rejected by pre-screen, 2026-11-27)

Because the base model barely reacts to y* in its context, a second teacher was built by taking one SFT step on y* for each question and
scoring the student's own solutions with the adapted weights (no y* in the context). Two batches of 256 questions:

| Signal | Tokens changed by > 1 nat | Transfer cosine (reference GRPO: +0.14) |
|---|---|---|
| Plain SFT gradient on y* | — | **−0.18** |
| Weight teacher, step 0.3 / 1 / 3 | 0.7% / 21% / 89% | −0.16 to −0.22 at every step size |
| In-context teacher, full prompt | 1.4% | **+0.15** |
| In-context teacher, concise prompt | 1.4% | about 0 |

**Findings.** Imitating the text of y* points against the reward for this base model, whether through SFT directly or through a teacher
that has absorbed y* into its weights. This confirms, at first order, the phase-1 result that SFT on the reference solutions lowers
accuracy. The only y*-based signal with positive transfer is the in-context teacher with the full prompt, at about the level of GRPO itself.

### 6. Why SFT on the reference solutions opposes the reward (2026-11-27)

The reference solutions are short: a median of 115 tokens, against 356–489 tokens for the model's own correct solutions to the same
questions. The SFT gradient on y* was split by token class and each part's transfer was measured as above (two batches of 256 questions).

| Part of the SFT gradient on y* | Share of tokens | Mean NLL per token under the student | Transfer cosine (reference GRPO: +0.08) |
|---|---|---|---|
| All tokens | 100% | 0.95 | −0.15 |
| Prose (words outside formulas) | 26% | 2.03 | −0.13 |
| Mathematical notation without digits | 46% | 0.74 | −0.12 |
| Numbers inside formulas | 18% | 0.41 | −0.06 |
| Final boxed answer | 9% | 0.36 | about 0 |
| Shortest / longest half of y* | — | — | −0.15 / −0.14 |
| SFT on the student's own correct solutions | — | 0.41 | **+0.07** |

The SFT gradients on y* and on the student's own correct solutions have a cosine of −0.12.

**Findings.** The harm does not come from skipped steps or unexplained numbers: the student predicts the numbers of y* easily, and that part
of the gradient is close to neutral. The harmful part is the writing style of the reference model, in both its prose and its notation,
which the student finds far less likely than its own text. Imitating that style moves the model away from its own way of writing, and the
reward gradient favours the model's own way of writing. Masking token classes does not help, since every class transfers at or below zero.
The result agrees with recent reports of style drift under imitation of reference-conditioned text (DASD, arXiv 2606.00628; arXiv 2603.24472).
Signals with positive transfer all live on the student's own text: GRPO, SFT on its own correct solutions, and the in-context teacher with
the full prompt scoring the student's samples.

### 7. Causal pre-screens: separating content from style (2026-11-27)

The previous result treats the writing style of y* as a confounder: y* differs from the student's own text in both content and style,
and SFT cannot separate the two. Three pre-screens test ways of isolating the content (two batches of 256 questions, reference GRPO +0.08).

| Signal | Transfer cosine (batch 1 / batch 2) |
|---|---|
| **Counterfactual pair: log p(y*) − log p(y* with one or two derived values changed)** | **+0.10 / +0.11** |
| Counterfactual pair: one or two + / − signs flipped | +0.08 / +0.02 |
| Control: lines of y* shuffled | 0.00 / −0.08 |
| Plain SFT on y* (same batches) | +0.03 / +0.07 |
| Invariant update: coordinates where the y* and own-solution gradients agree in sign | +0.04 / +0.15 (unmasked average: +0.05 / +0.14) |
| Invariant update with GRPO as a third environment | +0.05 / +0.13 |
| In-context teacher with y* / with the reference of another question / difference | +0.06 / +0.05 / +0.02 (means) |

**Findings.**
- The counterfactual pair keeps the style of y* in both texts, so style cancels, and only the preference for the correct value over a
  wrong value in the same context remains. This signal transfers above the GRPO reference in both batches, is close to orthogonal to plain
  SFT on y* (cosine −0.05), and the shuffled-lines control does not transfer. It passes the pre-screen.
- The sign-agreement mask adds nothing over simply averaging the two gradients, so the invariance idea is dropped.
- A teacher given the reference solution of a different question produces almost the same signal as the true teacher. Most of the
  in-context teacher's signal therefore comes from the prompt format rather than from the content of y*.
- Correction to section 6: the transfer of plain SFT on y* varies strongly between batches (from −0.23 to +0.07 over six batches, mean
  −0.09, negative in four of six). The token-class split in section 6 came from two consistent batches, but single-batch values should not be
  read as exact.

### 8. Training run of the counterfactual pair (stopped, 2026-11-27)

The counterfactual-pair term (λ = 0.1, one pair per question) was added to GRPO with everything else identical to the GRPO baseline run
(same data order, seed, LoRA, 2,048-token budget). The run was stopped at rollout 30 of 66 because the training reward fell clearly below
the baseline.

| Rollouts | Training reward (pairs / GRPO) | Mean length (tokens) | Truncated | All-incorrect groups | Margin log p(y*) − log p(ỹ*) per token |
|---|---|---|---|---|---|
| 0–9 | 0.359 / 0.358 | 688 / 674 | 4.1% / 3.2% | 0.29 / 0.29 | 0.19 |
| 10–19 | 0.518 / 0.507 | 735 / 712 | 6.0% / 2.1% | 0.21 / 0.21 | 0.38 |
| 20–24 | 0.483 / 0.515 | 549 / 732 | 3.1% / 2.1% | 0.24 / 0.22 | 0.75 |
| 25–29 | 0.403 / 0.543 | 572 / 708 | 5.2% / 1.4% | 0.30 / 0.19 | 1.08 |

**Findings.** The positive first-order transfer measured at the initial model held for about twenty rollouts and then reversed. The
margin between the correct and the counterfactual text kept growing without bound, because the term can always be lowered further by
pushing the probability of the counterfactual text down, and the policy degraded as the margin grew. A bounded form of the term (logistic or
hinge on the margin) would be needed for any further attempt; a first-order pre-screen at the initial model does not detect this failure.

### 9. Causal pre-screens of a new round (2026-11-27)

**Doubly robust use of the teacher (rejected).** The y*-teacher log-ratio was used as an action-dependent control variate for the GRPO
gradient, with an exact full-vocabulary correction, so that the expected update equals the GRPO gradient and any bias of the teacher cannot
enter. Both sanity checks held (the control variate has no alignment across halves or with the reward gradient), but its correlation with
the GRPO noise was about ±0.05 and changed sign between batches, which caps the variance reduction below 0.5%.

**Why (forward-pass diagnosis, 171 mixed groups).**

| Teacher | AUC correct vs incorrect (mean log-ratio) | AUC, answer region only | Token-level correlation with the advantage |
|---|---|---|---|
| Full reference prompt | 0.66 | 0.60 | 0.007 |
| Concise reference prompt | 0.67 | 0.63 | 0.006 |
| Placebo: reference of another question | 0.64 | 0.47 | 0.004 |

The teacher separates correct from incorrect solutions, but a placebo teacher does almost as well: most of the separation comes from the
prompt frame, which penalises degenerate text (the largest log-ratios fall on garbage tokens, language switches and loops), not from the
content of y*. The part specific to y* sits in the final-answer region, where the teacher knows the answer, which the reward already
provides. At the level of single tokens the log-ratio is uncorrelated with the advantage, so a token-level control variate cannot help.

**Self-chosen interventions on the context (rejected), and the effect of one sentence of y*.** On 224 hard questions (at most 2 of 8
correct), generic self-prompts (restate the problem, list relevant theorems, solve a special case, work backwards) did not beat eight more
plain samples (−0.01 to −0.02), and choosing a prompt per question did worse than one fixed choice. Giving only the first sentence of y*
as a hint raised accuracy from 0.143 to 0.207 (+0.065, 90% interval +0.045 to +0.085) and solved 41% of all-incorrect questions against
25%. The bottleneck on hard questions is the choice of the first move, and y* carries that information; a teacher that reads all of y*
changes only about 1% of the tokens of the student's own solutions.

**Opening swap: is the benefit of y* carried by the opening of the solution?** On 214 hard questions, every arm is continued by the
model without y* in context. An opening of about 64 tokens written by the model with y* in context, then continued without y*, raised
accuracy from 0.124 to 0.150 (+0.025, 90% interval +0.006 to +0.043). The complete solution written with y* in context reached 0.685, so
the opening carries about 4.5% of the total effect. Openings of the informed model mostly restate the problem, and the decisive step comes
later. The first sentence of y* helped as a hint (+0.040) but hurt when inserted as the start of the answer (−0.055): outputs became half
as long and adopted the terse style of the reference. The hypothesis that y* acts through a sparse mechanism at the start of the solution
is rejected at this granularity; the effect is spread along the solution.

**Where along the solution does y* act? (mediation curve).** On 247 hard questions, solutions written with y* in context were cut after
16 to 256 tokens or 25/50/75% of their length and continued by the model without y*. The transferred share of the total effect (0.130 →
0.655) rises steadily: 2% at 16 tokens, 6% at 64, 18% at half the solution, 31% at three quarters. There is no localised key step.
Informed solutions are not self-sufficient: given 75% of a correct informed solution, the uninformed model finishes correctly 40% of the
time, against 69% for 75% of its own correct solutions. Informed text contains steps justified only by the hidden reference, so imitating
it means learning from data confounded by a variable the student never observes. Forcing a prefix of a plain solution has no effect.

**Causal skeleton of the reference solutions (rejected by pre-screen).** Each step of y* was deleted in turn and the drop in log-probability
of the computed values in later steps and of the final answer was measured, giving a step graph; the skeleton is the set of ancestors of
the answer. The graph separates computation from prose (prose is 2.5% of skeleton tokens against 54% elsewhere). Over 6 batches, SFT on
the skeleton without the answer has cosine −0.065 with the GRPO gradient, no better than randomly chosen steps (−0.049); the full reference
gives −0.077 and the answer region alone +0.026. The computational steps of the reference, not only its style, oppose the reward
gradient, consistent with the mediation result above.

### Next steps
1. Brainstorm a different family of mechanisms, outside "a teacher conditioned on y* supervises the student", each with a cheap
   pre-screen of this kind before any training run.
2. Log the cosine at θ (not at θ′) during training, so that guidance quality can be tracked over a run.
3. Tune generation throughput (`max_num_seqs`), the remaining unoptimised share of each rollout (about 35%).
