# Speaker script: Combining SFT and RL

One block per slide. The script adds context to each slide and does not repeat the slide text.

**1. Cover.** Post-training for mathematical reasoning relies on two signals: supervision from reference solutions (SFT)
and reward on solutions sampled from the model (RL). The proposal asks how a single training run can retain the strengths
of both signals.

**2. Two sources of training signal.** SFT maximises the likelihood of every token in a reference solution. GRPO samples
8 solutions per question, assigns each solution a binary reward, and increases the probability of solutions whose reward
exceeds the group mean.

**3. Complementary strengths.** SFT provides a dense signal, a signal on unsolved questions, and access to knowledge that
the model does not yet have. RL trains on the model's own distribution and causes less forgetting [1]. According to [3],
RL mainly improves accuracy on problems that the base model already solves. The shaded cells mark the strengths that a
combined method should retain.

**4. SFT alone lowers accuracy.** In the team's setting, SFT on reference solutions lowers average accuracy from 14.4 to
8.1. The drop is consistent with [1] and [2], which report larger forgetting and weaker generalisation after SFT.

**5. No RL signal on unsolved questions.** When all 8 solutions are incorrect, every advantage is zero and the question
contributes no gradient. These unsolved questions are the cases in which a reference solution would be most useful.

**6. Common combination schedules.** Sequential training, a fixed mixture and an adaptive mixture differ only in the
weight assigned to SFT at each step. In all three schedules the SFT target remains the reference text.

**7. Sequential training underperforms.** GRPO initialised from the SFT model scores 4.7 points below GRPO initialised
from the base model. The loss from the SFT stage therefore persists through the RL stage.

**8. Notation.** The reference solution y* defines the guidance. The verifier defines the reward r.

**9. A general objective.** The first term is the expected reward on sampled solutions. The second term measures, token by
token, the divergence between the model and a target derived from y*. A combination method is specified by five choices:
the source of guided tokens (μ), the target (q), the divergence (D), and the guidance weights per step and per token
(λ, w).

**10. Existing methods as special cases.** The schedules on slide 6 and the gradient-norm balancing from the earlier plan
all fix μ and q to the reference text and vary only λ. On-policy distillation applies guidance to model samples instead,
so training stays on the model's distribution. OPSD uses the model itself, conditioned on y*, as the teacher. D is
cross-entropy for the rows that use y* directly and usually reverse KL for OPD and OPSD.

**11. Explored directions (updated 25/11/2026).** Two directions were tested on Qwen2.5-3B. First, SFT on correct
solutions that the model writes itself: accuracy rises from 16.8 to 19.6, while SFT on the reference text lowers accuracy
to 12.7. Second, Meta-OPSD: the model conditioned on the reference solution acts as a teacher on its own samples, and a
meta-learning objective keeps only guidance that raises reward on other questions. Both Meta-OPSD variants collapse, and
the teacher gradient has negative cosine similarity with the reward gradient at every logged step. The next runs let the
reward set the direction of each update and use the teacher only to scale the update per token.

**12. Workstreams.** Three workstreams: SFT/RL combination, RL optimisation, and on-policy guidance (slide 11). Integration, frozen evaluation and the final report are shared. All methods are compared at an equal
GPU-hour budget, which also separates algorithmic gains from gains due to a faster RL engine.

**A1–A2.** Accuracy of the untrained model across answer budgets, and per-benchmark scores behind slides 4, 7 and A1.
