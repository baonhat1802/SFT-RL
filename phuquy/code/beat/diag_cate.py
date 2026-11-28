"""Pre-screen for R04-3 (no training): can self-applied interventions on the context unlock knowledge the base model already
has, and is the best intervention predictable per question (heterogeneous treatment effects, CATE)?

Stage 1: Q random questions x 8 plain samples -> hard set = questions with <= 2/8 correct.
Stage 2 (hard set, randomised arms, 8 samples each): plain (more samples: the key control, ~24% of all-fail groups are
rescued by 8 more samples), restate, recall (formulas/theorems), special case, backward (answer form), ystar_first (first
sentence of y* as a hint; leak-checked).
Honest policy values: each arm's 8 samples are split 4 + 4; a policy picks an arm on the first 4 and is scored on the other 4.

    PYTHONPATH=code python -m beat.diag_cate --questions 384 --out runs/diag_cate
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
ARMS = {
    "plain": None,
    "restate": "Before solving, restate the problem precisely in your own words: list the given quantities and conditions, "
               "and state exactly what is asked.",
    "recall": "Before solving, list the definitions, formulas, and theorems that are relevant to this problem, and say how "
              "each one could be used.",
    "special": "Before solving, work out a small special case or a simpler version of the problem to see the pattern, "
               "then solve the general problem.",
    "backward": "Before solving, describe what form the final answer must take, then work backwards from that form to "
                "decide what you need to compute.",
    "ystar_first": "Hint: a good way to start is: {first}",
}


def first_move(ref):
    t = ref.strip()
    m = re.search(r"(?<=[.;])\s|\n", t[20:])
    s = t[:20 + m.start()] if m else t[:200]
    return s.strip()[:240]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=384)
    ap.add_argument("--num-gen", type=int, default=8)
    ap.add_argument("--hard-max", type=int, default=2, help="hard = at most this many correct of num-gen plain samples")
    ap.add_argument("--max-completion", type=int, default=2048)
    ap.add_argument("--vllm-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import numpy as np
    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from train.common import BASE_MODEL, GRPO, configure_tokenizer
    from train.data import load_track
    from train.models import eos_ids
    from train.rewards import score_all

    t0 = time.time()
    random.seed(a.seed); np.random.seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tok = configure_tokenizer(AutoTokenizer.from_pretrained(BASE_MODEL), "curated")
    stop_ids = sorted(set(eos_ids(tok, "curated")) | {tok.convert_tokens_to_ids("<|im_end|>")})
    meta = pd.read_parquet(Path(__file__).resolve().parents[1] / "data/bridge_curated_teachers/sft/train.parquet",
                           columns=["id", "subject", "level"]).set_index("id")

    rows = []
    for r in load_track("curated").itertuples():
        msgs = [{"role": m["role"], "content": m["content"]} for m in r.prompt]
        user = msgs[1]["content"]
        body = user[:-len(SUFFIX)] if user.endswith(SUFFIX) else user
        sid = r.extra_info.get("source_id")
        rows.append({"sys": msgs[0], "body": body, "ref": r.target[0]["content"], "ans": str(r.extra_info["answer"]),
                     "gt": r.reward_model["ground_truth"],
                     "subject": meta.loc[sid, "subject"] if sid in meta.index else "?",
                     "level": int(meta.loc[sid, "level"]) if sid in meta.index else -1})
    order = np.random.permutation(len(rows))[:a.questions].tolist()
    Q = [rows[i] for i in order]

    def prompt(x, arm):
        ins = ARMS[arm]
        if arm == "ystar_first":
            ins = ins.format(first=x["first"])
        content = x["body"] + ("\n\n" + ins if ins else "") + SUFFIX
        return tok(tok.apply_chat_template([x["sys"], {"role": "user", "content": content}], tokenize=False,
                                           add_generation_prompt=True), add_special_tokens=False)["input_ids"]

    llm = LLM(model=BASE_MODEL, gpu_memory_utilization=a.vllm_mem, max_model_len=2048 + a.max_completion, seed=a.seed,
              enable_prefix_caching=True)
    sp = SamplingParams(temperature=GRPO["temperature"], top_p=1.0, max_tokens=a.max_completion, stop_token_ids=stop_ids,
                        n=a.num_gen)

    def sample(items):
        """items: (question dict, arm). Returns list of (rewards list, lengths list, truncated list)."""
        outs = llm.generate([{"prompt_token_ids": prompt(x, arm)} for x, arm in items], sp, use_tqdm=False)
        texts = [c.text for o in outs for c in o.outputs]
        rew = score_all(texts, [x["gt"] for (x, _), o in zip(items, outs) for _ in o.outputs], 8)
        res, k = [], 0
        for o in outs:
            n = len(o.outputs)
            res.append((list(map(float, rew[k:k + n])), [len(c.token_ids) for c in o.outputs],
                        [c.finish_reason == "length" for c in o.outputs]))
            k += n
        return res

    s1 = sample([(x, "plain") for x in Q])
    for x, (r, _, _) in zip(Q, s1):
        x["stage1"] = sum(r)
    hard = [x for x in Q if x["stage1"] <= a.hard_max]
    print(f"STAGE1 {time.time() - t0:.0f}s reward {np.mean([np.mean(r) for r, _, _ in s1]):.3f} hard {len(hard)}/{len(Q)}", flush=True)

    leak = 0
    for x in hard:
        x["first"] = first_move(x["ref"])
        a_ = x["ans"].strip()
        x["leak"] = ("boxed" in x["first"]) or (len(a_) >= 2 and a_ in x["first"])
        leak += x["leak"]
    items = [(x, arm) for x in hard for arm in ARMS if not (arm == "ystar_first" and x["leak"])]
    s2 = sample(items)
    for (x, arm), (r, L, tr) in zip(items, s2):
        x.setdefault("arms", {})[arm] = {"r": r, "len": L, "trunc": tr}
    print(f"STAGE2 {time.time() - t0:.0f}s items {len(items)} ystar leaks dropped {leak}", flush=True)

    res = {"questions": len(Q), "hard": len(hard), "allfail": sum(1 for x in hard if x["stage1"] == 0),
           "ystar_leak_dropped": leak, "arms": {}}
    rng = np.random.default_rng(a.seed)

    def boot(vals):
        vals = np.array(vals)
        bs = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(2000)]
        return [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))]

    for arm in ARMS:
        X = [x for x in hard if arm in x["arms"]]
        acc = [np.mean(x["arms"][arm]["r"]) for x in X]
        af = [x for x in X if x["stage1"] == 0]
        d = [np.mean(x["arms"][arm]["r"]) - np.mean(x["arms"]["plain"]["r"]) for x in X]
        res["arms"][arm] = {
            "n": len(X), "acc": float(np.mean(acc)),
            "rescue_allfail": float(np.mean([max(x["arms"][arm]["r"]) for x in af])) if af else None,
            "delta_vs_plain": float(np.mean(d)), "delta_ci90": boot(d),
            "len": float(np.mean([np.mean(x["arms"][arm]["len"]) for x in X])),
            "trunc": float(np.mean([np.mean(x["arms"][arm]["trunc"]) for x in X]))}

    # honest policy values: choose on samples 0-3, score on samples 4-7
    common = [x for x in hard if all(arm in x["arms"] for arm in ARMS)]
    fit = lambda x, arm: np.mean(x["arms"][arm]["r"][:4])
    ev = lambda x, arm: np.mean(x["arms"][arm]["r"][4:])
    arms = list(ARMS)
    pv = {"plain": float(np.mean([ev(x, "plain") for x in common]))}
    best_single = max(arms, key=lambda arm: np.mean([fit(x, arm) for x in common]))
    pv["best_single(" + best_single + ")"] = float(np.mean([ev(x, best_single) for x in common]))
    subj_best = {}
    for s in {x["subject"] for x in common}:
        Xs = [x for x in common if x["subject"] == s]
        subj_best[s] = max(arms, key=lambda arm: np.mean([fit(x, arm) for x in Xs]))
    pv["per_subject"] = float(np.mean([ev(x, subj_best[x["subject"]]) for x in common]))
    # per-question: arm with most successes in the first 4 (ties -> plain first, then random)
    def q_choice(x):
        sc = {arm: fit(x, arm) for arm in arms}
        m = max(sc.values())
        cands = [arm for arm in arms if sc[arm] == m]
        return "plain" if "plain" in cands else random.choice(cands)
    pv["per_question"] = float(np.mean([ev(x, q_choice(x)) for x in common]))
    pv["oracle_any_arm_rescue"] = float(np.mean([max(max(x["arms"][arm]["r"]) for arm in arms) for x in common]))
    pv["plain_arm_rescue"] = float(np.mean([max(x["arms"]["plain"]["r"]) for x in common]))   # 8 fresh samples; oracle uses 48
    res["policy_values"] = pv
    res["subject_best_arm"] = subj_best
    res["n_common"] = len(common)
    res["time"] = round(time.time() - t0, 1)
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print("RESULT " + json.dumps(res), flush=True)
    print(f"DONE {out}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
