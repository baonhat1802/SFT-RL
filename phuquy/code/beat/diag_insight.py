"""Interventional causal discovery of the KEY INSIGHT in y* (no training).

Reading survey (27/11): on all-fail questions the model lacks ONE structural insight that Luna states in its first 1-2 steps
(g^2 | 180; U + 2V = 3x^2; alternating => fixed positions), then executes fine in its own way. Teacher-forced log-probs did not
find those steps (an insight is plausible when READ). Here each step of y* is an intervention do(context += step_i), and its
effect is measured on what we care about: the student's own success rate.

Hard questions (<= 2/8 correct). Arms, 8 samples each, plain model, step given as a hint after the question:
  plain                      8 fresh samples (baseline)
  step_k   (k < max_steps)   "Hint: {step_k of y*}"                       single-step intervention
  abs_k                      same step with derived numbers masked (_)    is the effect the IDEA or the VALUES?
  prefix_k                   "Hint: {steps 0..k}"                         cumulative (is one step enough?)
  full                       all of y* as hint                            upper bound
Leak: a step containing \\boxed or the final answer string is flagged; reported with and without.
Honest selection: the best step per question is chosen on samples 0-3 and scored on samples 4-7.

    PYTHONPATH=code python -m beat.diag_insight --questions 384 --out runs/diag_insight
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=384)
    ap.add_argument("--max-hard", type=int, default=160)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--num-gen", type=int, default=8)
    ap.add_argument("--max-completion", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=59)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import numpy as np
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
    MATH = re.compile(r"\\\((.*?)\\\)|\\\[(.*?)\\\]|\$\$(.*?)\$\$|\$(.*?)\$", re.S)

    def steps_of(ref):
        spans = [(m.start(), m.end()) for m in MATH.finditer(ref)]
        inside = lambda c: any(s < c < e for s, e in spans)
        cuts = [m.end() for m in re.finditer(r"(?<=[.;:])\s+|\n+", ref) if not inside(m.start())]
        b = [0] + [c for c in cuts if 0 < c < len(ref)] + [len(ref)]
        st = [ref[u:v].strip() for u, v in zip(b, b[1:])]
        st = [s for s in st if len(s) >= 8]
        # merge a display-math line into the sentence before it (a formula alone is not a "step")
        merged = []
        for s in st:
            if merged and (s.startswith("\\[") or s.startswith("$$")):
                merged[-1] = merged[-1] + " " + s
            else:
                merged.append(s)
        return merged

    def abstract(step, question):
        given = set(re.findall(r"\d+", question))
        return re.sub(r"\d+", lambda m: m.group() if m.group() in given else "_", step)

    rows = []
    for r in load_track("curated").itertuples():
        msgs = [{"role": m["role"], "content": m["content"]} for m in r.prompt]
        user = msgs[1]["content"]
        rows.append({"sys": msgs[0], "user": user, "body": user[:-len(SUFFIX)] if user.endswith(SUFFIX) else user,
                     "q": r.extra_info["question"], "ref": r.target[0]["content"], "ans": str(r.extra_info["answer"]).strip(),
                     "gt": r.reward_model["ground_truth"]})
    Q = [rows[i] for i in np.random.permutation(len(rows))[:a.questions].tolist()]
    chat = lambda x, user: tok(tok.apply_chat_template([x["sys"], {"role": "user", "content": user}], tokenize=False,
                                                       add_generation_prompt=True), add_special_tokens=False)["input_ids"]
    hint = lambda x, h: chat(x, x["body"] + "\n\nHint: " + h + SUFFIX)

    llm = LLM(model=BASE_MODEL, gpu_memory_utilization=0.85, max_model_len=4096 + a.max_completion, seed=a.seed,
              enable_prefix_caching=True)
    sp = SamplingParams(temperature=GRPO["temperature"], top_p=1.0, max_tokens=a.max_completion, stop_token_ids=stop_ids,
                        n=a.num_gen)

    def run(items):
        """items: (x, arm, prompt_ids) -> fills x['res'][arm] = list of 0/1."""
        outs = llm.generate([{"prompt_token_ids": p} for _, _, p in items], sp, use_tqdm=False)
        texts = [c.text for o in outs for c in o.outputs]
        rr = score_all(texts, [x["gt"] for (x, _, _), o in zip(items, outs) for _ in o.outputs], 8)
        k = 0
        for (x, arm, _), o in zip(items, outs):
            n = len(o.outputs); x.setdefault("res", {})[arm] = [float(v) for v in rr[k:k + n]]; k += n

    run([(x, "stage1", chat(x, x["user"])) for x in Q])
    H = [x for x in Q if sum(x["res"]["stage1"]) <= 2][:a.max_hard]
    print(f"STAGE1 {time.time() - t0:.0f}s hard {len(H)}/{len(Q)}", flush=True)

    items = []
    for x in H:
        x["steps"] = steps_of(x["ref"])[:a.max_steps]
        x["leak"] = [("boxed" in s) or (len(x["ans"]) >= 2 and x["ans"] in s) for s in x["steps"]]
        items.append((x, "plain", chat(x, x["user"])))
        items.append((x, "full", hint(x, x["ref"])))
        for k, s in enumerate(x["steps"]):
            items.append((x, f"step_{k}", hint(x, s)))
            items.append((x, f"abs_{k}", hint(x, abstract(s, x["q"]))))
            if k > 0:
                items.append((x, f"prefix_{k}", hint(x, " ".join(x["steps"][:k + 1]))))
    run(items)
    print(f"STAGE2 {time.time() - t0:.0f}s items {len(items)}", flush=True)

    rng = np.random.default_rng(a.seed)
    boot = lambda v: [float(np.percentile([np.mean(rng.choice(v, len(v))) for _ in range(2000)], q)) for q in (5, 95)] if len(v) else None
    acc = lambda x, arm: float(np.mean(x["res"][arm]))
    res = {"questions": len(Q), "hard": len(H), "allfail": int(sum(sum(x["res"]["stage1"]) == 0 for x in H)),
           "plain": float(np.mean([acc(x, "plain") for x in H])), "full": float(np.mean([acc(x, "full") for x in H])),
           "mean_steps": float(np.mean([len(x["steps"]) for x in H])), "by_position": {}}
    for k in range(a.max_steps):
        X = [x for x in H if len(x["steps"]) > k]
        Xn = [x for x in X if not x["leak"][k]]
        d = [acc(x, f"step_{k}") - acc(x, "plain") for x in Xn]
        da = [acc(x, f"abs_{k}") - acc(x, "plain") for x in Xn]
        dp = [acc(x, f"prefix_{k}") - acc(x, "plain") for x in X if k > 0]
        res["by_position"][k] = {"n": len(X), "n_noleak": len(Xn), "leak_rate": float(np.mean([x["leak"][k] for x in X])) if X else None,
                                 "delta_step": float(np.mean(d)) if d else None, "ci90_step": boot(d),
                                 "delta_abs": float(np.mean(da)) if da else None,
                                 "delta_prefix": float(np.mean(dp)) if dp else None}
    # per question: best non-leaking single step, chosen on samples 0-3, scored on 4-7
    best_pos, honest, honest_plain, maxeff, examples = [], [], [], [], []
    for x in H:
        cand = [k for k in range(len(x["steps"])) if not x["leak"][k]]
        if not cand:
            continue
        fit = {k: np.mean(x["res"][f"step_{k}"][:4]) for k in cand}
        kb = max(cand, key=lambda k: (fit[k], -k))
        best_pos.append(kb); honest.append(np.mean(x["res"][f"step_{kb}"][4:])); honest_plain.append(np.mean(x["res"]["plain"][4:]))
        eff = {k: acc(x, f"step_{k}") - acc(x, "plain") for k in cand}
        maxeff.append(max(eff.values()))
        if eff[kb] >= 0.375:
            examples.append({"q": x["q"][:300], "ans": x["ans"], "pos": kb, "step": x["steps"][kb], "abs": abstract(x["steps"][kb], x["q"]),
                             "eff": eff[kb], "eff_abs": acc(x, f"abs_{kb}") - acc(x, "plain"), "steps": x["steps"]})
    res["best_step_position_hist"] = {str(k): int(sum(1 for b in best_pos if b == k)) for k in range(a.max_steps)}
    res["honest_best_step"] = float(np.mean(honest)) if honest else None
    res["honest_plain"] = float(np.mean(honest_plain)) if honest_plain else None
    res["share_q_with_key_step_ge_0.25"] = float(np.mean([m >= 0.25 for m in maxeff])) if maxeff else None
    res["share_q_with_key_step_ge_0.5"] = float(np.mean([m >= 0.5 for m in maxeff])) if maxeff else None
    res["examples"] = sorted(examples, key=lambda e: -e["eff"])[:25]
    res["time"] = round(time.time() - t0, 1)
    (out / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("RESULT " + json.dumps({k: v for k, v in res.items() if k != "examples"}), flush=True)
    print(f"DONE {out}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
