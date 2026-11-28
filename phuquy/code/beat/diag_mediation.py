"""Mediation curve (no training): WHERE along a y*-informed solution does the benefit of y* enter?

Hard questions (<= 2/8 correct over 8 plain samples). For each hard question, 8 solutions are sampled by the model WITH y* in
context (HINT_TEMPLATE) and 8 by the plain model. Each solution is cut after k tokens (k = 16, 32, 64, 128, 256) or after a fraction
f of its own length (25/50/75%); the plain model (no y* in context) writes the rest, with the total length capped at max-completion.
  hint@cut   prefix from the informed solution    -> how much of the total effect of y* has already entered by this point
  plain@cut  prefix from a plain solution          -> control for "a forced prefix, then a fresh continuation"
Mediated share at a cut = (hint@cut - plain) / (hinted_full - plain), plain = 8 fresh plain samples.
A sharp rise between two cuts marks a localised key step; a steady rise means the effect is spread along the solution.
Prefixes that already contain the final answer (or \\boxed) are flagged as leaks and reported separately.

    PYTHONPATH=code python -m beat.diag_mediation --questions 384 --out runs/diag_mediation
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

CUTS = ["t16", "t32", "t64", "t128", "t256", "f25", "f50", "f75"]
PLAIN_CUTS = ["t64", "f25", "f50", "f75"]


def cut_len(cut, n):
    k = int(cut[1:]) if cut[0] == "t" else int(round(n * int(cut[1:]) / 100))
    return max(1, min(k, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=384)
    ap.add_argument("--num-gen", type=int, default=8)
    ap.add_argument("--hard-max", type=int, default=2)
    ap.add_argument("--max-completion", type=int, default=2048)
    ap.add_argument("--vllm-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import numpy as np
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from beat.loop import HINT_TEMPLATE
    from train.common import BASE_MODEL, GRPO, configure_tokenizer
    from train.data import load_track
    from train.models import eos_ids
    from train.rewards import score_all

    t0 = time.time()
    random.seed(a.seed); np.random.seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tok = configure_tokenizer(AutoTokenizer.from_pretrained(BASE_MODEL), "curated")
    stop_ids = sorted(set(eos_ids(tok, "curated")) | {tok.convert_tokens_to_ids("<|im_end|>")})

    rows = []
    for r in load_track("curated").itertuples():
        msgs = [{"role": m["role"], "content": m["content"]} for m in r.prompt]
        rows.append({"sys": msgs[0], "user": msgs[1]["content"], "q": r.extra_info["question"], "ref": r.target[0]["content"],
                     "ans": str(r.extra_info["answer"]).strip(), "gt": r.reward_model["ground_truth"]})
    Q = [rows[i] for i in np.random.permutation(len(rows))[:a.questions].tolist()]

    chat = lambda x, user: tok.apply_chat_template([x["sys"], {"role": "user", "content": user}], tokenize=False,
                                                   add_generation_prompt=True)
    ids = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    for x in Q:
        x["pp"] = ids(chat(x, x["user"]))
        x["hp"] = ids(chat(x, HINT_TEMPLATE.format(question=x["q"], reference=x["ref"])))

    llm = LLM(model=BASE_MODEL, gpu_memory_utilization=a.vllm_mem, max_model_len=4096 + a.max_completion, seed=a.seed,
              enable_prefix_caching=True)
    sp = lambda n, m=a.max_completion: SamplingParams(temperature=GRPO["temperature"], top_p=1.0, max_tokens=m,
                                                      stop_token_ids=stop_ids, n=n)

    def gen(prompt_ids, params):
        return llm.generate([{"prompt_token_ids": p} for p in prompt_ids], params, use_tqdm=False)

    # ---- stage 1: hard set
    o1 = gen([x["pp"] for x in Q], sp(a.num_gen))
    r1 = score_all([c.text for o in o1 for c in o.outputs], [x["gt"] for x, o in zip(Q, o1) for _ in o.outputs], 8)
    k = 0
    for x, o in zip(Q, o1):
        x["s1"] = sum(r1[k:k + len(o.outputs)]); k += len(o.outputs)
    H = [x for x in Q if x["s1"] <= a.hard_max]
    print(f"STAGE1 {time.time() - t0:.0f}s hard {len(H)}/{len(Q)} allfail {sum(x['s1'] == 0 for x in H)}", flush=True)

    # ---- stage 2: fresh plain samples (the baseline) and informed samples (the sources of the prefixes)
    op = gen([x["pp"] for x in H], sp(a.num_gen))
    oh = gen([x["hp"] for x in H], sp(a.num_gen))
    rp = score_all([c.text for o in op for c in o.outputs], [x["gt"] for x, o in zip(H, op) for _ in o.outputs], 8)
    rh = score_all([c.text for o in oh for c in o.outputs], [x["gt"] for x, o in zip(H, oh) for _ in o.outputs], 8)
    k = 0
    for x, o_p, o_h in zip(H, op, oh):
        n = len(o_p.outputs)
        x["plain"] = [float(v) for v in rp[k:k + n]]; x["hfull"] = [float(v) for v in rh[k:k + n]]; k += n
        x["src_plain"] = [list(c.token_ids) for c in o_p.outputs]
        x["src_hint"] = [list(c.token_ids) for c in o_h.outputs]
    print(f"STAGE2 {time.time() - t0:.0f}s plain {np.mean([np.mean(x['plain']) for x in H]):.3f} "
          f"hinted_full {np.mean([np.mean(x['hfull']) for x in H]):.3f}", flush=True)

    # ---- stage 3: continuations from prefixes
    jobs = []        # (x, arm, cut, src index, prefix tokens)
    for x in H:
        for src, arm, cuts in (("src_hint", "hint", CUTS), ("src_plain", "plain", PLAIN_CUTS)):
            for j, toks in enumerate(x[src]):
                for cut in cuts:
                    jobs.append((x, arm, cut, j, toks[:cut_len(cut, len(toks))]))
    outs = llm.generate([{"prompt_token_ids": x["pp"] + pre} for x, _, _, _, pre in jobs],
                        [sp(1, max(1, a.max_completion - len(pre))) for _, _, _, _, pre in jobs], use_tqdm=False)
    texts, gts = [], []
    for (x, _, _, _, pre), o in zip(jobs, outs):
        texts.append(tok.decode(pre) + o.outputs[0].text); gts.append(x["gt"])
    rr = score_all(texts, gts, 8)
    leak = lambda s, x: "boxed" in s or (len(x["ans"]) >= 2 and x["ans"] in s)
    res_by = {}
    for (x, arm, cut, j, pre), o, r in zip(jobs, outs, rr):
        ptxt = tok.decode(pre)
        res_by.setdefault((id(x), arm, cut), []).append({"r": float(r), "j": j, "plen": len(pre), "leak": leak(ptxt, x),
                                                         "trunc": o.outputs[0].finish_reason == "length",
                                                         "mentions_ref": "reference" in ptxt.lower()})
    print(f"STAGE3 {time.time() - t0:.0f}s jobs {len(jobs)}", flush=True)

    # ---- summary
    rng = np.random.default_rng(a.seed)
    boot = lambda v: [float(np.percentile([np.mean(rng.choice(v, len(v))) for _ in range(2000)], q)) for q in (5, 95)]
    plain = np.array([np.mean(x["plain"]) for x in H]); hfull = np.array([np.mean(x["hfull"]) for x in H])
    tot = float(hfull.mean() - plain.mean())
    res = {"questions": len(Q), "hard": len(H), "allfail": int(sum(x["s1"] == 0 for x in H)),
           "plain": float(plain.mean()), "hinted_full": float(hfull.mean()), "total_effect": tot,
           "hint_len_tokens": float(np.mean([len(t) for x in H for t in x["src_hint"]])), "cuts": {}}
    for arm, cuts in (("hint", CUTS), ("plain", PLAIN_CUTS)):
        for cut in cuts:
            R = [res_by[(id(x), arm, cut)] for x in H]
            acc = np.array([np.mean([e["r"] for e in L]) for L in R])
            d = acc - plain
            flat = [(e, x) for L, x in zip(R, H) for e in L]
            nl = [e["r"] for e, _ in flat if not e["leak"]]
            # agreement with the source: P(correct | source correct), P(correct | source wrong)
            src_r = lambda e, x: (x["hfull"] if arm == "hint" else x["plain"])[e["j"]]
            sc = [e["r"] for e, x in flat if src_r(e, x) == 1]; sw = [e["r"] for e, x in flat if src_r(e, x) == 0]
            res["cuts"][f"{arm}@{cut}"] = {
                "acc": float(acc.mean()), "delta_vs_plain": float(d.mean()), "ci90": boot(d),
                "mediated_share": float(d.mean() / tot) if tot else None,
                "prefix_tokens": float(np.mean([e["plen"] for e, _ in flat])),
                "leak_rate": float(np.mean([e["leak"] for e, _ in flat])),
                "acc_no_leak": float(np.mean(nl)) if nl else None,
                "acc_given_source_correct": float(np.mean(sc)) if sc else None,
                "acc_given_source_wrong": float(np.mean(sw)) if sw else None,
                "mentions_reference": float(np.mean([e["mentions_ref"] for e, _ in flat])),
                "trunc": float(np.mean([e["trunc"] for e, _ in flat]))}
    res["time"] = round(time.time() - t0, 1)
    (out / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("RESULT " + json.dumps(res), flush=True)
    print(f"DONE {out}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
