"""Opening-swap pre-screen (mechanism swap, no training): is the benefit of y* carried by the OPENING (plan-choice mechanism),
and does it survive when the opening is written in the student's own voice?

Hard questions (<= 2/8 correct over 8 plain samples). Arms, 8 samples each, all CONTINUED by the plain model (no y* in context):
  plain            8 fresh plain samples                                         (control: more samples)
  luna_hint        first sentence of y* as a hint in the prompt                  (replicates R04-3, +0.065)
  luna_prefix      first sentence of y* as the start of the answer               (do(opening := Luna's))
  self_plain       opening = start of a plain sample, then a fresh continuation  (control for "forced opening")
  self_hinted      opening = start of a sample of the model WITH y* in context (HINT_TEMPLATE), then a fresh continuation
                   without y*                                                     (do(opening := own-voice, y*-informed))
  hinted_full      complete solution of the model WITH y* in context              (total effect of y* in context)
Opening of a generated text = up to the last sentence end within its first 64 tokens (64 tokens if none).
Mediated share = (self_hinted - plain) / (hinted_full - plain). Openings containing the final answer are counted (leak).

    PYTHONPATH=code python -m beat.diag_opening --questions 384 --out runs/diag_opening
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


def luna_first(ref):
    t = ref.strip()
    m = re.search(r"(?<=[.;])\s|\n", t[20:])
    return (t[:20 + m.start()] if m else t[:200]).strip()[:240]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=384)
    ap.add_argument("--num-gen", type=int, default=8)
    ap.add_argument("--hard-max", type=int, default=2)
    ap.add_argument("--open-tokens", type=int, default=64)
    ap.add_argument("--max-completion", type=int, default=2048)
    ap.add_argument("--vllm-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=41)
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
        user = msgs[1]["content"]
        body = user[:-len(SUFFIX)] if user.endswith(SUFFIX) else user
        rows.append({"sys": msgs[0], "user": user, "body": body, "q": r.extra_info["question"], "ref": r.target[0]["content"],
                     "ans": str(r.extra_info["answer"]).strip(), "gt": r.reward_model["ground_truth"]})
    Q = [rows[i] for i in np.random.permutation(len(rows))[:a.questions].tolist()]

    chat = lambda x, user: tok.apply_chat_template([x["sys"], {"role": "user", "content": user}], tokenize=False,
                                                   add_generation_prompt=True)
    ids = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    plain_p = lambda x: chat(x, x["user"])
    hint_p = lambda x: chat(x, HINT_TEMPLATE.format(question=x["q"], reference=x["ref"]))
    lhint_p = lambda x: chat(x, x["body"] + "\n\nHint: a good way to start is: " + x["lfirst"] + SUFFIX)

    llm = LLM(model=BASE_MODEL, gpu_memory_utilization=a.vllm_mem, max_model_len=4096 + a.max_completion, seed=a.seed,
              enable_prefix_caching=True)
    full = lambda n: SamplingParams(temperature=GRPO["temperature"], top_p=1.0, max_tokens=a.max_completion,
                                    stop_token_ids=stop_ids, n=n)
    short = SamplingParams(temperature=GRPO["temperature"], top_p=1.0, max_tokens=a.open_tokens, stop_token_ids=stop_ids,
                           n=a.num_gen)

    def gen(prompts, sp):
        return llm.generate([{"prompt_token_ids": ids(p)} for p in prompts], sp, use_tqdm=False)

    def opening(c):
        """Up to the last sentence end within the first open_tokens tokens of a sampled continuation."""
        t = c.text
        ends = [m.end() for m in re.finditer(r"[.;:]\s|\n", t)]
        return t[:ends[-1]] if ends else t

    # ---- stage 1: hard set
    o1 = gen([plain_p(x) for x in Q], full(a.num_gen))
    r1 = score_all([c.text for o in o1 for c in o.outputs], [x["gt"] for x, o in zip(Q, o1) for _ in o.outputs], 8)
    k = 0
    for x, o in zip(Q, o1):
        x["s1"] = sum(r1[k:k + len(o.outputs)]); k += len(o.outputs)
    H = [x for x in Q if x["s1"] <= a.hard_max]
    for x in H:
        x["lfirst"] = luna_first(x["ref"])
    print(f"STAGE1 {time.time() - t0:.0f}s hard {len(H)}/{len(Q)} allfail {sum(x['s1'] == 0 for x in H)}", flush=True)

    # ---- openings: plain-model and y*-informed (own voice)
    op_plain = gen([plain_p(x) for x in H], short)
    op_hint = gen([hint_p(x) for x in H], short)
    for x, op, oh in zip(H, op_plain, op_hint):
        x["op_plain"] = [opening(c) for c in op.outputs]
        x["op_hint"] = [opening(c) for c in oh.outputs]

    # ---- arms (each a list of (prompt text, prefix text) with one sample each, or n samples of one prompt)
    jobs = []        # (question, arm, prompt_text, prefix_text, n)
    for x in H:
        jobs.append((x, "plain", plain_p(x), "", a.num_gen))
        jobs.append((x, "luna_hint", lhint_p(x), "", a.num_gen))
        jobs.append((x, "luna_prefix", plain_p(x), x["lfirst"] + " ", a.num_gen))
        jobs.append((x, "hinted_full", hint_p(x), "", a.num_gen))
        for o in x["op_plain"]:
            jobs.append((x, "self_plain", plain_p(x), o, 1))
        for o in x["op_hint"]:
            jobs.append((x, "self_hinted", plain_p(x), o, 1))
    res_by = {}
    for n in sorted({j[4] for j in jobs}):
        J = [j for j in jobs if j[4] == n]
        outs = gen([p + pre for _, _, p, pre, _ in J], full(n))
        texts, gts, keys = [], [], []
        for (x, arm, _, pre, _), o in zip(J, outs):
            for c in o.outputs:
                texts.append(pre + c.text); gts.append(x["gt"]); keys.append((id(x), arm, len(c.token_ids), c.finish_reason == "length"))
        rr = score_all(texts, gts, 8)
        for (qid, arm, L, tr), r in zip(keys, rr):
            res_by.setdefault((qid, arm), []).append((float(r), L, tr))
    print(f"ARMS {time.time() - t0:.0f}s jobs {len(jobs)}", flush=True)

    rng = np.random.default_rng(a.seed)
    boot = lambda v: [float(np.percentile([np.mean(rng.choice(v, len(v))) for _ in range(2000)], q)) for q in (5, 95)]
    ARMS = ["plain", "luna_hint", "luna_prefix", "self_plain", "self_hinted", "hinted_full"]
    acc = {arm: [np.mean([r for r, _, _ in res_by[(id(x), arm)]]) for x in H] for arm in ARMS}
    res = {"questions": len(Q), "hard": len(H), "allfail": int(sum(x["s1"] == 0 for x in H)), "arms": {}}
    for arm in ARMS:
        d = np.array(acc[arm]) - np.array(acc["plain"])
        af = [i for i, x in enumerate(H) if x["s1"] == 0]
        res["arms"][arm] = {"acc": float(np.mean(acc[arm])), "delta_vs_plain": float(d.mean()), "ci90": boot(d),
                            "rescue_allfail": float(np.mean([max(r for r, _, _ in res_by[(id(H[i]), arm)]) for i in af])) if af else None,
                            "len": float(np.mean([L for x in H for _, L, _ in res_by[(id(x), arm)]])),
                            "trunc": float(np.mean([t for x in H for _, _, t in res_by[(id(x), arm)]]))}
    tot = res["arms"]["hinted_full"]["acc"] - res["arms"]["plain"]["acc"]
    res["mediated_share_self_hinted"] = (res["arms"]["self_hinted"]["acc"] - res["arms"]["plain"]["acc"]) / tot if tot else None
    res["mediated_share_luna_prefix"] = (res["arms"]["luna_prefix"]["acc"] - res["arms"]["plain"]["acc"]) / tot if tot else None
    leak = lambda o, x: len(x["ans"]) >= 1 and ("boxed" in o or (len(x["ans"]) >= 2 and x["ans"] in o))
    res["opening_leak_rate_hinted"] = float(np.mean([leak(o, x) for x in H for o in x["op_hint"]]))
    res["opening_leak_rate_plain"] = float(np.mean([leak(o, x) for x in H for o in x["op_plain"]]))
    res["opening_len_chars"] = {"hinted": float(np.mean([len(o) for x in H for o in x["op_hint"]])),
                                "plain": float(np.mean([len(o) for x in H for o in x["op_plain"]])),
                                "luna": float(np.mean([len(x["lfirst"]) for x in H]))}
    ex = random.sample(H, min(6, len(H)))
    res["examples"] = [{"luna": x["lfirst"], "hinted": x["op_hint"][0], "plain": x["op_plain"][0]} for x in ex]
    res["time"] = round(time.time() - t0, 1)
    (out / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("RESULT " + json.dumps({k: v for k, v in res.items() if k != "examples"}), flush=True)
    print(f"DONE {out}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
