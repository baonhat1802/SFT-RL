"""Counterfactual twins under privileged interventions + interventional causal discovery (pre-screen, no training).

Framework: privileged information at train time (here: steps of Luna's y*, derived numbers masked) is used ONLY as an intervention
do(context += hint). Twins share the exogenous noise (same per-request sampling seed -> coupled Gumbel/exponential draws in vLLM),
so a twin pair is identical until the point where the hint changes a decision.

Stage 1  plain 8 samples on Q questions -> hard set H (<= 2/8 correct) and mixed groups (3..7/8, disjoint) for the GRPO direction.
Stage 2  for each hard question and each coupled seed s: base (no hint), each single step hint, each pair of steps, all steps.
Discovery (per question, from the coupled twins):
  PS(S)  probability of sufficiency of hint set S = P(success under do(S) | base failed, same noise)   (Pearl)
  minimal sufficient set = smallest S with PS >= ps_min; synergy edge i-j when PS({i,j}) - max(PS(i), PS(j)) >= ps_min
  divergence point = first token where a twin differs from its base; reported as position and as share of the base length
Learning direction (--grad): LoRA gradients at the initial policy, all with the PLAIN prompt as context:
  twin_after  mean log p(success twin) - mean log p(failed base), tokens after the divergence point only
  twin_full   same on whole sequences
  natural     success vs failure of the plain model at DIFFERENT seeds (same count)
  compared by cosine to the GRPO gradient of the mixed groups; anchor = GRPO half A vs half B.

    PYTHONPATH=code python -m beat.diag_twins --out runs/diag_twins
"""
from __future__ import annotations

import argparse
import itertools
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
    ap.add_argument("--max-hard", type=int, default=120)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--max-completion", type=int, default=1536)
    ap.add_argument("--ps-min", type=float, default=0.25)
    ap.add_argument("--grad", type=int, default=1)
    ap.add_argument("--max-pairs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from train.common import BASE_MODEL, GRPO, configure_tokenizer
    from train.data import load_track
    from train.models import eos_ids
    from train.rewards import score_all

    t0 = time.time()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tok = configure_tokenizer(AutoTokenizer.from_pretrained(BASE_MODEL), "curated")
    stop_ids = sorted(set(eos_ids(tok, "curated")) | {tok.convert_tokens_to_ids("<|im_end|>")})
    eos_id = tok.convert_tokens_to_ids(tok.eos_token)
    MATH = re.compile(r"\\\((.*?)\\\)|\\\[(.*?)\\\]|\$\$(.*?)\$\$|\$(.*?)\$", re.S)

    def steps_of(ref):
        spans = [(m.start(), m.end()) for m in MATH.finditer(ref)]
        inside = lambda c: any(s < c < e for s, e in spans)
        cuts = [m.end() for m in re.finditer(r"(?<=[.;:])\s+|\n+", ref) if not inside(m.start())]
        b = [0] + [c for c in cuts if 0 < c < len(ref)] + [len(ref)]
        st = [ref[u:v].strip() for u, v in zip(b, b[1:])]
        merged = []
        for s in st:
            if len(s) < 8:
                continue
            if merged and (s.startswith("\\[") or s.startswith("$$")):
                merged[-1] += " " + s
            else:
                merged.append(s)
        return merged

    def abstract(step, q):
        given = set(re.findall(r"\d+", q))
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
    for x in Q:
        x["pp"] = chat(x, x["user"])

    llm = LLM(model=BASE_MODEL, gpu_memory_utilization=0.5 if a.grad else 0.85, max_model_len=4096, seed=a.seed,
              enable_sleep_mode=bool(a.grad))
    sp = lambda n, seed=None: SamplingParams(temperature=GRPO["temperature"], top_p=1.0, max_tokens=a.max_completion,
                                             stop_token_ids=stop_ids, n=n, seed=seed)

    # ---- stage 1
    o1 = llm.generate([{"prompt_token_ids": x["pp"]} for x in Q], sp(8), use_tqdm=False)
    r1 = score_all([c.text for o in o1 for c in o.outputs], [x["gt"] for x, o in zip(Q, o1) for _ in o.outputs], 8)
    k = 0
    for x, o in zip(Q, o1):
        x["s1"] = [{"ids": list(c.token_ids), "r": float(r1[k + j]), "trunc": c.finish_reason == "length"} for j, c in enumerate(o.outputs)]
        k += len(o.outputs)
    H = [x for x in Q if sum(s["r"] for s in x["s1"]) <= 2][:a.max_hard]
    MIX = [x for x in Q if 3 <= sum(s["r"] for s in x["s1"]) <= 7]
    print(f"STAGE1 {time.time() - t0:.0f}s hard {len(H)} mixed {len(MIX)}", flush=True)

    # ---- stage 2: coupled twins under hint interventions
    jobs = []
    for qi, x in enumerate(H):
        st = [abstract(s, x["q"]) for s in steps_of(x["ref"])]
        st = [s for s in st if "boxed" not in s and not (len(x["ans"]) >= 2 and x["ans"] in s)][:a.max_steps]
        x["hints"] = st
        K = len(st)
        sets = [()] + [(i,) for i in range(K)] + list(itertools.combinations(range(K), 2)) + ([tuple(range(K))] if K > 2 else [])
        x["sets"] = sets
        for S in sets:
            user = x["user"] if not S else x["body"] + "\n\nHint: " + " ".join(st[i] for i in S) + SUFFIX
            pid = chat(x, user)
            for s in range(a.seeds):
                jobs.append((qi, S, s, pid))
    outs = llm.generate([{"prompt_token_ids": p} for *_, p in jobs], [sp(1, 1000 + s) for _, _, s, _ in jobs], use_tqdm=False)
    rr = score_all([o.outputs[0].text for o in outs], [H[qi]["gt"] for qi, *_ in jobs], 8)
    for (qi, S, s, _), o, r in zip(jobs, outs, rr):
        H[qi].setdefault("tw", {})[(S, s)] = {"ids": list(o.outputs[0].token_ids), "r": float(r), "trunc": o.outputs[0].finish_reason == "length"}
    print(f"STAGE2 {time.time() - t0:.0f}s twins {len(jobs)}", flush=True)

    # ---- discovery
    def common_prefix(u, v):
        n = 0
        for p, q in zip(u, v):
            if p != q:
                break
            n += 1
        return n

    res = {"questions": len(Q), "hard": len(H), "seeds": a.seeds, "ps_min": a.ps_min}
    min_size, n_edges, n_pairsets, twin_pairs, div_pos, div_share = [], 0, 0, [], [], []
    base_identical_check = []
    for x in H:
        tw = x["tw"]; seeds = range(a.seeds)
        F = [s for s in seeds if tw[((), s)]["r"] == 0]
        if not F or len(x["hints"]) == 0:
            min_size.append("none" if F else "base_ok"); continue
        PS = {S: float(np.mean([tw[(S, s)]["r"] for s in F])) for S in x["sets"] if S}
        singles = [S for S in PS if len(S) == 1 and PS[S] >= a.ps_min]
        pairs = [S for S in PS if len(S) == 2 and PS[S] >= a.ps_min]
        full = [S for S in PS if len(S) > 2 and PS[S] >= a.ps_min]
        min_size.append("1" if singles else "2" if pairs else "all" if full else "none")
        for S in PS:
            if len(S) == 2:
                n_pairsets += 1
                if PS[S] - max(PS[(S[0],)], PS[(S[1],)]) >= a.ps_min:
                    n_edges += 1
        best = max(PS, key=lambda S: (PS[S], -len(S)))
        for s in F:
            if tw[(best, s)]["r"] == 1:
                b0, b1 = tw[((), s)]["ids"], tw[(best, s)]["ids"]
                d = common_prefix(b0, b1)
                div_pos.append(d); div_share.append(d / max(1, len(b0)))
                twin_pairs.append((x, b1, b0, d))
    from collections import Counter
    res["minimal_sufficient_set_size"] = dict(Counter(min_size))
    res["synergy_edges"] = {"edges": n_edges, "pair_sets_tested": n_pairsets}
    res["twin_pairs"] = len(twin_pairs)
    res["divergence_tokens_median"] = float(np.median(div_pos)) if div_pos else None
    res["divergence_share_of_base_median"] = float(np.median(div_share)) if div_share else None
    res["divergence_at_token0_frac"] = float(np.mean([d == 0 for d in div_pos])) if div_pos else None
    # natural pairs: plain success vs plain failure at different seeds (questions with >= 1 plain success)
    nat = []
    for x in H:
        ok = [s for s in x["s1"] if s["r"] == 1 and not s["trunc"]]; bad = [s for s in x["s1"] if s["r"] == 0 and not s["trunc"]]
        for g, b in zip(ok, bad):
            nat.append((x, g["ids"], b["ids"], common_prefix(g["ids"], b["ids"])))
    res["natural_pairs"] = len(nat)
    res["natural_common_prefix_median"] = float(np.median([d for *_, d in nat])) if nat else None
    # examples
    ex = []
    for x, b1, b0, d in twin_pairs[:6]:
        ex.append({"q": x["q"][:200], "hint_count": len(x["hints"]), "shared_prefix": tok.decode(b0[:d])[-200:],
                   "base_after": tok.decode(b0[d:d + 60]), "twin_after": tok.decode(b1[d:d + 60])})
    res["examples"] = ex
    print("DISCOVERY " + json.dumps({k: v for k, v in res.items() if k != "examples"}), flush=True)
    (out / "partial.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))

    # ---- learning direction
    if a.grad and twin_pairs:
        llm.sleep(level=2); del llm; torch.cuda.empty_cache()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=32, lora_alpha=64, lora_dropout=0.0,
                                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        params = [p for p in model.parameters() if p.requires_grad]
        model.train()

        def lp(ctx, comp):
            ids = torch.tensor([ctx + comp], device="cuda")
            logits = model(input_ids=ids, logits_to_keep=len(comp) + 1).logits[0, :-1].float()
            return torch.log_softmax(logits, -1).gather(-1, torch.tensor(comp, device="cuda")[:, None])[:, 0]

        def flat():
            g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).float().flatten() for p in params])
            for p in params:
                p.grad = None
            return g

        def contrast_grad(pairs, after):
            for x, good, bad, d in pairs:
                gi, bi = (good[d:], bad[d:]) if after else (good, bad)
                if not gi or not bi:
                    continue
                gi, bi = gi[:a.max_completion], bi[:a.max_completion]
                loss = -(lp(x["pp"], gi).mean() - lp(x["pp"], bi).mean()) / len(pairs)
                loss.backward()
            return flat()

        def grpo_grad(groups):
            items = []
            for x in groups:
                rs = [s["r"] for s in x["s1"]]; m = float(np.mean(rs))
                for s in x["s1"]:
                    if not s["trunc"] and s["r"] != m:
                        items.append((x, s["ids"], s["r"] - m))
            for x, ids, A in items:
                (-(A * lp(x["pp"], ids[:a.max_completion]).mean()) / max(1, len(items))).backward()
            return flat()

        cos = lambda u, v: float(u @ v / (u.norm() * v.norm() + 1e-12))
        random.shuffle(MIX)
        mA, mB = MIX[: len(MIX) // 2][:40], MIX[len(MIX) // 2:][:40]
        gA, gB = grpo_grad(mA), grpo_grad(mB)
        random.shuffle(twin_pairs); random.shuffle(nat)
        n = min(a.max_pairs, len(twin_pairs), max(1, len(nat)))
        g_after = contrast_grad(twin_pairs[:n], True)
        g_full = contrast_grad(twin_pairs[:n], False)
        g_nat = contrast_grad(nat[:n], False) if nat else None
        res["grad"] = {"n_pairs": n, "anchor_grpoA_vs_grpoB": cos(gA, gB),
                       "twin_after_vs_grpo": cos(g_after, gB), "twin_full_vs_grpo": cos(g_full, gB),
                       "natural_vs_grpo": cos(g_nat, gB) if g_nat is not None else None,
                       "twin_after_vs_natural": cos(g_after, g_nat) if g_nat is not None else None}
        print("GRAD " + json.dumps(res["grad"]), flush=True)
    res["time"] = round(time.time() - t0, 1)
    (out / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("RESULT " + json.dumps({k: v for k, v in res.items() if k != "examples"}), flush=True)
    print(f"DONE {out}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
