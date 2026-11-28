"""Pre-screen for R03-4 `r03_crlsim` (no policy training): does a CONTENT space learned from counterfactual pairs rank a
student's correct solutions above its wrong ones by closeness to y*, better than surface similarity?

Content space = linear head on mean-pooled hidden states of the base model (completion tokens, plain prompt). Trained with
InfoNCE on training questions only: anchor y*, positive = an own CORRECT solution of the same question (same content, other
style), negatives = y~* (same style, 1-2 derived values changed; beat.loop.perturb) and positives of other questions.
Variant `crl_wrongneg` also uses the question's own wrong solutions as negatives (a reference-anchored verifier; upper bound).

Metric on held-out questions: within-group AUC over mixed groups (P[score(correct) > score(wrong)], ties 1/2), which removes
question difficulty. Baselines: untrained pooled-state cosine, lexical token F1 to y* (RL2R-like), y*-teacher log-ratio
mean(log p_T - log p_S) with HINT_TEMPLATE, and -length.

    PYTHONPATH=code python -m beat.diag_crl --questions 1024 --out runs/diag_crl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=1024)
    ap.add_argument("--eval-frac", type=float, default=0.25)
    ap.add_argument("--num-gen", type=int, default=8)
    ap.add_argument("--max-completion", type=int, default=2048)
    ap.add_argument("--layers", default="18,27,36")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--vllm-mem", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from beat.loop import HINT_TEMPLATE, perturb
    from train.common import BASE_MODEL, GRPO, configure_tokenizer
    from train.data import load_track
    from train.models import eos_ids
    from train.rewards import score_all

    t_start = time.time()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tok = configure_tokenizer(AutoTokenizer.from_pretrained(BASE_MODEL), "curated")
    stop_ids = sorted(set(eos_ids(tok, "curated")) | {tok.convert_tokens_to_ids("<|im_end|>")})
    eos_id = tok.convert_tokens_to_ids(tok.eos_token)
    rng = random.Random(a.seed)

    rows = []
    for r in load_track("curated").itertuples():
        msgs = [{"role": m["role"], "content": m["content"]} for m in r.prompt]
        q, ref = r.extra_info["question"], r.target[0]["content"]
        p_ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]
        h_ids = tok(tok.apply_chat_template([msgs[0], {"role": "user", "content": HINT_TEMPLATE.format(question=q, reference=ref)}],
                                            tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]
        if len(p_ids) > 1024 or len(h_ids) > 2048:
            continue
        cf = perturb(ref, "numbers", q, rng)
        if cf == ref:
            continue
        rows.append({"p": p_ids, "h": h_ids, "ref": ref, "y": tok(ref, add_special_tokens=False)["input_ids"] + [eos_id],
                     "cf": tok(cf, add_special_tokens=False)["input_ids"] + [eos_id], "gt": r.reward_model["ground_truth"]})
    order = np.random.permutation(len(rows))[:a.questions].tolist()
    Q = [rows[i] for i in order]
    n_eval = int(len(Q) * a.eval_frac)
    split = ["eval"] * n_eval + ["train"] * (len(Q) - n_eval)

    # ---- 1. student samples from the base model
    llm = LLM(model=BASE_MODEL, gpu_memory_utilization=a.vllm_mem, max_model_len=2048 + a.max_completion, seed=a.seed,
              enable_prefix_caching=True, enable_sleep_mode=True)
    sp = SamplingParams(temperature=GRPO["temperature"], top_p=1.0, max_tokens=a.max_completion, stop_token_ids=stop_ids,
                        n=a.num_gen)
    outs = llm.generate([{"prompt_token_ids": x["p"]} for x in Q], sp, use_tqdm=False)
    llm.sleep(level=2)
    for x, o in zip(Q, outs):
        x["s"] = []
        for c in o.outputs:
            comp = list(c.token_ids)
            if c.finish_reason == "stop" and (not comp or comp[-1] not in stop_ids):
                comp.append(eos_id)
            x["s"].append({"comp": comp, "text": c.text, "trunc": c.finish_reason == "length"})
    rew = score_all([s["text"] for x in Q for s in x["s"]], [x["gt"] for x in Q for _ in x["s"]], 8)
    k = 0
    for x in Q:
        for s in x["s"]:
            s["r"] = float(rew[k]); k += 1
    print(f"GEN done {time.time() - t_start:.0f}s reward {np.mean(rew):.3f}", flush=True)
    torch.cuda.empty_cache()

    # ---- 2. pooled hidden states (plain prompt) for y*, y~*, samples; teacher log-ratio for eval samples
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    LAY = [int(v) for v in a.layers.split(",")]

    @torch.no_grad()
    def run(ctx, comp, want_h=True):
        ids = torch.tensor([ctx + comp], device="cuda")
        o = model(input_ids=ids, output_hidden_states=want_h)
        n = len(comp)
        lg = o.logits[0, len(ctx) - 1:len(ctx) - 1 + n].float()
        lp = torch.log_softmax(lg, -1).gather(-1, torch.tensor(comp, device="cuda")[:, None])[:, 0]
        h = None
        if want_h:
            h = torch.stack([o.hidden_states[L][0, len(ctx):len(ctx) + n].float().mean(0) for L in LAY]).cpu()
        return lp, h

    for qi, x in enumerate(Q):
        _, x["hy"] = run(x["p"], x["y"])
        _, x["hcf"] = run(x["p"], x["cf"])
        for s in x["s"]:
            lpS, s["h"] = run(x["p"], s["comp"])
            if split[qi] == "eval":
                lpT, _ = run(x["h"], s["comp"], want_h=False)
                s["teacher"] = float((lpT - lpS).mean())
        if qi % 128 == 0:
            print(f"FEAT {qi}/{len(Q)} {time.time() - t_start:.0f}s", flush=True)

    # ---- 3. scores
    def toks(text):
        import re
        return Counter(re.findall(r"[A-Za-z]+|\d+|\\[A-Za-z]+|[^\sA-Za-z\d]", text))

    def f1(a_, b_):
        inter = sum((a_ & b_).values())
        return 0.0 if inter == 0 else 2 * inter / (sum(a_.values()) + sum(b_.values()))

    def auc_within(score_fn):
        num = den = 0.0
        for qi, x in enumerate(Q):
            if split[qi] != "eval":
                continue
            c = [score_fn(x, s) for s in x["s"] if s["r"] == 1]
            w = [score_fn(x, s) for s in x["s"] if s["r"] == 0]
            for u in c:
                for v in w:
                    num += 1.0 if u > v else 0.5 if u == v else 0.0
                    den += 1
        return num / den if den else None, int(den)

    res = {"questions": len(Q), "eval_q": n_eval, "reward": float(np.mean(rew)),
           "mixed_eval_groups": sum(1 for qi, x in enumerate(Q) if split[qi] == "eval" and 0 < sum(s["r"] for s in x["s"]) < a.num_gen)}
    for x in Q:
        x["ty"] = toks(x["ref"])
    res["auc_lexical_f1"], res["pairs"] = auc_within(lambda x, s: f1(toks(s["text"]), x["ty"]))
    res["auc_teacher_logratio"], _ = auc_within(lambda x, s: s["teacher"])
    res["auc_neg_length"], _ = auc_within(lambda x, s: -len(s["comp"]))

    tr = [x for qi, x in enumerate(Q) if split[qi] == "train"]
    for li, L in enumerate(LAY):
        allh = torch.stack([s["h"][li] for x in tr for s in x["s"]] + [x["hy"][li] for x in tr])
        mu, sd = allh.mean(0), allh.std(0) + 1e-4
        z = lambda h: (h[li] - mu) / sd
        res[f"auc_untrained_L{L}"], _ = auc_within(lambda x, s: float(F.cosine_similarity(z(s["h"]), z(x["hy"]), 0)))
        for variant in ("crl", "crl_wrongneg"):
            pos = [x for x in tr if any(s["r"] == 1 for s in x["s"])]
            nv = int(0.1 * len(pos)); val, fit = pos[:nv], pos[nv:]
            W = torch.nn.Linear(len(mu), a.dim, bias=False).cuda()
            opt = torch.optim.AdamW(W.parameters(), lr=1e-3, weight_decay=1e-2)
            enc = lambda v: F.normalize(W(v.cuda()), dim=-1)

            def val_acc():
                with torch.no_grad():
                    ok = []
                    for x in val:
                        c = [s for s in x["s"] if s["r"] == 1]
                        ey = enc(z(x["hy"])[None])[0]
                        ok.append(float(enc(z(c[0]["h"])[None])[0] @ ey > enc(z(x["hcf"])[None])[0] @ ey))
                    return float(np.mean(ok)) if ok else None
            best, best_state = -1, None
            for step in range(a.steps):
                B = random.sample(fit, min(64, len(fit)))
                ay = enc(torch.stack([z(x["hy"]) for x in B]))
                pp = enc(torch.stack([z(random.choice([s for s in x["s"] if s["r"] == 1])["h"]) for x in B]))
                hc = enc(torch.stack([z(x["hcf"]) for x in B]))
                logits = torch.cat([ay @ pp.T, (ay * hc).sum(-1, keepdim=True)], 1) / a.tau
                if variant == "crl_wrongneg":
                    wn = []
                    for x in B:
                        ws = [s for s in x["s"] if s["r"] == 0]
                        wn.append(z(random.choice(ws)["h"]) if ws else z(x["hcf"]))
                    wn = enc(torch.stack(wn))
                    logits = torch.cat([logits, (ay * wn).sum(-1, keepdim=True) / a.tau], 1)
                loss = F.cross_entropy(logits, torch.arange(len(B), device="cuda"))
                opt.zero_grad(); loss.backward(); opt.step()
                if step % 50 == 49:
                    acc = val_acc()
                    if acc is not None and acc > best:
                        best, best_state = acc, W.weight.detach().clone()
            if best_state is not None:
                with torch.no_grad():
                    W.weight.copy_(best_state)
            res[f"val_rank_{variant}_L{L}"] = best
            with torch.no_grad():
                res[f"auc_{variant}_L{L}"], _ = auc_within(
                    lambda x, s: float(enc(z(s["h"])[None])[0] @ enc(z(x["hy"])[None])[0]))
                ev = [x for qi, x in enumerate(Q) if split[qi] == "eval" and any(s["r"] == 1 for s in x["s"])]
                res[f"neo_rank_{variant}_L{L}"] = float(np.mean([
                    float(enc(z([s for s in x["s"] if s["r"] == 1][0]["h"])[None])[0] @ enc(z(x["hy"])[None])[0]
                          > enc(z(x["hcf"])[None])[0] @ enc(z(x["hy"])[None])[0]) for x in ev]))
        print("PARTIAL " + json.dumps(res), flush=True)
    res["time"] = round(time.time() - t_start, 1)
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print("RESULT " + json.dumps(res), flush=True)
    print(f"DONE {out}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
