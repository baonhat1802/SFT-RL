"""Pre-screen: SFT+RL as data fusion with hidden confounding ("experimental grounding", Kallus-Puli-Shalit 2018). No training.

Two data sources about the effect of the student's token choices on reward:
  experimental  = the student's own rollouts. GRPO advantage A_i = r_i - mean_group(r) is unbiased for the student, but empty on
                  all-fail groups (every A_i = 0).
  observational = the teacher. Per-token signal d_t = log p(y_t | teacher context with y*) - log p(y_t | plain prompt), same
                  model (on-policy self-distillation reward). Biased: the teacher's choices depend on knowledge U the student
                  lacks (style, skipped steps).
Grounding: on mixed groups (both signals exist) fit the bias by weighted least squares, A_i ~ f(d_t, token class, position), then
use the corrected signal where RL is blind.

Signals compared (per token; a sample's score = token mean):
  raw        d_t                                                     (plain on-policy self-distillation)
  global     a*d_t + b                                               (one scale)
  noteacher  sum_c b_c 1[class=c] + sum_k e_k 1[pos bin=k]            (what RL data alone teaches: a class/position prior)
  corrected  noteacher + sum_c a_c d_t 1[class=c]                     (grounded teacher signal; nests noteacher)
Tests
  T1 held-out mixed groups: within-group AUC (correct vs wrong) and cosine of the per-token-reward gradient with the GRPO
     gradient; anchor = GRPO half A vs half B.
  T2 transport (main): questions all-fail at 8 samples but solvable with 24 more ("hard"). Signal gradients use ONLY the 8
     failed samples (what training would have); the reference is the GRPO gradient of all 32 samples. Also AUC over 32.
PASS (written before running)
  T1: corrected beats max(raw, noteacher) by >= 0.03 AUC or >= 0.05 cosine.
  T2: cos(corrected) - max(cos(raw), cos(noteacher)) >= 0.05 and cos(corrected) > 0.

    PYTHONPATH=code python -m beat.diag_fusion --out runs/diag_fusion
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

CLS = ("prose", "mnum", "msym", "num_out", "ws", "box")
NPOS = 3
MATH = re.compile(r"\\\((.*?)\\\)|\\\[(.*?)\\\]|\$\$(.*?)\$\$|\$(.*?)\$", re.S)


def classify(pieces):
    """Token classes from decoded pieces: box (from the last \\boxed on), math with/without digit, digit outside math,
    letters outside math (prose), else whitespace/punctuation."""
    text = "".join(pieces)
    offs, c = [], 0
    for p in pieces:
        offs.append((c, c + len(p))); c += len(p)
    spans = [(m.start(), m.end()) for m in MATH.finditer(text)]
    bx = text.rfind("\\boxed")
    out = []
    for (u, v), p in zip(offs, pieces):
        if bx >= 0 and u >= bx - 1:
            out.append("box")
        elif any(u < e and v > s for s, e in spans):
            out.append("mnum" if any(ch.isdigit() for ch in p) else "msym")
        elif any(ch.isdigit() for ch in p):
            out.append("num_out")
        elif any(ch.isalpha() for ch in p):
            out.append("prose")
        else:
            out.append("ws")
    return out


def feats(d, cls, n, variant):
    """Feature rows for one sample (numpy). d: per-token teacher signal, cls: class names, n: length."""
    import numpy as np
    C = np.zeros((n, len(CLS))); C[np.arange(n), [CLS.index(c) for c in cls]] = 1
    P = np.zeros((n, NPOS)); P[np.arange(n), np.minimum((np.arange(n) * NPOS) // max(1, n), NPOS - 1)] = 1
    if variant == "global":
        return np.stack([d, np.ones(n)], 1)
    if variant == "noteacher":
        return np.concatenate([C, P], 1)
    if variant == "corrected":
        return np.concatenate([C, P, C * d[:, None]], 1)
    raise ValueError(variant)


def wls(X, y, w, ridge=1e-4):
    import numpy as np
    XtW = X.T * w
    A = XtW @ X
    lam = ridge * np.trace(A) / A.shape[0]
    return np.linalg.solve(A + lam * np.eye(A.shape[0]), XtW @ y)


def auc_within(groups):
    """groups: list of [(score, r)]; mean over groups with both outcomes of P(score_correct > score_wrong) (+0.5 ties)."""
    import numpy as np
    vals = []
    for g in groups:
        pos = [s for s, r in g if r == 1]; neg = [s for s, r in g if r == 0]
        if pos and neg:
            vals.append(np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg]))
    return (float(np.mean(vals)), len(vals)) if vals else (None, 0)


def selftest():
    import numpy as np
    assert classify(["We", " get", " $", "x", "=", "12", "$", ".", " So", " \\boxed", "{3}"]) == \
        ["prose", "prose", "msym", "msym", "msym", "mnum", "msym", "ws", "prose", "box", "box"]
    X = np.array([[1.0, 0], [0, 1], [1, 1]]); b = wls(X, X @ np.array([2.0, -1]), np.ones(3), ridge=0)
    assert np.allclose(b, [2, -1])
    assert auc_within([[(0.9, 1), (0.1, 0)], [(0.2, 1), (0.8, 0), (0.5, 0)]])[0] == 0.5
    f = feats(np.array([0.5, -1.0, 2.0]), ["prose", "mnum", "box"], 3, "corrected")
    assert f.shape == (3, 15) and f[1, CLS.index("mnum")] == 1 and f[1, 9 + CLS.index("mnum")] == -1.0
    print("SELFTEST_OK", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=640)
    ap.add_argument("--num-gen", type=int, default=8)
    ap.add_argument("--extra", type=int, default=24)
    ap.add_argument("--max-completion", type=int, default=1536)
    ap.add_argument("--max-mixed", type=int, default=200)
    ap.add_argument("--max-hard", type=int, default=60)
    ap.add_argument("--grad-groups", type=int, default=40)
    ap.add_argument("--tok-budget", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--diag", type=int, default=0, help="1: extra decomposition variants, per-class AUC, shift stats, dump samples")
    a = ap.parse_args()
    selftest()
    if a.selftest:
        return

    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from beat import fastlp
    from beat.loop import TEACHER_TEMPLATE
    from train.common import BASE_MODEL, GRPO, configure_tokenizer
    from train.data import load_track
    from train.models import eos_ids
    from train.rewards import score_all

    t0 = time.time()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tok = configure_tokenizer(AutoTokenizer.from_pretrained(BASE_MODEL), "curated")
    stop_ids = sorted(set(eos_ids(tok, "curated")) | {tok.convert_tokens_to_ids("<|im_end|>")})
    pad_id = tok.pad_token_id

    rows = []
    for r in load_track("curated").itertuples():
        msgs = [{"role": m["role"], "content": m["content"]} for m in r.prompt]
        q, ref = r.extra_info["question"], r.target[0]["content"]
        enc = lambda user: tok(tok.apply_chat_template([msgs[0], {"role": "user", "content": user}], tokenize=False,
                                                       add_generation_prompt=True), add_special_tokens=False)["input_ids"]
        rows.append({"p": tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
                              add_special_tokens=False)["input_ids"],
                     "tp": enc(TEACHER_TEMPLATE.format(question=q, reference=ref)), "gt": r.reward_model["ground_truth"]})
    rows = [x for x in rows if len(x["p"]) <= 1024 and len(x["tp"]) <= 2048]
    Q = [rows[i] for i in np.random.permutation(len(rows))[:a.questions].tolist()]

    llm = LLM(model=BASE_MODEL, gpu_memory_utilization=0.5, max_model_len=2048 + a.max_completion, seed=a.seed,
              enable_sleep_mode=True)
    sp = lambda n: SamplingParams(temperature=GRPO["temperature"], top_p=1.0, max_tokens=a.max_completion,
                                  stop_token_ids=stop_ids, n=n)

    def gen(xs, n):
        o = llm.generate([{"prompt_token_ids": x["p"]} for x in xs], sp(n), use_tqdm=False)
        comps = [[list(c.token_ids) for c in oo.outputs] for oo in o]
        txt = [c.text for oo in o for c in oo.outputs]
        rr = score_all(txt, [x["gt"] for x, oo in zip(xs, o) for _ in oo.outputs], 8)
        k = 0
        for x, cs in zip(xs, comps):
            x.setdefault("S", [])
            for c in cs:
                comp = [t for t in c if t not in stop_ids] or [stop_ids[0]]   # never empty
                x["S"].append({"comp": comp, "r": float(rr[k])}); k += 1

    gen(Q, a.num_gen)
    nc = lambda x: sum(s["r"] for s in x["S"])
    MIX = [x for x in Q if 0 < nc(x) < a.num_gen][:a.max_mixed]
    AF = [x for x in Q if nc(x) == 0]
    stats = {"questions": len(Q), "mixed": sum(0 < nc(x) < a.num_gen for x in Q), "allfail": len(AF),
             "allcorrect": sum(nc(x) == a.num_gen for x in Q)}
    gen(AF, a.extra)
    HARD = [x for x in AF if 0 < nc(x) < a.num_gen + a.extra][:a.max_hard]
    stats["allfail_rescued_by_extra"] = sum(0 < nc(x) for x in AF)
    stats["hard_used"] = len(HARD)
    print(f"GEN {time.time() - t0:.0f}s " + json.dumps(stats), flush=True)

    llm.sleep(level=2); del llm; torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=32, lora_alpha=64, lora_dropout=0.0,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                                                             "down_proj"]))
    params = [p for p in model.parameters() if p.requires_grad]

    # ---- per-token teacher signal d_t and classes (LoRA B = 0, so the model is exactly the base)
    samples = [(x, s) for x in MIX + HARD for s in x["S"]]
    model.eval()
    with torch.no_grad():
        lS = fastlp.logps(model, pad_id, [x["p"] for x, _ in samples], [s["comp"] for _, s in samples], False, 2 * a.tok_budget)
        lT = fastlp.logps(model, pad_id, [x["tp"] for x, _ in samples], [s["comp"] for _, s in samples], False, 2 * a.tok_budget)
    for (x, s), u, v in zip(samples, lS, lT):
        s["d"] = (v - u).cpu().numpy().astype("float64")
        s["cls"] = classify([tok.decode([t]) for t in s["comp"]])
    for x in MIX + HARD:
        m = np.mean([s["r"] for s in x["S"]])
        for s in x["S"]:
            s["A"] = s["r"] - m
    print(f"LOGP {time.time() - t0:.0f}s samples {len(samples)}", flush=True)

    # ---- fit the bias on half of the mixed groups
    random.shuffle(MIX)
    TR, TE = MIX[: len(MIX) // 2], MIX[len(MIX) // 2:]
    beta = {}
    for v in ("global", "noteacher", "corrected"):
        Xs, ys, ws = [], [], []
        for x in TR:
            for s in x["S"]:
                n = len(s["comp"])
                Xs.append(feats(s["d"], s["cls"], n, v)); ys.append(np.full(n, s["A"])); ws.append(np.full(n, 1.0 / n))
        beta[v] = wls(np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws))

    MATHC = np.array([c in ("mnum", "msym", "box") for c in CLS])

    def sig(s, v):
        if v == "raw":
            return s["d"]
        ci = np.array([CLS.index(c) for c in s["cls"]])
        if v == "nopros":      # raw teacher signal with prose tokens zeroed
            return s["d"] * (ci != CLS.index("prose"))
        if v == "mathonly":    # raw teacher signal on math / box tokens only
            return s["d"] * MATHC[ci]
        if v in ("slope", "intercept"):   # the two halves of the corrected fit
            f = feats(s["d"], s["cls"], len(s["comp"]), "corrected")
            return f[:, 9:] @ beta["corrected"][9:] if v == "slope" else f[:, :9] @ beta["corrected"][:9]
        return feats(s["d"], s["cls"], len(s["comp"]), v) @ beta[v]

    VARS = ("raw", "global", "noteacher", "corrected") + (("slope", "intercept", "nopros", "mathonly") if a.diag else ())
    res = {"gen": stats, "fit_groups": len(TR), "test_groups": len(TE),
           "coef_corrected": {"intercept_class": dict(zip(CLS, beta["corrected"][:6].round(5).tolist())),
                              "intercept_pos": beta["corrected"][6:9].round(5).tolist(),
                              "slope_d_by_class": dict(zip(CLS, beta["corrected"][9:].round(5).tolist()))},
           "coef_global": beta["global"].round(5).tolist(),
           "mean_d_by_class": {}}
    for c in CLS:
        v_ = np.concatenate([s["d"][[k for k, cc in enumerate(s["cls"]) if cc == c]] for x in TR for s in x["S"]])
        res["mean_d_by_class"][c] = float(v_.mean()) if len(v_) else None

    # ---- AUC
    res["T1_auc"] = {v: auc_within([[(float(np.mean(sig(s, v))), s["r"]) for s in x["S"]] for x in TE])[0] for v in VARS}
    res["T2_auc32"] = {v: auc_within([[(float(np.mean(sig(s, v))), s["r"]) for s in x["S"]] for x in HARD])[0] for v in VARS}
    print("AUC " + json.dumps({"T1": res["T1_auc"], "T2": res["T2_auc32"]}), flush=True)

    # ---- gradients of token-mean "per-token reward" losses
    model.train()

    def wgrad(items):
        """items: [(ctx, comp, w np.array)] -> flattened gradient of -sum_i sum_t w_it log p(y_it) / ntok."""
        for p in params:
            p.grad = None
        items = [it for it in items if np.any(it[2] != 0)]
        if not items:
            return None
        ntok = sum(len(it[1]) for it in items)
        for b in fastlp.token_batches([len(c) + len(d) for c, d, _ in items], a.tok_budget):
            mb = [items[i] for i in b]
            lp = fastlp.logps(model, pad_id, [c for c, _, _ in mb], [d for _, d, _ in mb], True, a.tok_budget)
            loss = sum(-(torch.tensor(w, device=l.device, dtype=torch.float32) * l).sum() for (_, _, w), l in zip(mb, lp))
            (loss / ntok).backward()
        g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).float().flatten() for p in params])
        for p in params:
            p.grad = None
        return g, ntok

    def cos(u, v):
        u = u[0] if isinstance(u, tuple) else u
        v = v[0] if isinstance(v, tuple) else v
        return None if u is None or v is None else float(u @ v / (u.norm() * v.norm() + 1e-12))

    def rl_items(xs):
        return [(x["p"], s["comp"], np.full(len(s["comp"]), s["A"])) for x in xs for s in x["S"]]

    def combine(ga, gb):  # token-mean gradient of the union from the two halves (either half may be empty)
        if ga is None or gb is None:
            return ga or gb
        (u, nu), (v, nv) = ga, gb
        return (u * nu + v * nv) / (nu + nv)

    # T1: held-out mixed groups
    G = TE[:a.grad_groups]
    hA, hB = G[: len(G) // 2], G[len(G) // 2:]
    gA, gB = wgrad(rl_items(hA)), wgrad(rl_items(hB))
    g_rl = combine(gA, gB)
    t1 = {"anchor_grpo_halfA_vs_halfB": cos(gA, gB)}
    for v in VARS:
        g = wgrad([(x["p"], s["comp"], sig(s, v)) for x in G for s in x["S"]])
        t1[f"cos_{v}_vs_grpo"] = cos(g, g_rl)
    res["T1_grad"] = t1
    print("T1 " + json.dumps(t1), flush=True)

    # T2: transport to all-fail-at-8 questions; signals see only the 8 failed samples, reference = GRPO over all 32
    H = HARD[:a.grad_groups]
    hA, hB = H[: len(H) // 2], H[len(H) // 2:]
    gA, gB = wgrad(rl_items(hA)), wgrad(rl_items(hB))
    g32 = combine(gA, gB)
    t2 = {"questions": len(H), "anchor_grpo32_halfA_vs_halfB": cos(gA, gB)}
    for v in VARS:
        g = wgrad([(x["p"], s["comp"], sig(s, v)) for x in H for s in x["S"][:a.num_gen]])
        t2[f"cos_{v}_on8failed_vs_grpo32"] = cos(g, g32)
    res["T2_grad"] = t2
    print("T2 " + json.dumps(t2), flush=True)

    best = lambda dct, key: max(dct[key.format("raw")] or -9, dct[key.format("noteacher")] or -9)
    au = res["T1_auc"]
    res["T1_PASS"] = bool((au["corrected"] or 0) - max(au["raw"] or 0, au["noteacher"] or 0) >= 0.03
                          or (t1["cos_corrected_vs_grpo"] or -9) - best(t1, "cos_{}_vs_grpo") >= 0.05)
    c2 = t2["cos_corrected_on8failed_vs_grpo32"] or -9
    res["T2_PASS"] = bool(c2 - best(t2, "cos_{}_on8failed_vs_grpo32") >= 0.05 and c2 > 0)
    if a.diag:
        # per-class AUC of the raw teacher signal (sample score = mean d over that class's tokens), and distribution shift
        def cls_auc(xs, c):
            gs = []
            for x in xs:
                g = []
                for s_ in x["S"]:
                    m = [k for k, cc in enumerate(s_["cls"]) if cc == c]
                    if m:
                        g.append((float(np.mean(s_["d"][m])), s_["r"]))
                gs.append(g)
            return auc_within(gs)[0]
        def shift(xs):
            tot = sum(len(s_["comp"]) for x in xs for s_ in x["S"])
            o = {}
            for c in CLS:
                v_ = np.concatenate([s_["d"][[k for k, cc in enumerate(s_["cls"]) if cc == c]] for x in xs for s_ in x["S"]])
                o[c] = {"share": round(len(v_) / tot, 4), "mean_d": float(v_.mean()) if len(v_) else None,
                        "mean_abs_d": float(np.abs(v_).mean()) if len(v_) else None}
            o["len_mean"] = float(np.mean([len(s_["comp"]) for x in xs for s_ in x["S"]]))
            return o
        res["diag_auc_by_class"] = {"T1": {c: cls_auc(TE, c) for c in CLS}, "T2": {c: cls_auc(HARD, c) for c in CLS}}
        res["diag_shift"] = {"mixed_test": shift(TE), "hard": shift(HARD)}
        import pickle
        dump = [{"kind": "mixed_train" if any(x is y for y in TR) else "mixed_test" if any(x is y for y in TE) else "hard",
                 "q": qi, "r": s_["r"], "A": s_["A"], "d": s_["d"].astype("float32"),
                 "cls": np.array([CLS.index(c) for c in s_["cls"]], dtype="int8")}
                for qi, x in enumerate(MIX + HARD) for s_ in x["S"]]
        with open(out / "samples.pkl", "wb") as f:
            pickle.dump({"CLS": CLS, "samples": dump, "beta": {k: v.tolist() for k, v in beta.items()}}, f)
        print("DIAG " + json.dumps({"auc_by_class": res["diag_auc_by_class"]}), flush=True)
    res["time"] = round(time.time() - t0, 1)
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print("RESULT " + json.dumps({"T1_PASS": res["T1_PASS"], "T2_PASS": res["T2_PASS"]}), flush=True)
    print(f"DONE {out}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
