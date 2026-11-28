"""Pre-screen R06-7: learn the CAUSAL MEDIATOR of the teacher's effect from a randomised experiment (no training of the policy).

Experiment (as diag_iv): at a state s (prefix of a failed student trajectory) one segment is written by the same model under a
randomised context Z in {z0 plain, z1 with y*, zp with another question's y*}; then the context is plain again and the student
continues; Y = reward. Z is randomised, so the effect of Z on Y is identified (diag_iv: 4.5% -> 9.4%).
Mediator. M = a function of the segment computed by the STUDENT (plain-context hidden states, mean over the segment tokens). A score
h(M) is a full mediator of Z when Y is independent of Z given M (front-door / exclusion); testable implication used here: the mean
residual E[Y - h(M) | Z = z] is the same for every arm. Fitted predictors (logistic on PCA-256 features of two layers):
  erm       plain fit of Y
  fd(l)     erm + l * sum_z (mean residual in arm z)^2      ("front-door moment" penalty), l in a grid
Baselines: segment length, F (segment establishes the teacher's next missing fact, diag_iv definition).
Held-out questions (50%):
  M1 mediation: share of the teacher effect explained, 1 - RF_resid / RF, RF = mean_s[Y_z1 - Y_z0], RF_resid same on Y - h(M).
  M2 transfer: within-state AUC of h on the student's OWN segments (z0), all-fail states (what GRPO would score).
  also: how well M detects the arm (AUC z1 vs z0) -> if M is just "looks teacher-written", M2 will not improve.
PASS (written before running)
  M1: the chosen fd model explains >= 50% of the effect AND its pooled held-out AUC >= erm AUC - 0.02.
  M2: chosen fd within-state AUC on z0 segments of all-fail states >= 0.60 and >= erm + 0.02.
  chosen fd = largest l whose validation AUC (20% of the training questions) >= erm validation AUC - 0.02.

    PYTHONPATH=code python -m beat.diag_mediator --out runs/diag_mediator
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from fractions import Fraction
from pathlib import Path


def within_auc(groups):
    """groups: lists of (score, y); mean within-group P(score_pos > score_neg)."""
    import numpy as np
    v = []
    for g in groups:
        pos = [s for s, y in g if y == 1]; neg = [s for s, y in g if y == 0]
        if pos and neg:
            P, N = np.array(pos)[:, None], np.array(neg)[None, :]
            v.append(float(((P > N) + 0.5 * (P == N)).mean()))
    return (float(np.mean(v)), len(v)) if v else (None, 0)


def avg_rank(v):
    """1-based ranks with ties averaged (Mann-Whitney)."""
    import numpy as np
    v = np.asarray(v, dtype=float); o = np.argsort(v, kind="mergesort"); r = np.empty(len(v)); r[o] = np.arange(1, len(v) + 1)
    for u in np.unique(v):
        m = v == u
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def selftest():
    assert within_auc([[(0.9, 1), (0.1, 0)], [(0.1, 1), (0.9, 0)]])[0] == 0.5
    assert list(avg_rank([1, 1, 2])) == [1.5, 1.5, 3.0]
    print("SELFTEST_OK", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=640)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seg", type=int, default=96)
    ap.add_argument("--cuts", default="0.33,0.66")
    ap.add_argument("--fails-allfail", type=int, default=4)
    ap.add_argument("--fails-mixed", type=int, default=2)
    ap.add_argument("--max-states", type=int, default=900)
    ap.add_argument("--max-completion", type=int, default=1536)
    ap.add_argument("--layers", default="18,35")
    ap.add_argument("--tok-budget", type=int, default=24576)
    ap.add_argument("--seed", type=int, default=74)
    ap.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest()
    if a.selftest:
        return

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from beat import fastlp
    from beat.diag_causal import facts_of, nums
    from beat.loop import TEACHER_TEMPLATE
    from train.common import BASE_MODEL, GRPO, configure_tokenizer
    from train.data import load_track
    from train.models import eos_ids
    from train.rewards import score_all

    t0 = time.time()
    T = GRPO["temperature"]
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tok = configure_tokenizer(AutoTokenizer.from_pretrained(BASE_MODEL), "curated")
    stop_ids = sorted(set(eos_ids(tok, "curated")) | {tok.convert_tokens_to_ids("<|im_end|>")})
    pad_id = tok.pad_token_id
    NL = {i for i, t_ in enumerate(tok.convert_ids_to_tokens(list(range(len(tok))))) if t_ is not None and "Ċ" in t_}

    rows = []
    for r in load_track("curated").itertuples():
        msgs = [{"role": m["role"], "content": m["content"]} for m in r.prompt]
        q, ref, ans = r.extra_info["question"], r.target[0]["content"], str(r.extra_info["answer"]).strip()
        rows.append({"msgs": msgs, "q": q, "ref": ref, "gt": r.reward_model["ground_truth"],
                     "p": tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
                              add_special_tokens=False)["input_ids"], "facts": facts_of(ref, q, ans, 12, mask=False)})
    rows = [x for x in rows if len(x["p"]) <= 1024 and any(f["sig"] for f in x["facts"])]
    Q = [rows[i] for i in np.random.permutation(len(rows))[:a.questions].tolist()]
    enc_t = lambda x, ref: tok(tok.apply_chat_template([x["msgs"][0], {"role": "user", "content": TEACHER_TEMPLATE.format(
        question=x["q"], reference=ref)}], tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]
    for qi, x in enumerate(Q):
        x["qi"] = qi; x["tp"] = enc_t(x, x["ref"])
        x["pp"] = enc_t(x, Q[(qi + 1 + random.randrange(len(Q) - 1)) % len(Q)]["ref"])

    llm = LLM(model=BASE_MODEL, gpu_memory_utilization=0.6, max_model_len=4096 + a.max_completion, seed=a.seed,
              enable_prefix_caching=True, enable_sleep_mode=True)
    sp = lambda n, mx: SamplingParams(temperature=T, top_p=1.0, max_tokens=mx, stop_token_ids=stop_ids, n=n)
    o = llm.generate([{"prompt_token_ids": x["p"]} for x in Q], sp(8, a.max_completion), use_tqdm=False)
    rr = score_all([c.text for oo in o for c in oo.outputs], [x["gt"] for x, oo in zip(Q, o) for _ in oo.outputs], 8)
    k = 0
    for x, oo in zip(Q, o):
        x["S"] = []
        for c in oo.outputs:
            x["S"].append({"comp": [t for t in c.token_ids if t not in stop_ids], "r": float(rr[k])}); k += 1
        x["nc"] = int(sum(s["r"] for s in x["S"]))
    cuts = [float(c) for c in a.cuts.split(",")]
    states = []
    for x in Q:
        kind, nf = ("allfail", a.fails_allfail) if x["nc"] == 0 else ("mixed12", a.fails_mixed) if x["nc"] <= 2 else (None, 0)
        if kind is None:
            continue
        for s_ in [s for s in x["S"] if s["r"] == 0 and len(s["comp"]) >= 40][:nf]:
            tau = s_["comp"]
            bounds = [i + 1 for i, t_ in enumerate(tau) if t_ in NL and 10 <= i + 1 <= len(tau) - 20]
            used = set()
            for fr in cuts:
                if not bounds:
                    break
                t = min(bounds, key=lambda v: abs(v - fr * len(tau)))
                if t in used:
                    continue
                used.add(t)
                seen = nums(tok.decode(tau[:t]))
                f = next((f for f in x["facts"] if f["sig"] and not any(Fraction(v) in seen for v in f["sig"])), None)
                states.append({"x": x, "kind": kind, "t": t, "tau": tau,
                               "sig": [Fraction(v) for v in f["sig"]] if f else []})
    random.shuffle(states)
    states = states[:a.max_states]
    ARMS = ("z0", "z1", "zp")
    ctx = {"z0": lambda s: s["x"]["p"], "z1": lambda s: s["x"]["tp"], "zp": lambda s: s["x"]["pp"]}
    seg_jobs = [(si, arm) for si in range(len(states)) for arm in ARMS]
    so = llm.generate([{"prompt_token_ids": ctx[arm](states[si]) + states[si]["tau"][:states[si]["t"]]} for si, arm in seg_jobs],
                      sp(a.k, a.seg), use_tqdm=False)
    segs = []
    for (si, arm), oo in zip(seg_jobs, so):
        s = states[si]
        for c in oo.outputs:
            seg = [t_ for t_ in c.token_ids if t_ not in stop_ids]
            ended = c.finish_reason == "stop"
            sn = nums(tok.decode(seg))
            segs.append({"si": si, "arm": arm, "seg": seg, "ended": ended, "F": int(any(v in sn for v in s["sig"]))})
    need = [g for g in segs if not g["ended"] and a.max_completion - states[g["si"]]["t"] - len(g["seg"]) > 0]
    co = llm.generate([{"prompt_token_ids": states[g["si"]]["x"]["p"] + states[g["si"]]["tau"][:states[g["si"]]["t"]] + g["seg"]}
                       for g in need],
                      [sp(1, a.max_completion - states[g["si"]]["t"] - len(g["seg"])) for g in need], use_tqdm=False)
    for g, oo in zip(need, co):
        g["cont"] = [t_ for t_ in oo.outputs[0].token_ids if t_ not in stop_ids]
    ys = score_all([tok.decode(states[g["si"]]["tau"][:states[g["si"]]["t"]] + g["seg"] + g.get("cont", [])) for g in segs],
                   [states[g["si"]]["x"]["gt"] for g in segs], 8)
    for g, y in zip(segs, ys):
        g["Y"] = float(y)
    print(f"GEN {time.time() - t0:.0f}s states {len(states)} segments {len(segs)} "
          + json.dumps({arm: round(float(np.mean([g['Y'] for g in segs if g['arm'] == arm])), 4) for arm in ARMS}), flush=True)
    llm.sleep(level=2); del llm; torch.cuda.empty_cache()

    # ---- student-side features of each segment (plain context), two layers, mean over segment tokens
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    layers = [int(v) for v in a.layers.split(",")]
    cap = {}
    hooks = [model.model.layers[L].register_forward_hook(lambda m, i, o_, L=L: cap.__setitem__(L, o_[0] if isinstance(o_, tuple) else o_))
             for L in layers]
    ctxs = [states[g["si"]]["x"]["p"] + states[g["si"]]["tau"][:states[g["si"]]["t"]] for g in segs]
    feats = np.zeros((len(segs), 2048 * len(layers)), dtype=np.float32)
    lens = [len(c) + len(g["seg"]) for c, g in zip(ctxs, segs)]
    with torch.no_grad():
        for idx in fastlp.token_batches(lens, a.tok_budget):
            seqs = [ctxs[i] + segs[i]["seg"] for i in idx]
            L_ = max(map(len, seqs))
            ids = torch.full((len(seqs), L_), pad_id, dtype=torch.long); att = torch.zeros((len(seqs), L_), dtype=torch.long)
            for j, s_ in enumerate(seqs):
                ids[j, L_ - len(s_):] = torch.tensor(s_); att[j, L_ - len(s_):] = 1
            ids, att = ids.cuda(), att.cuda()
            model.model(input_ids=ids, attention_mask=att, position_ids=(att.cumsum(-1) - 1).clamp(min=0), use_cache=False)
            for j, i in enumerate(idx):
                n = max(1, len(segs[i]["seg"]))          # empty segment -> last context token
                feats[i] = torch.cat([cap[L][j, L_ - n:].float().mean(0) for L in layers]).cpu().numpy()
    for h in hooks:
        h.remove()
    del model; torch.cuda.empty_cache()
    print(f"FEAT {time.time() - t0:.0f}s {feats.shape}", flush=True)

    # ---- splits by question
    qids = sorted({states[g["si"]]["x"]["qi"] for g in segs})
    rng = random.Random(a.seed); rng.shuffle(qids)
    te_q = set(qids[: len(qids) // 2]); va_q = set(qids[len(qids) // 2: len(qids) // 2 + len(qids) // 10])
    split = np.array([0 if states[g["si"]]["x"]["qi"] in te_q else 1 if states[g["si"]]["x"]["qi"] in va_q else 2 for g in segs])
    Y = np.array([g["Y"] for g in segs], dtype=np.float32)
    Zi = np.array([ARMS.index(g["arm"]) for g in segs])
    Xt = torch.tensor(feats, device="cuda")
    mu = Xt[split == 2].mean(0); sd = Xt[split == 2].std(0) + 1e-4
    Xs = (Xt - mu) / sd
    U, S_, V = torch.pca_lowrank(Xs[torch.tensor(split == 2, device="cuda")], q=min(256, int((split == 2).sum()) - 1), center=False)
    P = Xs @ V
    P = P / P[torch.tensor(split == 2, device="cuda")].std(0)
    Yt = torch.tensor(Y, device="cuda"); Zt = torch.tensor(Zi, device="cuda")

    def fit(lam, steps=1500, wd=1e-2):
        tr = torch.tensor(split == 2, device="cuda")
        w = torch.zeros(P.shape[1], device="cuda", requires_grad=True); b = torch.zeros(1, device="cuda", requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=0.02)
        pw = float((1 - Y[split == 2].mean()) / max(Y[split == 2].mean(), 1e-3))
        for _ in range(steps):
            logit = P[tr] @ w + b; y = Yt[tr]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y, pos_weight=torch.tensor(pw, device="cuda"))
            if lam > 0:
                pr = torch.sigmoid(logit); z = Zt[tr]
                pen = sum(((y[z == k_] - pr[z == k_]).mean() - (y - pr).mean()) ** 2 for k_ in range(3))
                loss = loss + lam * pen
            loss = loss + wd * (w ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            return torch.sigmoid(P @ w + b).cpu().numpy()

    def pooled_auc(score, mask):
        y = Y[mask]; s_ = score[mask]
        pos, neg = s_[y == 1], s_[y == 0]
        if len(pos) == 0 or len(neg) == 0:
            return None
        rk = avg_rank(np.concatenate([pos, neg]))
        return float((rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

    def evaluate(score, sel):
        """sel: segment mask (test split). Mediation share, within-state AUC on own segments of all-fail states, arm AUC."""
        by = defaultdict(lambda: defaultdict(list))
        for i in np.where(sel)[0]:
            by[segs[i]["si"]][segs[i]["arm"]].append(i)
        rf, rfr = [], []
        for si, d in by.items():
            if d["z0"] and d["z1"]:
                rf.append(Y[d["z1"]].mean() - Y[d["z0"]].mean())
                rfr.append((Y[d["z1"]] - score[d["z1"]]).mean() - (Y[d["z0"]] - score[d["z0"]]).mean())
        RF, RFr = float(np.mean(rf)), float(np.mean(rfr))
        own = [[(float(score[i]), Y[i]) for i in d["z0"]] for si, d in by.items() if states[si]["kind"] == "allfail"]
        armmask = sel & (Zi != 2)
        arm_auc = None
        s_ = score[armmask]; z_ = Zi[armmask]
        if (z_ == 1).any() and (z_ == 0).any():
            pos, neg = s_[z_ == 1], s_[z_ == 0]
            rk = avg_rank(np.concatenate([pos, neg]))
            arm_auc = float((rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
        wa, nwa = within_auc(own)
        return {"RF": RF, "RF_resid": RFr, "mediated_share": (1 - RFr / RF) if RF else None,
                "auc_pooled": pooled_auc(score, sel), "auc_within_own_allfail": wa, "n_states_auc": nwa,
                "auc_detect_teacher_arm": arm_auc}

    te = split == 0; va = split == 1
    res = {"config": vars(a), "segments": len(segs), "states": len(states),
           "Y_by_arm": {arm: float(Y[Zi == i].mean()) for i, arm in enumerate(ARMS)}}
    models = {"erm": fit(0.0)}
    for lam in (3.0, 30.0, 300.0, 3000.0):
        models[f"fd{int(lam)}"] = fit(lam)
    lens_seg = np.array([len(g["seg"]) for g in segs], dtype=np.float32)
    models["base_length"] = lens_seg / lens_seg.max()
    models["base_F"] = np.array([g["F"] for g in segs], dtype=np.float32)
    res["val_auc"] = {k_: pooled_auc(v, va) for k_, v in models.items()}
    res["test"] = {k_: evaluate(v, te) for k_, v in models.items()}
    ok = [k_ for k_ in models if k_.startswith("fd") and (res["val_auc"][k_] or 0) >= (res["val_auc"]["erm"] or 0) - 0.02]
    chosen = max(ok, key=lambda k_: int(k_[2:])) if ok else None
    res["chosen"] = chosen
    if chosen:
        c, e = res["test"][chosen], res["test"]["erm"]
        res["M1_PASS"] = bool((c["mediated_share"] or 0) >= 0.5 and (c["auc_pooled"] or 0) >= (e["auc_pooled"] or 0) - 0.02)
        res["M2_PASS"] = bool((c["auc_within_own_allfail"] or 0) >= 0.60
                              and (c["auc_within_own_allfail"] or 0) >= (e["auc_within_own_allfail"] or 0) + 0.02)
    else:
        res["M1_PASS"] = res["M2_PASS"] = False
    res["time"] = round(time.time() - t0, 1)
    (out / "results.json").write_text(json.dumps(res, indent=2))
    np.savez_compressed(out / "data.npz", P=P.cpu().numpy().astype("float16"), Y=Y, Z=Zi, split=split,
                        si=np.array([g["si"] for g in segs]), q=np.array([states[g["si"]]["x"]["qi"] for g in segs]),
                        kind=np.array([states[g["si"]]["kind"] == "allfail" for g in segs]), F=models["base_F"], L=lens_seg)
    print("VAL " + json.dumps(res["val_auc"]), flush=True)
    for k_, v in res["test"].items():
        print(f"TEST {k_} " + json.dumps({kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}), flush=True)
    print("RESULT " + json.dumps({"chosen": chosen, "M1_PASS": res["M1_PASS"], "M2_PASS": res["M2_PASS"]}), flush=True)
    print(f"DONE {out}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
