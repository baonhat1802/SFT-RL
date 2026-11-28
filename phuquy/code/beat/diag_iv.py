"""Pre-screen R06-6: the teacher as an INSTRUMENT, actions defined by CONTENT (no training).

Treatment F (coarsened, content-level): "the next segment of the student establishes the teacher's next missing fact" = the earliest
step of y* none of whose new numbers appear in the student's prefix (answer step excluded); established = one of those numbers appears
in the segment. Instrument Z, randomised per sample at a state s (prefix of a failed student trajectory):
  Z = 0  segment written by the student with the plain prompt          (pi)
  Z = 1  segment written by the SAME model with y* in the prompt        (q; TEACHER_TEMPLATE)
  Z = P  placebo: same template with the reference of ANOTHER question  (exclusion / anchor arm)
After the segment the context is plain again and the student continues to the end; Y = reward.
Exclusion: Z can affect Y only through the segment (the reference is removed afterwards).
Estimators (pooled over states, ratio of means = 2SLS with state fixed effects and equal cell sizes):
  first stage  FS  = mean_s [P(F|Z=1) - P(F|Z=0)]
  reduced form RF  = mean_s [P(Y|Z=1) - P(Y|Z=0)]
  LATE         = RF / FS   (effect of establishing F for states whose behaviour the hint changes = the learnable region)
  naive        = within Z=0, mean_s [E(Y|F=1) - E(Y|F=0)]  (observational, confounded by the student's latent "on track" state)
95% CIs by bootstrap over questions.
PASS (written before running)
  I1 strong, valid instrument: FS >= 0.10 with CI > 0; placebo first stage CI contains 0 or |FS_placebo| < 0.03.
  I2 overlap at content level: >= 20% of states have P(F|Z=0) > 0 and P(F|Z=1) > 0.
  I3 LATE CI > 0 and |LATE - naive| >= 0.10.

    PYTHONPATH=code python -m beat.diag_iv --out runs/diag_iv
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


def ratio_ci(per_q, num_key, den_key, B=2000, seed=0):
    """per_q: {qid: [state dicts]}; ratio of sums over states; bootstrap over questions."""
    import numpy as np
    rng = np.random.default_rng(seed)
    qs = list(per_q)
    def est(sample):
        n = sum(s[num_key] for q in sample for s in per_q[q]); d = sum(s[den_key] for q in sample for s in per_q[q])
        return n / d if d else float("nan")
    point = est(qs)
    bs = [est([qs[i] for i in rng.integers(0, len(qs), len(qs))]) for _ in range(B)]
    bs = [b for b in bs if b == b]
    return point, (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))) if bs else (None, None)


def mean_ci(per_q, key, B=2000, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    qs = [q for q in per_q if any(s.get(key) is not None for s in per_q[q])]
    def est(sample):
        v = [s[key] for q in sample for s in per_q[q] if s.get(key) is not None]
        return float(np.mean(v)) if v else float("nan")
    point = est(qs)
    bs = [est([qs[i] for i in rng.integers(0, len(qs), len(qs))]) for _ in range(B)]
    bs = [b for b in bs if b == b]
    return point, (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))) if bs else (None, None)


def selftest():
    per_q = {0: [{"rf": 0.2, "fs": 0.4}], 1: [{"rf": 0.1, "fs": 0.2}]}
    p, ci = ratio_ci(per_q, "rf", "fs", B=50)
    assert abs(p - 0.5) < 1e-12 and ci[0] <= 0.5 <= ci[1]
    m, _ = mean_ci({0: [{"x": 1.0}, {"x": None}], 1: [{"x": 0.0}]}, "x", B=20)
    assert abs(m - 0.5) < 1e-12
    print("SELFTEST_OK", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=640)
    ap.add_argument("--k", type=int, default=8, help="segments per arm per state")
    ap.add_argument("--seg", type=int, default=96, help="segment length (tokens)")
    ap.add_argument("--cuts", default="0.33,0.66")
    ap.add_argument("--fails-allfail", type=int, default=4)
    ap.add_argument("--fails-mixed", type=int, default=2)
    ap.add_argument("--max-states", type=int, default=900)
    ap.add_argument("--max-completion", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=73)
    ap.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest()
    if a.selftest:
        return

    import numpy as np
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from beat.diag_causal import answer_value, facts_of, nums
    from beat.loop import TEACHER_TEMPLATE
    from train.common import BASE_MODEL, GRPO, configure_tokenizer
    from train.data import load_track
    from train.models import eos_ids
    from train.rewards import score_all

    t0 = time.time()
    T = GRPO["temperature"]
    random.seed(a.seed); np.random.seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tok = configure_tokenizer(AutoTokenizer.from_pretrained(BASE_MODEL), "curated")
    stop_ids = sorted(set(eos_ids(tok, "curated")) | {tok.convert_tokens_to_ids("<|im_end|>")})
    NL = {i for i, t_ in enumerate(tok.convert_ids_to_tokens(list(range(len(tok))))) if t_ is not None and "Ċ" in t_}

    rows = []
    for r in load_track("curated").itertuples():
        msgs = [{"role": m["role"], "content": m["content"]} for m in r.prompt]
        q, ref, ans = r.extra_info["question"], r.target[0]["content"], str(r.extra_info["answer"]).strip()
        av = answer_value(ans)
        rows.append({"msgs": msgs, "q": q, "ref": ref, "gt": r.reward_model["ground_truth"], "av": abs(av) if av is not None else None,
                     "p": tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
                              add_special_tokens=False)["input_ids"],
                     "facts": facts_of(ref, q, ans, 12, mask=False)})
    rows = [x for x in rows if len(x["p"]) <= 1024 and any(f["sig"] for f in x["facts"])]
    Q = [rows[i] for i in np.random.permutation(len(rows))[:a.questions].tolist()]
    enc_t = lambda x, ref: tok(tok.apply_chat_template([x["msgs"][0], {"role": "user", "content": TEACHER_TEMPLATE.format(
        question=x["q"], reference=ref)}], tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]
    for qi, x in enumerate(Q):
        x["qi"] = qi
        x["tp"] = enc_t(x, x["ref"])
        other = Q[(qi + 1 + random.randrange(len(Q) - 1)) % len(Q)]
        x["pp"] = enc_t(x, other["ref"])      # placebo: another question's reference

    llm = LLM(model=BASE_MODEL, gpu_memory_utilization=0.85, max_model_len=4096 + a.max_completion, seed=a.seed,
              enable_prefix_caching=True)
    sp = lambda n, mx: SamplingParams(temperature=T, top_p=1.0, max_tokens=mx, stop_token_ids=stop_ids, n=n)

    # ---- stage 1: 8 plain samples
    o = llm.generate([{"prompt_token_ids": x["p"]} for x in Q], sp(8, a.max_completion), use_tqdm=False)
    rr = score_all([c.text for oo in o for c in oo.outputs], [x["gt"] for x, oo in zip(Q, o) for _ in oo.outputs], 8)
    k = 0
    for x, oo in zip(Q, o):
        x["S"] = []
        for c in oo.outputs:
            x["S"].append({"comp": [t for t in c.token_ids if t not in stop_ids], "r": float(rr[k])}); k += 1
        x["nc"] = int(sum(s["r"] for s in x["S"]))
    print(f"STAGE1 {time.time() - t0:.0f}s allfail {sum(x['nc'] == 0 for x in Q)} mixed1-2 {sum(1 <= x['nc'] <= 2 for x in Q)}",
          flush=True)

    # ---- states
    cuts = [float(c) for c in a.cuts.split(",")]
    states = []
    for x in Q:
        if x["nc"] == 0:
            kind, nf = "allfail", a.fails_allfail
        elif x["nc"] <= 2:
            kind, nf = "mixed12", a.fails_mixed
        else:
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
                if f is None:
                    continue
                states.append({"x": x, "kind": kind, "t": t, "tau": tau, "sig": [Fraction(v) for v in f["sig"]],
                               "fact": f["hint"]})
    random.shuffle(states)
    states = states[:a.max_states]
    print(f"STATES {time.time() - t0:.0f}s {len(states)} " + json.dumps({kd: sum(s['kind'] == kd for s in states)
                                                                      for kd in ('allfail', 'mixed12')}), flush=True)

    # ---- segments under each arm
    ARMS = ("z0", "z1", "zp")
    ctx = {"z0": lambda s: s["x"]["p"], "z1": lambda s: s["x"]["tp"], "zp": lambda s: s["x"]["pp"]}
    seg_jobs = [(si, arm) for si in range(len(states)) for arm in ARMS]
    so = llm.generate([{"prompt_token_ids": ctx[arm](states[si]) + states[si]["tau"][:states[si]["t"]]} for si, arm in seg_jobs],
                      sp(a.k, a.seg), use_tqdm=False)
    cont_jobs = []
    for (si, arm), oo in zip(seg_jobs, so):
        s = states[si]
        for c in oo.outputs:
            seg = list(c.token_ids)
            ended = any(t_ in stop_ids for t_ in seg) or c.finish_reason == "stop"
            seg = [t_ for t_ in seg if t_ not in stop_ids]
            st_ = tok.decode(seg); sn = nums(st_)
            F = int(any(v in sn for v in s["sig"]))
            ans_in = int("boxed" in st_ or (s["x"]["av"] is not None and s["x"]["av"] in sn))   # exclusion check: answer leaked
            cont_jobs.append({"si": si, "arm": arm, "seg": seg, "F": F, "ans": ans_in, "ended": ended})
    need = [j for j in cont_jobs if not j["ended"] and a.max_completion - states[j["si"]]["t"] - len(j["seg"]) > 0]
    co = llm.generate([{"prompt_token_ids": states[j["si"]]["x"]["p"] + states[j["si"]]["tau"][:states[j["si"]]["t"]] + j["seg"]}
                       for j in need],
                      [sp(1, a.max_completion - states[j["si"]]["t"] - len(j["seg"])) for j in need], use_tqdm=False)
    for j, oo in zip(need, co):
        j["cont"] = [t_ for t_ in oo.outputs[0].token_ids if t_ not in stop_ids]
    texts = [tok.decode(states[j["si"]]["tau"][:states[j["si"]]["t"]] + j["seg"] + j.get("cont", [])) for j in cont_jobs]
    ys = score_all(texts, [states[j["si"]]["x"]["gt"] for j in cont_jobs], 8)
    for j, y in zip(cont_jobs, ys):
        j["Y"] = float(y)
    print(f"GEN {time.time() - t0:.0f}s segments {len(cont_jobs)} continuations {len(need)}", flush=True)

    # ---- per-state cells
    cell = defaultdict(list)
    for j in cont_jobs:
        cell[(j["si"], j["arm"])].append(j)
    per = []
    for si, s in enumerate(states):
        d = {"q": s["x"]["qi"], "kind": s["kind"]}
        for arm in ARMS:
            c = cell[(si, arm)]
            d[f"F_{arm}"] = float(np.mean([j["F"] for j in c])); d[f"Y_{arm}"] = float(np.mean([j["Y"] for j in c]))
            d[f"A_{arm}"] = float(np.mean([j["ans"] for j in c]))
        d["fs"] = d["F_z1"] - d["F_z0"]; d["rf"] = d["Y_z1"] - d["Y_z0"]
        d["fs_p"] = d["F_zp"] - d["F_z0"]; d["rf_p"] = d["Y_zp"] - d["Y_z0"]
        z0 = cell[(si, "z0")]
        y1 = [j["Y"] for j in z0 if j["F"] == 1]; y0 = [j["Y"] for j in z0 if j["F"] == 0]
        d["naive"] = (np.mean(y1) - np.mean(y0)) if y1 and y0 else None
        allj = cell[(si, "z0")] + cell[(si, "z1")]
        y1 = [j["Y"] for j in allj if j["F"] == 1]; y0 = [j["Y"] for j in allj if j["F"] == 0]
        d["naive_pooled"] = (np.mean(y1) - np.mean(y0)) if y1 and y0 else None
        d["overlap"] = float(d["F_z0"] > 0 and d["F_z1"] > 0)
        per.append(d)

    def summarize(sel):
        per_q = defaultdict(list)
        for d in per:
            if sel(d):
                per_q[d["q"]].append(d)
        if not per_q:
            return None
        r = {"states": sum(len(v) for v in per_q.values()), "questions": len(per_q)}
        for key in ("A_z0", "A_z1", "A_zp", "F_z0", "F_z1", "F_zp", "Y_z0", "Y_z1", "Y_zp", "fs", "rf", "fs_p", "rf_p", "naive", "naive_pooled", "overlap"):
            m, ci = mean_ci(per_q, key)
            r[key] = {"mean": m, "ci": ci}
        late, ci = ratio_ci(per_q, "rf", "fs")
        r["LATE"] = {"mean": late, "ci": ci}
        return r

    res = {"config": vars(a), "all": summarize(lambda d: True), "allfail": summarize(lambda d: d["kind"] == "allfail"),
           "mixed12": summarize(lambda d: d["kind"] == "mixed12")}
    A = res["all"]
    fs, fsp, late, nv = A["fs"], A["fs_p"], A["LATE"], A["naive"]
    res["I1_PASS"] = bool(fs["mean"] >= 0.10 and fs["ci"][0] > 0 and
                          ((fsp["ci"][0] <= 0 <= fsp["ci"][1]) or abs(fsp["mean"]) < 0.03))
    res["I2_PASS"] = bool(A["overlap"]["mean"] >= 0.20)
    res["I3_PASS"] = bool(late["ci"][0] is not None and late["ci"][0] > 0 and nv["mean"] == nv["mean"]
                          and abs(late["mean"] - nv["mean"]) >= 0.10)
    res["time"] = round(time.time() - t0, 1)
    (out / "results.json").write_text(json.dumps(res, indent=2))
    (out / "per_state.json").write_text(json.dumps(per))
    ex = []
    for si in range(min(12, len(states))):
        s = states[si]
        z1 = [j for j in cell[(si, "z1")] if j["F"] == 1][:1]
        ex.append({"fact": s["fact"], "prefix_tail": tok.decode(s["tau"][max(0, s["t"] - 60):s["t"]]),
                   "seg_z0": tok.decode(cell[(si, "z0")][0]["seg"]), "seg_z1_withF": tok.decode(z1[0]["seg"]) if z1 else None})
    (out / "examples.json").write_text(json.dumps(ex, ensure_ascii=False, indent=1))
    short = lambda r: None if r is None else {k_: (round(v["mean"], 4) if isinstance(v, dict) and v["mean"] is not None
                                                  else v) for k_, v in r.items()}
    print("SUMMARY " + json.dumps({k_: short(res[k_]) for k_ in ("all", "allfail", "mixed12")}), flush=True)
    print("CI " + json.dumps({k_: A[k_]["ci"] for k_ in ("fs", "fs_p", "rf", "rf_p", "LATE", "naive", "overlap")}), flush=True)
    print("RESULT " + json.dumps({k_: res[k_] for k_ in ("I1_PASS", "I2_PASS", "I3_PASS")}), flush=True)
    print(f"DONE {out}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
