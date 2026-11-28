"""Pre-screen D1-D4 for "sparse mechanism shift + interventional discovery + counterfactual twins" (no training).

Causal model. The student (Qwen2.5-3B base) is an SCM: tokens = f(context, noise U). Teacher facts F1..FK = the first K steps of
Luna's y* (leak-filtered: no step that contains the final answer, none after it). An intervention do(S) puts the facts in S (derived
numbers masked, so the student must still compute them) into the prompt as a hint. Every fact j that introduces new numbers has a
signature (those numbers, canonicalised); "the student produced Fj" = a signature number appears in the student's text.

Stage 1  8 plain samples on Q questions -> hard (<= 2/8 correct) and mixed (3..7/8) questions.
Stage 2  (vLLM, coupled: same per-request seed s for every set, so twins share the sampling noise)
         sets: {} , {i}, {i,j}, all, all-minus-{i}, placebo (a masked fact of another question, matched length).
D1  sparse shift.   PS(S) = P(success | do(S), base failed under the same noise). A set is "sufficient" when its success count
    on the failed seeds is significant vs the question's own plain success rate (one-sided binomial, Benjamini-Hochberg q=0.10 over
    all (question, set) tests). Placebo sets are scored with the same p-value cutoff = false-positive anchor.
    Report: min sufficient set size, # necessary facts given the rest, where the sufficient single fact sits in y*.
D2  graph.          Total-effect edge i->j (i<j): P(prod Fj | do({i})) - P(prod Fj | do({})), coupled seeds, exact McNemar.
    Placebo edges (do(placebo) -> prod Fj) = false-positive anchor.
D4  confounding.    obs_j = P(R | Fj produced) - P(R | not), plain samples;  int_j = P(R | do({j})) - P(R | do({})).
    Spearman across (question, fact) + split-half reliabilities (noise ceiling).
Stage 3  (HF, exact Gumbel-max SCM: noise u(seed, student-token position) so a note inserted mid-trajectory does not shift the noise)
D3  minimal counterfactuals.  For hard questions whose best single fact f* is sufficient: regenerate base trajectories tau; for each
    failed tau insert "(Note: f*)" at cut t in {0, ~25%, ~50%, ~75%} of tau (newline boundaries) and continue with the SAME noise.
    Anchors: null insert (must reproduce tau exactly), placebo note. Report flip rate per cut, shared share of tau for flipped
    pairs, how often the counterfactual continuation mentions the note.

    PYTHONPATH=code python -m beat.diag_causal --out runs/diag_causal
"""
from __future__ import annotations

import argparse
import gzip
import itertools
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
MATH = re.compile(r"\\\((.*?)\\\)|\\\[(.*?)\\\]|\$\$(.*?)\$\$|\$(.*?)\$", re.S)
FRAC = re.compile(r"\\[dt]?frac\s*\{\s*(-?\d+(?:\.\d+)?)\s*\}\s*\{\s*(-?\d+(?:\.\d+)?)\s*\}")
NUM = re.compile(r"(?<![\d.])\d+(?:\.\d+)?(?:/\d+)?")
NOTE_WORDS = re.compile(r"\b(note|hint)\b", re.I)


# ---------------------------------------------------------------- text helpers (pure, unit-tested in main --selftest)
def nums(text: str) -> set:
    """Canonical numbers in a text: 1,000 -> 1000, \\frac{3}{4} -> 3/4, 0.50 -> 1/2."""
    t = FRAC.sub(lambda m: f"{m.group(1)}/{m.group(2)}", text)
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)
    out = set()
    for m in NUM.finditer(t):
        try:
            out.add(Fraction(m.group()))
        except (ValueError, ZeroDivisionError):
            pass
    return out


def answer_value(ans: str):
    s = ans.strip().strip("$").replace("\\!", "").replace(" ", "")
    s = re.sub(r"^\\boxed\{(.*)\}$", r"\1", s)
    s = FRAC.sub(lambda m: f"{m.group(1)}/{m.group(2)}", s)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def steps_of(ref: str):
    """Split y* into steps at sentence ends / newlines outside math. Returns [(text, end_char)]."""
    spans = [(m.start(), m.end()) for m in MATH.finditer(ref)]
    inside = lambda c: any(s < c < e for s, e in spans)
    cuts = [m.end() for m in re.finditer(r"(?<=[.;:])\s+|\n+", ref) if not inside(m.start())]
    b = [0] + [c for c in cuts if 0 < c < len(ref)] + [len(ref)]
    merged = []
    for u, v in zip(b, b[1:]):
        s = ref[u:v].strip()
        if len(s) < 8:
            if merged:
                merged[-1] = (merged[-1][0], v)
            continue
        if merged and (s.startswith("\\[") or s.startswith("$$")):
            merged[-1] = (merged[-1][0] + " " + s, v)
        else:
            merged.append((s, v))
    return merged


def abstract(step: str, q: str) -> str:
    given = set(re.findall(r"\d+", q))
    return re.sub(r"\d+", lambda m: m.group() if m.group() in given else "_", step)


def facts_of(ref: str, q: str, ans: str, kmax: int, mask: bool = True):
    """Leak-filtered teacher facts: [{hint (masked), sig (new numbers), end_share (share of y* after the step)}]."""
    av = answer_value(ans)
    av = abs(av) if av is not None else None  # nums() drops signs
    qn = nums(q)
    small = {Fraction(i) for i in range(11)}
    seen = set(qn)
    out = []
    for text, end in steps_of(ref):
        sn = nums(text)
        if "boxed" in text or (av is not None and av in sn) or (av is None and len(ans.strip()) >= 2 and ans.strip() in text):
            break  # the answer appears: this step and everything after it are excluded
        sig = sn - seen - small
        seen |= sn
        out.append({"hint": abstract(text, q) if mask else text, "sig": sorted(str(v) for v in sig), "end_share": round(1 - end / max(1, len(ref)), 3)})
        if len(out) >= kmax:
            break
    return out


def produced(text_nums: set, sig: list) -> bool:
    return bool(sig) and any(Fraction(v) in text_nums for v in sig)


def common_prefix(u, v):
    n = 0
    for p, q in zip(u, v):
        if p != q:
            break
        n += 1
    return n


def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k), X ~ Bin(n, p)."""
    if k <= 0:
        return 1.0
    return float(sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def bh_cutoff(pvals, q: float) -> float:
    """Benjamini-Hochberg: largest p_(i) with p_(i) <= q*i/m; returns -1 if none."""
    ps = sorted(pvals)
    m = len(ps)
    cut = -1.0
    for i, p in enumerate(ps, 1):
        if p <= q * i / m:
            cut = p
    return cut


def mcnemar_p(n10: int, n01: int) -> float:
    """One-sided exact McNemar: P(X >= n10), X ~ Bin(n10+n01, 1/2)."""
    return binom_sf(n10, n10 + n01, 0.5) if n10 + n01 else 1.0


def spearman(x, y):
    import numpy as np
    if len(x) < 3:
        return None
    rx = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)
    ry = np.argsort(np.argsort(y, kind="stable"), kind="stable").astype(float)
    # average ranks for ties
    def avg(v, r):
        v = np.asarray(v); r = r.copy()
        for val in np.unique(v):
            idx = np.where(v == val)[0]
            r[idx] = r[idx].mean()
        return r
    rx, ry = avg(x, rx), avg(y, ry)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def selftest():
    assert nums("so 1,000 + \\frac{3}{4} = 0.75 and 12") == {Fraction(1000), Fraction(3, 4), Fraction(12)}
    assert answer_value("\\frac{3}{4}") == Fraction(3, 4) and answer_value("1,024") == 1024 and answer_value("\\sqrt{2}") is None
    f = facts_of("Let t be the common value. Then A = t - 3 and D = 3t. Summing, 6t = 72 so t = 12. The product is 1440. \\boxed{1440}",
                 "sum is 64, A+3=B-3", "1440", 5)
    assert [x["hint"] for x in f] == ["Let t be the common value.", "Then A = t - 3 and D = 3t.", "Summing, _t = _ so t = _."], f
    assert f[2]["sig"] == ["12", "72"], f
    assert produced({Fraction(12)}, ["12", "72"]) and not produced({Fraction(5)}, ["12"]) and not produced({Fraction(12)}, [])
    assert bh_cutoff([0.001, 0.01, 0.5], 0.1) == 0.01 and bh_cutoff([0.5, 0.9], 0.1) == -1.0
    assert abs(binom_sf(2, 2, 0.5) - 0.25) < 1e-12 and binom_sf(0, 5, 0.3) == 1.0
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-9
    print("SELFTEST_OK", flush=True)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=384)
    ap.add_argument("--max-hard", type=int, default=120)
    ap.add_argument("--max-mixed", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--max-facts", type=int, default=5)
    ap.add_argument("--max-completion", type=int, default=1536)
    ap.add_argument("--fdr", type=float, default=0.10)
    ap.add_argument("--mask", type=int, default=1, help="1: derived numbers in hints masked (do(move)); 0: kept (do(fact))")
    ap.add_argument("--d3-questions", type=int, default=40)
    ap.add_argument("--d3-seeds", type=int, default=6)
    ap.add_argument("--d3-batch", type=int, default=48)
    ap.add_argument("--d3-dtype", default="float32")
    ap.add_argument("--d3-force", type=int, default=0, help="smoke only: use the best raw-PS single fact when none is significant")
    ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest()
    if a.selftest:
        return

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
    T = GRPO["temperature"]
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tok = configure_tokenizer(AutoTokenizer.from_pretrained(BASE_MODEL), "curated")
    stop_ids = sorted(set(eos_ids(tok, "curated")) | {tok.convert_tokens_to_ids("<|im_end|>")})

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
    hinted = lambda x, hints: x["user"] if not hints else x["body"] + "\n\nHint: " + " ".join(hints) + SUFFIX
    for x in Q:
        x["pp"] = chat(x, x["user"])
        x["facts"] = facts_of(x["ref"], x["q"], x["ans"], a.max_facts, bool(a.mask))

    llm = LLM(model=BASE_MODEL, gpu_memory_utilization=0.5, max_model_len=4096, seed=a.seed, enable_sleep_mode=True)
    sp = lambda n, seed=None: SamplingParams(temperature=T, top_p=1.0, max_tokens=a.max_completion,
                                             stop_token_ids=stop_ids, n=n, seed=seed)

    # ---- stage 1
    o1 = llm.generate([{"prompt_token_ids": x["pp"]} for x in Q], sp(8), use_tqdm=False)
    txt1 = [c.text for o in o1 for c in o.outputs]
    r1 = score_all(txt1, [x["gt"] for x, o in zip(Q, o1) for _ in o.outputs], 8)
    k = 0
    for x, o in zip(Q, o1):
        x["s1"] = [{"r": float(r1[k + j]), "nums": nums(c.text)} for j, c in enumerate(o.outputs)]
        k += len(o.outputs)
    ok = lambda x: len(x["facts"]) >= 1
    H = [x for x in Q if ok(x) and sum(s["r"] for s in x["s1"]) <= 2][:a.max_hard]
    M = [x for x in Q if ok(x) and 3 <= sum(s["r"] for s in x["s1"]) <= 7][:a.max_mixed]
    for x in H:
        x["kind"] = "hard"
    for x in M:
        x["kind"] = "mixed"
    print(f"STAGE1 {time.time() - t0:.0f}s hard {len(H)} mixed {len(M)} "
          f"facts_per_q {np.mean([len(x['facts']) for x in H + M]):.2f} "
          f"with_sig {np.mean([sum(bool(f['sig']) for f in x['facts']) for x in H + M]):.2f}", flush=True)

    # placebo: a masked fact of another question, closest length
    pool = [(len(f["hint"]), id(x), f["hint"]) for x in Q for f in x["facts"]]
    for x in H + M:
        L = int(np.mean([len(f["hint"]) for f in x["facts"]]))
        cands = sorted((abs(l - L), h) for l, xid, h in pool if xid != id(x))
        x["placebo"] = cands[0][1] if cands else "Consider the problem carefully."

    # ---- stage 2: coupled interventions
    jobs = []
    for qi, x in enumerate(H + M):
        K = len(x["facts"])
        full = tuple(range(K))
        sets = [()] + [(i,) for i in range(K)] + list(itertools.combinations(range(K), 2))
        if K > 2:
            sets += [full] + [tuple(j for j in full if j != i) for i in range(K)]
        sets = list(dict.fromkeys(sets)) + ["placebo"]
        x["sets"] = sets
        for S in sets:
            hints = [x["placebo"]] if S == "placebo" else [x["facts"][i]["hint"] for i in S]
            pid = chat(x, hinted(x, hints))
            for s in range(a.seeds):
                jobs.append((qi, S, s, pid))
    XS = H + M
    outs = llm.generate([{"prompt_token_ids": p} for *_, p in jobs], [sp(1, 1000 + s) for _, _, s, _ in jobs], use_tqdm=False)
    rr = score_all([o.outputs[0].text for o in outs], [XS[qi]["gt"] for qi, *_ in jobs], 8)
    raw = gzip.open(out / "stage2.jsonl.gz", "wt")
    for (qi, S, s, _), o, r in zip(jobs, outs, rr):
        c = o.outputs[0]
        rec = {"r": float(r), "trunc": c.finish_reason == "length", "nums": nums(c.text), "n_tok": len(c.token_ids)}
        XS[qi].setdefault("tw", {})[(S, s)] = rec
        raw.write(json.dumps({"qi": qi, "kind": XS[qi]["kind"], "set": S if S == "placebo" else list(S), "seed": s, "r": rec["r"],
                              "trunc": rec["trunc"], "n_tok": rec["n_tok"],
                              "prod": [produced(rec["nums"], f["sig"]) for f in XS[qi]["facts"]]}) + "\n")
    raw.close()
    print(f"STAGE2 {time.time() - t0:.0f}s gens {len(jobs)}", flush=True)
    res = {"config": vars(a), "questions": len(Q), "hard": len(H), "mixed": len(M), "seeds": a.seeds}

    # ---- D1 sparse mechanism shift (hard questions)
    tests = []  # (x, S, pval, ps, nF)
    d1_skip = Counter()
    for x in H:
        tw = x["tw"]
        F = [s for s in range(a.seeds) if tw[((), s)]["r"] == 0]
        if not F:
            d1_skip["base_ok_all_seeds"] += 1; x["F"] = []; continue
        x["F"] = F
        plain = [s["r"] for s in x["s1"]] + [tw[((), s)]["r"] for s in range(a.seeds)]
        p0 = (sum(plain) + 1) / (len(plain) + 2)  # smoothed plain success rate = null for a decoupled, irrelevant hint
        x["p0"] = p0
        for S in x["sets"]:
            if S == ():
                continue
            kS = int(sum(tw[(S, s)]["r"] for s in F))
            tests.append((x, S, binom_sf(kS, len(F), p0), kS / len(F), len(F)))
    real = [t for t in tests if t[1] != "placebo"]
    cut = bh_cutoff([t[2] for t in real], a.fdr)
    sig = lambda t: t[2] <= cut and t[3] > 0
    for x in H:
        x["suff"] = {}
    for t in tests:
        t[0]["suff"][t[1]] = (sig(t), t[3])
    plc = [t for t in tests if t[1] == "placebo"]
    d1 = {"bh_p_cutoff": cut, "tests_real": len(real), "sufficient_real": sum(map(sig, real)),
          "placebo_tests": len(plc), "placebo_sufficient": sum(map(sig, plc)),
          "placebo_PS_mean": float(np.mean([t[3] for t in plc])) if plc else None,
          "skipped": dict(d1_skip)}
    min_size, min_size3, n_nec, pos_single, last_single, best_single_ps = Counter(), Counter(), [], [], [], []
    for x in H:
        if not x["F"]:
            continue
        K = len(x["facts"]); su = x["suff"]
        sizes = sorted(len(S) for S, (ok_, _) in su.items() if S != "placebo" and ok_)
        min_size[str(sizes[0]) if sizes else "none"] += 1
        if K >= 3:
            min_size3[str(sizes[0]) if sizes else "none"] += 1
        singles = [(su[(i,)][1], i) for i in range(K) if su[(i,)][0]]
        if singles:
            ps, i = max(singles)
            best_single_ps.append(ps); pos_single.append(x["facts"][i]["end_share"]); last_single.append(i == K - 1)
            x["fstar"] = i
        full = tuple(range(K))
        if K > 2 and su[full][0]:
            n_nec.append(sum(1 for i in range(K) if su[full][1] - su[tuple(j for j in full if j != i)][1] >= 0.25))
    solv = sum(1 for x in H if x["F"] and any(ok_ for S, (ok_, _) in x["suff"].items() if S != "placebo"))
    d1.update({"hard_with_failed_seeds": sum(1 for x in H if x["F"]), "solvable_by_some_set": solv,
               "min_sufficient_size": dict(min_size),
               "frac_min_size_le2_among_solvable": (sum(v for kk, v in min_size.items() if kk in ("1", "2")) / solv) if solv else None,
               "min_sufficient_size_K3plus": dict(min_size3),
               "frac_min_size_le2_among_solvable_K3plus": (sum(v for kk, v in min_size3.items() if kk in ("1", "2"))
                                                           / max(1, sum(v for kk, v in min_size3.items() if kk != "none"))),
               "necessary_facts_given_rest_hist": dict(Counter(n_nec)),
               "sufficient_single_yref_share_after_median": float(np.median(pos_single)) if pos_single else None,
               "sufficient_single_is_last_kept_fact_frac": float(np.mean(last_single)) if last_single else None,
               "best_single_PS_mean": float(np.mean(best_single_ps)) if best_single_ps else None})
    d1["PASS"] = bool(sum(v for kk, v in min_size3.items() if kk != "none") >= 10 and d1["frac_min_size_le2_among_solvable_K3plus"] >= 0.5
                      and d1["placebo_sufficient"] <= max(1, 0.1 * max(1, d1["placebo_tests"])))
    res["D1"] = d1
    print("D1 " + json.dumps(d1), flush=True)

    # ---- D2 interventional graph (hard + mixed)
    e_tests, p_tests, self_c = [], [], []
    for x in XS:
        tw, K = x["tw"], len(x["facts"])
        prod = lambda S, s, j: produced(tw[(S, s)]["nums"], x["facts"][j]["sig"])
        for j in range(K):
            if not x["facts"][j]["sig"]:
                continue
            base = [prod((), s, j) for s in range(a.seeds)]
            for i in range(j):
                d = [prod((i,), s, j) for s in range(a.seeds)]
                n10 = sum(u and not v for u, v in zip(d, base)); n01 = sum(v and not u for u, v in zip(d, base))
                e_tests.append((x, i, j, (sum(d) - sum(base)) / a.seeds, mcnemar_p(n10, n01)))
            d = [prod("placebo", s, j) for s in range(a.seeds)]
            n10 = sum(u and not v for u, v in zip(d, base)); n01 = sum(v and not u for u, v in zip(d, base))
            p_tests.append((x, j, (sum(d) - sum(base)) / a.seeds, mcnemar_p(n10, n01)))
            dj = [prod((j,), s, j) for s in range(a.seeds)]
            if a.mask:  # unmasked hint j contains its own numbers
                self_c.append((sum(dj) - sum(base)) / a.seeds)
    ecut = bh_cutoff([t[4] for t in e_tests], a.fdr) if e_tests else -1
    edge = lambda t: t[4] <= ecut and t[3] >= 0.25
    q_with_edge = {id(t[0]) for t in e_tests if edge(t)}
    q_testable = {id(t[0]) for t in e_tests}
    d2 = {"edge_tests": len(e_tests), "bh_p_cutoff": ecut, "edges": sum(map(edge, e_tests)),
          "edge_frac": (sum(map(edge, e_tests)) / len(e_tests)) if e_tests else None,
          "questions_testable": len(q_testable), "questions_with_edge": len(q_with_edge),
          "placebo_tests": len(p_tests), "placebo_edges": sum(1 for t in p_tests if t[3] <= ecut and t[2] >= 0.25),
          "mean_effect_edges_real": float(np.mean([t[3] for t in e_tests])) if e_tests else None,
          "mean_effect_placebo": float(np.mean([t[2] for t in p_tests])) if p_tests else None,
          "self_completion_mean": float(np.mean(self_c)) if self_c else None,
          "by_kind": {kd: {"tests": sum(1 for t in e_tests if t[0]["kind"] == kd),
                           "edges": sum(1 for t in e_tests if t[0]["kind"] == kd and edge(t))} for kd in ("hard", "mixed")}}
    d2["PASS"] = bool(q_testable and len(q_with_edge) / len(q_testable) >= 0.3
                      and d2["placebo_edges"] <= max(1, 0.1 * max(1, len(p_tests))))
    res["D2"] = d2
    print("D2 " + json.dumps(d2), flush=True)

    # ---- D4 observational vs interventional
    pairs = []
    for x in XS:
        tw, K = x["tw"], len(x["facts"])
        plainA = [(s["r"], s["nums"]) for s in x["s1"]]
        plainB = [(tw[((), s)]["r"], tw[((), s)]["nums"]) for s in range(a.seeds)]

        def obs(sample, sigj):
            pr = [r for r, n in sample if produced(n, sigj)]; npr = [r for r, n in sample if not produced(n, sigj)]
            return (np.mean(pr) - np.mean(npr)) if len(pr) >= 2 and len(npr) >= 2 else None

        for j in range(K):
            sj = x["facts"][j]["sig"]
            if not sj:
                continue
            o = obs(plainA + plainB, sj)
            if o is None:
                continue
            ints = [tw[((j,), s)]["r"] - tw[((), s)]["r"] for s in range(a.seeds)]
            h = a.seeds // 2
            pairs.append({"kind": x["kind"], "obs": float(o), "int": float(np.mean(ints)),
                          "obsA": obs(plainA, sj), "obsB": obs(plainB, sj),
                          "intA": float(np.mean(ints[:h])), "intB": float(np.mean(ints[h:]))})
    d4 = {"pairs": len(pairs)}
    if pairs:
        ob = [p["obs"] for p in pairs]; it = [p["int"] for p in pairs]
        half = [p for p in pairs if p["obsA"] is not None and p["obsB"] is not None]
        r_obs = spearman([p["obsA"] for p in half], [p["obsB"] for p in half]) if len(half) >= 3 else None
        r_int = spearman([p["intA"] for p in pairs], [p["intB"] for p in pairs])
        pos = [p for p in pairs if p["obs"] >= 0.25]
        d4.update({"spearman_obs_int": spearman(ob, it), "reliability_obs_splithalf": r_obs, "reliability_int_splithalf": r_int,
                   "noise_ceiling": (math.sqrt(max(0, r_obs) * max(0, r_int)) if r_obs is not None and r_int is not None else None),
                   "obs_mean": float(np.mean(ob)), "int_mean": float(np.mean(it)),
                   "obs_positive": len(pos), "spurious_among_obs_positive": sum(1 for p in pos if p["int"] <= 0),
                   "hidden_causes": sum(1 for p in pairs if p["obs"] <= 0 and p["int"] >= 0.25)})
        d4["spurious_frac"] = d4["spurious_among_obs_positive"] / len(pos) if pos else None
        ceil, rho = d4["noise_ceiling"], d4["spearman_obs_int"]
        # low agreement only means confounding if the two measures are each reliable; otherwise it is just noise
        d4["PASS"] = None if (ceil is None or ceil < 0.3 or rho is None) else bool(rho < 0.5 * ceil)
    res["D4"] = d4
    print("D4 " + json.dumps(d4), flush=True)
    (out / "partial.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))

    # ---- stage 3 / D3: exact Gumbel-max SCM in HF
    if a.d3_force:
        for x in H:
            if x.get("F") and "fstar" not in x:
                x["fstar"] = max(range(len(x["facts"])), key=lambda i: x["suff"][(i,)][1])
    cand = [x for x in H if "fstar" in x][:a.d3_questions]
    if not cand:
        res["D3"] = {"questions": 0, "PASS": False, "note": "no hard question with a significant single fact"}
        print("D3 " + json.dumps(res["D3"]), flush=True)
        res["time"] = round(time.time() - t0, 1)
        (out / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"DONE {out}", flush=True)
        return
    llm.sleep(level=2); del llm; torch.cuda.empty_cache()
    from transformers import AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
    dtype = getattr(torch, a.d3_dtype)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype, attn_implementation="sdpa").cuda().eval()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else stop_ids[0]

    class Gumbel(LogitsProcessor):
        """argmax(logits/T + G) with G = Gumbel noise indexed by (seed, student-token position): an exact sampler whose noise is
        shared by all interventions on the same unit; inserted note tokens do not consume noise."""
        def __init__(self, P, seeds, offsets):
            self.P, self.seeds, self.offsets = P, seeds, offsets
            self.gen = torch.Generator(device="cuda")

        def __call__(self, input_ids, scores):
            step = input_ids.shape[1] - self.P
            s = scores.float() / T
            for b, (sd, off) in enumerate(zip(self.seeds, self.offsets)):
                self.gen.manual_seed(int(sd) * 1_000_003 + off + step + 17)
                u = torch.rand(s.shape[-1], generator=self.gen, device="cuda").clamp_(1e-20, 1 - 1e-7)
                s[b] += -torch.log(-torch.log(u))
            return s

    def run(batch):
        """batch: list of dict(ctx ids, seed, offset, budget). Returns completion ids (stop token removed) and truncated flag."""
        res_ = [None] * len(batch)
        order = sorted(range(len(batch)), key=lambda i: -batch[i]["budget"])
        for c0 in range(0, len(order), a.d3_batch):
            idx = order[c0:c0 + a.d3_batch]
            P = max(len(batch[i]["ctx"]) for i in idx)
            ids = torch.full((len(idx), P), pad_id, dtype=torch.long)
            att = torch.zeros((len(idx), P), dtype=torch.long)
            for r_, i in enumerate(idx):
                c = batch[i]["ctx"]; ids[r_, P - len(c):] = torch.tensor(c); att[r_, P - len(c):] = 1
            budget = max(batch[i]["budget"] for i in idx)
            with torch.no_grad():
                g = model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(), max_new_tokens=budget, do_sample=False,
                                   temperature=None, top_p=None, top_k=None, repetition_penalty=1.0, num_beams=1,
                                   logits_processor=LogitsProcessorList([Gumbel(P, [batch[i]["seed"] for i in idx],
                                                                                 [batch[i]["offset"] for i in idx])]),
                                   eos_token_id=stop_ids, pad_token_id=pad_id)
            for r_, i in enumerate(idx):
                seq = g[r_, P:].tolist()
                stop_at = next((k_ for k_, t_ in enumerate(seq) if t_ in stop_ids), None)
                comp = seq[:stop_at] if stop_at is not None else seq
                comp = comp[:batch[i]["budget"]]
                res_[i] = (comp, stop_at is None or stop_at >= batch[i]["budget"])
        return res_

    newline_tok = [i for i, t_ in enumerate(tok.convert_ids_to_tokens(list(range(len(tok))))) if t_ is not None and "Ċ" in t_]
    NL = set(newline_tok)
    t3 = time.time()
    base_jobs = [{"x": x, "seed": s, "ctx": x["pp"], "offset": 0, "budget": a.max_completion} for x in cand for s in range(a.d3_seeds)]
    bo = run(base_jobs)
    br = score_all([tok.decode(c) for c, _ in bo], [j["x"]["gt"] for j in base_jobs], 8)
    taus = [(j, c, tr) for j, (c, tr), r in zip(base_jobs, bo, br) if r == 0 and len(c) >= 20]
    print(f"STAGE3_BASE {time.time() - t0:.0f}s base {len(base_jobs)} failed {len(taus)} base_success "
          f"{float(np.mean(br)):.3f}", flush=True)

    note = lambda h: tok("\n(Note: " + h + ")\n\n", add_special_tokens=False)["input_ids"]
    cj = []
    for ti, (j, tau, _) in enumerate(taus):
        x = j["x"]
        bounds = [k_ + 1 for k_, t_ in enumerate(tau) if t_ in NL and 5 <= k_ + 1 <= len(tau) - 5]
        cuts = [0]
        for fr in (0.25, 0.5, 0.75):
            if bounds:
                b = min(bounds, key=lambda v: abs(v - fr * len(tau)))
                if b not in cuts:
                    cuts.append(b)
        fh = x["facts"][x["fstar"]]["hint"]
        mid = cuts[len(cuts) // 2] if len(cuts) > 1 else None
        for ci, t_ in enumerate(cuts):
            for kind, h in (("fact", fh), ("placebo", x["placebo"])):
                cj.append({"ti": ti, "kind": kind, "cut_i": ci, "t": t_, "seed": j["seed"], "offset": t_,
                           "ctx": x["pp"] + tau[:t_] + note(h), "budget": a.max_completion - t_})
        if mid:
            cj.append({"ti": ti, "kind": "null", "cut_i": len(cuts) // 2, "t": mid, "seed": j["seed"], "offset": mid,
                       "ctx": x["pp"] + tau[:mid], "budget": a.max_completion - mid})
        cj.append({"ti": ti, "kind": "prompt_fact", "cut_i": -1, "t": 0, "seed": j["seed"], "offset": 0,
                   "ctx": chat(x, hinted(x, [fh])), "budget": a.max_completion})
    co = run(cj)
    texts = [tok.decode(taus[c["ti"]][1][:c["t"]] + comp) for c, (comp, _) in zip(cj, co)]
    cr = score_all(texts, [taus[c["ti"]][0]["x"]["gt"] for c in cj], 8)
    print(f"STAGE3_CF {time.time() - t0:.0f}s continuations {len(cj)} ({time.time() - t3:.0f}s for stage 3)", flush=True)

    recs = []
    for c, (comp, trunc), r in zip(cj, co, cr):
        tau = taus[c["ti"]][1]
        d = common_prefix(tau[c["t"]:], comp)
        recs.append({"kind": c["kind"], "cut_i": c["cut_i"], "t": c["t"], "len_tau": len(tau), "r": float(r), "trunc": trunc,
                     "div_after_cut": d, "shared_share": (c["t"] + d) / len(tau), "identical": comp == tau[c["t"]:],
                     "mentions_note": bool(NOTE_WORDS.search(tok.decode(comp[d:]))) if r == 1 else None, "ti": c["ti"]})
    with gzip.open(out / "stage3.jsonl.gz", "wt") as f:
        for c, (comp, _), rc in zip(cj, co, recs):
            f.write(json.dumps({**rc, "q": taus[c["ti"]][0]["x"]["q"][:300], "tau": tok.decode(taus[c["ti"]][1]),
                                "comp": tok.decode(comp)}) + "\n")
    d3 = {"questions": len(cand), "failed_tau": len(taus), "base_success": float(np.mean(br)), "continuations": len(cj)}
    nul = [r_ for r_ in recs if r_["kind"] == "null"]
    d3["anchor_null_identical_frac"] = float(np.mean([r_["identical"] for r_ in nul])) if nul else None
    d3["anchor_null_div_share_median"] = float(np.median([r_["shared_share"] for r_ in nul])) if nul else None
    per_cut = {}
    for ci in sorted({r_["cut_i"] for r_ in recs if r_["kind"] in ("fact", "placebo")}):
        f_ = [r_ for r_ in recs if r_["kind"] == "fact" and r_["cut_i"] == ci]
        p_ = [r_ for r_ in recs if r_["kind"] == "placebo" and r_["cut_i"] == ci]
        fl = [r_ for r_ in f_ if r_["r"] == 1]
        per_cut[str(ci)] = {"n": len(f_), "cut_share_median": float(np.median([r_["t"] / r_["len_tau"] for r_ in f_])),
                            "flip_fact": float(np.mean([r_["r"] for r_ in f_])), "flip_placebo": float(np.mean([r_["r"] for r_ in p_])),
                            "flipped_shared_share_median": float(np.median([r_["shared_share"] for r_ in fl])) if fl else None,
                            "flipped_div_after_cut_median": float(np.median([r_["div_after_cut"] for r_ in fl])) if fl else None,
                            "flipped_mentions_note_frac": float(np.mean([r_["mentions_note"] for r_ in fl])) if fl else None}
    d3["per_cut"] = per_cut
    pf = [r_ for r_ in recs if r_["kind"] == "prompt_fact"]
    pfl = [r_ for r_ in pf if r_["r"] == 1]
    d3["prompt_hint"] = {"flip": float(np.mean([r_["r"] for r_ in pf])) if pf else None,
                         "flipped_div_tokens_median": float(np.median([r_["div_after_cut"] for r_ in pfl])) if pfl else None}
    good = [v for kk, v in per_cut.items() if kk != "0" and v["flipped_shared_share_median"] is not None
            and v["flip_fact"] - v["flip_placebo"] >= 0.15 and v["flipped_shared_share_median"] >= 0.25]
    d3["PASS"] = bool(good and (d3["anchor_null_identical_frac"] or 0) >= 0.95)
    res["D3"] = d3
    print("D3 " + json.dumps(d3), flush=True)
    res["time"] = round(time.time() - t0, 1)
    (out / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("RESULT " + json.dumps({kk: res[kk].get("PASS") for kk in ("D1", "D2", "D3", "D4")}), flush=True)
    print(f"DONE {out}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
