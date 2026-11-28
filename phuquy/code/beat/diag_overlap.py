"""Pre-screen R06-5: overlap-weighted counterfactual policy gradient inside GRPO (no training).

Causal reading of the GRPO estimator sum_t A_t grad log pi(a_t|s_t): A_t estimates the effect of the action on reward. With samples
from the student only, actions the student never takes have no data (positivity / overlap violation) -> all-fail groups give no
gradient. At a state s_t of a failed student trajectory tau we compare two actions:
  a   = the student's own next line (prob pi(a|s), q(a|s))
  a'  = the teacher's next missing step: earliest step of y* none of whose new numbers appear in tau[:t] (answer step excluded),
        written as the student's own text.
pi = student (plain prompt + tau[:t]); q = same model with y* in the context (TEACHER_TEMPLATE + tau[:t]). Both computable.
Outcomes are COUNTERFACTUAL: the continuation after the action uses the same Gumbel noise as tau (noise indexed by the student-token
position of tau after its own action), so Y(a) = 0 is tau itself and Y(a') differs only through the action.
Overlap weights (Li, Morgan & Zaslavsky 2018) for "proposed by teacher" vs "proposed by student", equal proposal rates:
  e(x) = q(x) / (pi(x) + q(x));  teacher action weight 1 - e(a') = sigmoid(-Delta'),  student action weight e(a) = sigmoid(Delta_a),
  Delta = log q - log pi summed over the action's tokens. Ybar = weighted mean; A = Y - Ybar; gradient sum_x w(x) A(x) grad log pi(x, cont).
Variants: overlap | unweighted (w = 1) | is (exact off-policy weight pi/q on the teacher action, 1 on the student) | placebo (overlap
weights, action = a step of another question).
Reference: GRPO gradient of 32 plain samples on questions all-fail at 8 but solvable at 32; the variants see only the 8 failed samples.
Anchor: the student's own action replayed with the same noise must reproduce tau (identical fraction reported).
PASS (written before running)
  P1 overlap is not empty: >= 20% of teacher actions have w' >= 0.05, and among them the flip rate is >= 2x the placebo flip rate.
  P2 direction: cos(overlap, GRPO32) > cos(unweighted) + 0.03, > cos(placebo) + 0.05, and > 0.

    PYTHONPATH=code python -m beat.diag_overlap --out runs/diag_overlap
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x)) if x > -700 else 0.0


def overlap_adv(Yt, Ys, wt, ws):
    """Two actions at one state: weighted mean and advantages."""
    Z = wt + ws
    ybar = (wt * Yt + ws * Ys) / Z if Z > 0 else 0.0
    return Yt - ybar, Ys - ybar


def selftest():
    assert abs(sigmoid(0) - 0.5) < 1e-12 and sigmoid(-800) == 0.0
    at, as_ = overlap_adv(1.0, 0.0, 1.0, 1.0)
    assert abs(at - 0.5) < 1e-12 and abs(as_ + 0.5) < 1e-12
    at, as_ = overlap_adv(1.0, 0.0, 1e-9, 1.0)          # teacher action off the overlap: tiny weight, Ybar ~ 0
    assert at > 0.99 and abs(as_) < 1e-6
    print("SELFTEST_OK", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=640)
    ap.add_argument("--extra", type=int, default=24)
    ap.add_argument("--max-hard", type=int, default=30)
    ap.add_argument("--fails", type=int, default=8)
    ap.add_argument("--cuts", default="0.33,0.66")
    ap.add_argument("--max-completion", type=int, default=1536)
    ap.add_argument("--gen-batch", type=int, default=64)
    ap.add_argument("--tok-budget", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=72)
    ap.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest()
    if a.selftest:
        return

    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
    from vllm import LLM, SamplingParams

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from beat import fastlp
    from fractions import Fraction
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
        enc = lambda user: tok(tok.apply_chat_template([msgs[0], {"role": "user", "content": user}], tokenize=False,
                                                       add_generation_prompt=True), add_special_tokens=False)["input_ids"]
        rows.append({"p": tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
                              add_special_tokens=False)["input_ids"],
                     "tp": enc(TEACHER_TEMPLATE.format(question=q, reference=ref)), "gt": r.reward_model["ground_truth"],
                     "facts": facts_of(ref, q, ans, 12, mask=False)})
    rows = [x for x in rows if len(x["p"]) <= 1024 and len(x["tp"]) <= 2048 and any(f["sig"] for f in x["facts"])]
    Q = [rows[i] for i in np.random.permutation(len(rows))[:a.questions].tolist()]

    # ---- vLLM: find questions all-fail at 8 but solvable at 32 (the reference GRPO-32 gradient uses these samples)
    llm = LLM(model=BASE_MODEL, gpu_memory_utilization=0.45, max_model_len=2048 + a.max_completion, seed=a.seed,
              enable_sleep_mode=True)
    sp = lambda n: SamplingParams(temperature=T, top_p=1.0, max_tokens=a.max_completion, stop_token_ids=stop_ids, n=n)

    def gen(xs, n):
        o = llm.generate([{"prompt_token_ids": x["p"]} for x in xs], sp(n), use_tqdm=False)
        txt = [c.text for oo in o for c in oo.outputs]
        rr = score_all(txt, [x["gt"] for x, oo in zip(xs, o) for _ in oo.outputs], 8)
        k = 0
        for x, oo in zip(xs, o):
            x.setdefault("V", [])
            for c in oo.outputs:
                comp = [t for t in c.token_ids if t not in stop_ids] or [stop_ids[0]]
                x["V"].append({"comp": comp, "r": float(rr[k])}); k += 1

    gen(Q, 8)
    AF = [x for x in Q if sum(s["r"] for s in x["V"]) == 0]
    gen(AF, a.extra)
    HARD = [x for x in AF if 0 < sum(s["r"] for s in x["V"]) < 8 + a.extra][:a.max_hard]
    stats = {"questions": len(Q), "allfail8": len(AF), "solvable32": sum(0 < sum(s["r"] for s in x["V"]) for x in AF),
             "hard_used": len(HARD)}
    print(f"VLLM {time.time() - t0:.0f}s " + json.dumps(stats), flush=True)
    llm.sleep(level=2); del llm; torch.cuda.empty_cache()

    # ---- HF model (LoRA B = 0 -> identical to base); exact Gumbel-max SCM generation
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=32, lora_alpha=64, lora_dropout=0.0,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                                                             "down_proj"]))
    params = [p for p in model.parameters() if p.requires_grad]
    model.eval()

    class Gumbel(LogitsProcessor):
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

    def run(jobs):
        res = [None] * len(jobs)
        order = sorted(range(len(jobs)), key=lambda i: -jobs[i]["budget"])
        for c0 in range(0, len(order), a.gen_batch):
            idx = [i for i in order[c0:c0 + a.gen_batch] if jobs[i]["budget"] > 0]
            for i in order[c0:c0 + a.gen_batch]:
                if jobs[i]["budget"] <= 0:
                    res[i] = ([], True)
            if not idx:
                continue
            P = max(len(jobs[i]["ctx"]) for i in idx)
            ids = torch.full((len(idx), P), pad_id, dtype=torch.long)
            att = torch.zeros((len(idx), P), dtype=torch.long)
            for r_, i in enumerate(idx):
                c = jobs[i]["ctx"]; ids[r_, P - len(c):] = torch.tensor(c); att[r_, P - len(c):] = 1
            with torch.no_grad():
                g = model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(),
                                   max_new_tokens=max(jobs[i]["budget"] for i in idx), do_sample=False, temperature=None,
                                   top_p=None, top_k=None, repetition_penalty=1.0, num_beams=1,
                                   logits_processor=LogitsProcessorList([Gumbel(P, [jobs[i]["seed"] for i in idx],
                                                                                 [jobs[i]["offset"] for i in idx])]),
                                   eos_token_id=stop_ids, pad_token_id=pad_id, use_cache=True)
            for r_, i in enumerate(idx):
                seq = g[r_, P:].tolist()
                st = next((k for k, t_ in enumerate(seq) if t_ in stop_ids), None)
                comp = (seq[:st] if st is not None else seq)[:jobs[i]["budget"]]
                res[i] = (comp, st is None or st >= jobs[i]["budget"])
        return res

    base_jobs = [{"x": x, "seed": s, "ctx": x["p"], "offset": 0, "budget": a.max_completion} for x in HARD for s in range(a.fails)]
    bo = run(base_jobs)
    br = score_all([tok.decode(c) for c, _ in bo], [j["x"]["gt"] for j in base_jobs], 8)
    taus = [(j, c) for j, (c, _), r in zip(base_jobs, bo, br) if r == 0 and len(c) >= 30]
    print(f"BASE {time.time() - t0:.0f}s scm_samples {len(base_jobs)} failed {len(taus)} success {float(np.mean(br)):.3f}", flush=True)

    # ---- states and actions
    fact_pool = [(id(x), f["hint"]) for x in Q for f in x["facts"] if f["sig"]]
    cuts = [float(c) for c in a.cuts.split(",")]
    states = []
    for ti, (j, tau) in enumerate(taus):
        x = j["x"]
        bounds = [k + 1 for k, t_ in enumerate(tau) if t_ in NL and 10 <= k + 1 <= len(tau) - 10]
        used = set()
        for fr in cuts:
            if not bounds:
                break
            t = min(bounds, key=lambda v: abs(v - fr * len(tau)))
            if t in used:
                continue
            used.add(t)
            nxt = next((k + 1 for k in range(t, len(tau)) if tau[k] in NL), len(tau))
            a_own = tau[t:nxt]
            seen = nums(tok.decode(tau[:t]))
            fj = next((f for f in x["facts"] if f["sig"] and not any(Fraction(v) in seen for v in f["sig"])), None)
            if fj is None or not a_own:
                continue
            txt = fj["hint"]
            plc = min((h for xid, h in fact_pool if xid != id(x)), key=lambda h: abs(len(h) - len(txt)))
            enc = lambda s_: tok(s_ + "\n", add_special_tokens=False)["input_ids"]
            states.append({"ti": ti, "x": x, "seed": j["seed"], "t": t, "nxt": nxt, "tau": tau,
                           "act": {"own": a_own, "teacher": enc(txt), "placebo": enc(plc)}})
    print(f"STATES {time.time() - t0:.0f}s {len(states)}", flush=True)

    # log pi and log q of each action at its state
    keys = ("own", "teacher", "placebo")
    with torch.no_grad():
        for k in keys:
            lp = fastlp.logps(model, pad_id, [s["x"]["p"] + s["tau"][:s["t"]] for s in states], [s["act"][k] for s in states],
                              False, 2 * a.tok_budget)
            lq = fastlp.logps(model, pad_id, [s["x"]["tp"] + s["tau"][:s["t"]] for s in states], [s["act"][k] for s in states],
                              False, 2 * a.tok_budget)
            for s, u, v in zip(states, lp, lq):
                s.setdefault("lpi", {})[k] = float(u.sum()); s.setdefault("lq", {})[k] = float(v.sum())
                s.setdefault("ntok", {})[k] = len(u)

    # counterfactual continuations: noise index continues from tau's position after its own action (nxt)
    cj = []
    for si, s in enumerate(states):
        for k in ("teacher", "placebo"):
            ctx = s["x"]["p"] + s["tau"][:s["t"]] + s["act"][k]
            cj.append({"si": si, "k": k, "seed": s["seed"], "offset": s["nxt"], "ctx": ctx,
                       "budget": a.max_completion - s["t"] - len(s["act"][k])})
        if si % 4 == 0:   # anchor: replay the student's own action -> must reproduce tau
            cj.append({"si": si, "k": "own_replay", "seed": s["seed"], "offset": s["nxt"],
                       "ctx": s["x"]["p"] + s["tau"][:s["nxt"]], "budget": a.max_completion - s["nxt"]})
    co = run(cj)
    texts = [tok.decode(states[c["si"]]["tau"][:states[c["si"]]["t"]] + (states[c["si"]]["act"][c["k"]] if c["k"] != "own_replay"
                                                                           else states[c["si"]]["act"]["own"]) + comp)
             for c, (comp, _) in zip(cj, co)]
    cr = score_all(texts, [states[c["si"]]["x"]["gt"] for c in cj], 8)
    replay = []
    for c, (comp, _), r in zip(cj, co, cr):
        s = states[c["si"]]
        if c["k"] == "own_replay":
            replay.append(comp == s["tau"][s["nxt"]:])
        else:
            s.setdefault("Y", {})[c["k"]] = float(r); s.setdefault("cont", {})[c["k"]] = comp
    print(f"CF {time.time() - t0:.0f}s continuations {len(cj)}", flush=True)

    # ---- P1 overlap
    for s in states:
        s["D"] = {k: s["lq"][k] - s["lpi"][k] for k in keys}
        s["w"] = {"teacher": sigmoid(-s["D"]["teacher"]), "placebo": sigmoid(-s["D"]["placebo"]), "own": sigmoid(s["D"]["own"])}
    wt = np.array([s["w"]["teacher"] for s in states]); yt = np.array([s["Y"]["teacher"] for s in states])
    yp = np.array([s["Y"]["placebo"] for s in states])
    inov = wt >= 0.05
    res = {"stats": stats, "states": len(states), "failed_tau": len(taus),
           "anchor_replay_identical": float(np.mean(replay)) if replay else None,
           "flip_teacher": float(yt.mean()), "flip_placebo": float(yp.mean()),
           "teacher_w_quantiles": {q_: float(np.quantile(wt, q_)) for q_ in (0.1, 0.25, 0.5, 0.75, 0.9)},
           "teacher_frac_w_ge_0.05": float(inov.mean()), "teacher_frac_w_ge_0.2": float((wt >= 0.2).mean()),
           "teacher_delta_per_token_median": float(np.median([s["D"]["teacher"] / s["ntok"]["teacher"] for s in states])),
           "teacher_delta_total_median": float(np.median([s["D"]["teacher"] for s in states])),
           "own_w_median": float(np.median([s["w"]["own"] for s in states])),
           "flip_teacher_in_overlap": float(yt[inov].mean()) if inov.any() else None,
           "flip_teacher_off_overlap": float(yt[~inov].mean()) if (~inov).any() else None,
           "n_in_overlap": int(inov.sum()), "n_flip_in_overlap": int((yt[inov] == 1).sum())}
    pf = res["flip_placebo"]
    res["P1_PASS"] = bool(res["teacher_frac_w_ge_0.05"] >= 0.2 and res["flip_teacher_in_overlap"] is not None
                          and res["flip_teacher_in_overlap"] >= 2 * max(pf, 1e-9))
    print("P1 " + json.dumps(res), flush=True)
    (out / "partial.json").write_text(json.dumps(res, indent=2))

    # ---- P2 gradient direction vs GRPO-32
    model.train()

    def wgrad(items):
        for p in params:
            p.grad = None
        items = [it for it in items if it[2] != 0 and len(it[1]) > 0]
        if not items:
            return None
        ntok = sum(len(it[1]) for it in items)
        for b in fastlp.token_batches([len(c) + len(d) for c, d, _ in items], a.tok_budget):
            mb = [items[i] for i in b]
            lp = fastlp.logps(model, pad_id, [c for c, _, _ in mb], [d for _, d, _ in mb], True, a.tok_budget)
            (sum(-(w * l).sum() for (_, _, w), l in zip(mb, lp)) / ntok).backward()
        g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).float().flatten() for p in params])
        for p in params:
            p.grad = None
        return g, ntok

    def cos(u, v):
        u = u[0] if isinstance(u, tuple) else u; v = v[0] if isinstance(v, tuple) else v
        return None if u is None or v is None else float(u @ v / (u.norm() * v.norm() + 1e-12))

    def combine(ga, gb):
        if ga is None or gb is None:
            return ga or gb
        return ((ga[0] * ga[1] + gb[0] * gb[1]) / (ga[1] + gb[1]), ga[1] + gb[1])

    def rl_items(xs):
        out_ = []
        for x in xs:
            m = np.mean([s["r"] for s in x["V"]])
            out_ += [(x["p"], s["comp"], float(s["r"] - m)) for s in x["V"]]
        return out_

    hA, hB = HARD[: len(HARD) // 2], HARD[len(HARD) // 2:]
    gA, gB = wgrad(rl_items(hA)), wgrad(rl_items(hB))
    g32 = combine(gA, gB)

    def cf_items(variant):
        items = []
        for s in states:
            ctx = s["x"]["p"] + s["tau"][:s["t"]]
            k = "placebo" if variant == "placebo" else "teacher"
            Yt, Ys = s["Y"][k], 0.0
            if variant in ("overlap", "placebo"):
                wt_, ws_ = s["w"][k], s["w"]["own"]
                At, As = overlap_adv(Yt, Ys, wt_, ws_)
                ct, cs = wt_ * At, ws_ * As
            elif variant == "unweighted":
                At, As = overlap_adv(Yt, Ys, 1.0, 1.0); ct, cs = At, As
            elif variant == "is":
                At, As = overlap_adv(Yt, Ys, 1.0, 1.0)
                ct, cs = math.exp(min(0.0, -s["D"][k])) * At, As   # pi/q (capped at 1) on the teacher-proposed action
            items.append((ctx, s["act"][k] + s["cont"][k], ct))
            items.append((ctx, s["tau"][s["t"]:], cs))
        return items

    p2 = {"anchor_grpo32_halfA_vs_halfB": cos(gA, gB)}
    for v in ("overlap", "unweighted", "is", "placebo"):
        p2[f"cos_{v}_vs_grpo32"] = cos(wgrad(cf_items(v)), g32)
    res["P2"] = p2
    co_ = p2["cos_overlap_vs_grpo32"] or -9
    res["P2_PASS"] = bool(co_ > (p2["cos_unweighted_vs_grpo32"] or -9) + 0.03 and co_ > (p2["cos_placebo_vs_grpo32"] or -9) + 0.05
                          and co_ > 0)
    print("P2 " + json.dumps(p2), flush=True)
    res["time"] = round(time.time() - t0, 1)
    (out / "results.json").write_text(json.dumps(res, indent=2))
    dump = [{"q": s["x"]["gt"][:80] if isinstance(s["x"]["gt"], str) else None, "t": s["t"], "len_tau": len(s["tau"]),
             "D": s["D"], "w": s["w"], "Y": s["Y"], "ntok": s["ntok"],
             "teacher_action": tok.decode(s["act"]["teacher"]), "own_action": tok.decode(s["act"]["own"]),
             "teacher_cont_head": tok.decode(s["cont"]["teacher"][:120])} for s in states]
    (out / "states.json").write_text(json.dumps(dump, ensure_ascii=False))
    print("RESULT " + json.dumps({"P1_PASS": res["P1_PASS"], "P2_PASS": res["P2_PASS"]}), flush=True)
    print(f"DONE {out}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
