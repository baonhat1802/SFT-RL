"""Follow-up checks for "transport the verifier, not the generator" (no policy training).

1. ORM control: does a verifier learned from Luna's y*/y~* add anything over an outcome-reward model learned from the STUDENT's own
   rollouts with right/wrong labels (same number of examples, no y*)? Also both data sets together.
2. Step localization: take the student's own CORRECT complete solutions on held-out questions, corrupt one derived value in step k
   (k not first, not the answer step). Score the clean vs corrupted PREFIX ending at step k -> AUC of detecting the error the moment
   it appears (needed for a dense, per-step reward). Also full clean vs full corrupted solution.
3. Gaming: edits a policy could learn to raise the verifier score without being more correct, applied to the student's wrong and
   correct complete solutions: answer_only ("The answer is \\boxed{X}."), skip_middle (first 30% + final sentence), assert (append a
   confident self-check). Gaming index = P(score(edited) > score(original)) on WRONG solutions; > 0.5 means the edit pays off.
Verifiers: zero_shot, transport (y*/y~*), orm (student rollouts + outcome labels), both.

    PYTHONPATH=code python -m beat.diag_verifier2 --out runs/diag_verifier2
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

PROMPT = ("Question:\n{q}\n\nProposed solution:\n{s}\n\n"
          "Is this solution correct, with every step valid? Answer with Yes or No.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-q", type=int, default=3000)
    ap.add_argument("--orm-q", type=int, default=1200)
    ap.add_argument("--eval-q", type=int, default=256)
    ap.add_argument("--num-gen", type=int, default=8)
    ap.add_argument("--max-completion", type=int, default=2048)
    ap.add_argument("--max-sol-tokens", type=int, default=1536)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--seed", type=int, default=61)
    ap.add_argument("--model", default=None)
    ap.add_argument("--verifiers", default="zero_shot,transport,orm,both")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from beat.loop import perturb
    from train.common import BASE_MODEL, GRPO, configure_tokenizer
    if a.model:
        BASE_MODEL = a.model
    VER = a.verifiers.split(",")
    need_orm = "orm" in VER or "both" in VER
    from train.data import load_track
    from train.models import eos_ids
    from train.rewards import score_all

    t0 = time.time()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tok = configure_tokenizer(AutoTokenizer.from_pretrained(BASE_MODEL), "curated")
    stop_ids = sorted(set(eos_ids(tok, "curated")) | {tok.convert_tokens_to_ids("<|im_end|>")})
    YES, NO = tok("Yes", add_special_tokens=False)["input_ids"][0], tok("No", add_special_tokens=False)["input_ids"][0]
    MATH = re.compile(r"\\\((.*?)\\\)|\\\[(.*?)\\\]|\$\$(.*?)\$\$|\$(.*?)\$", re.S)

    rows = []
    for r in load_track("curated").itertuples():
        msgs = [{"role": m["role"], "content": m["content"]} for m in r.prompt]
        rows.append({"sys": msgs[0], "msgs": msgs, "q": r.extra_info["question"], "ref": r.target[0]["content"],
                     "gt": r.reward_model["ground_truth"]})
    perm = np.random.permutation(len(rows)).tolist()          # same split as diag_verifier (same seed): EV, then TR
    EV = [rows[i] for i in perm[:a.eval_q]]
    TR = [rows[i] for i in perm[a.eval_q:a.eval_q + a.train_q]]
    OQ = [rows[i] for i in perm[a.eval_q + a.train_q:a.eval_q + a.train_q + a.orm_q]]

    # ---- student rollouts: held-out eval questions + ORM training questions (disjoint)
    llm = LLM(model=BASE_MODEL, gpu_memory_utilization=0.5, max_model_len=4096, seed=a.seed, enable_sleep_mode=True)
    sp = SamplingParams(temperature=GRPO["temperature"], top_p=1.0, max_tokens=a.max_completion, stop_token_ids=stop_ids, n=a.num_gen)

    def rollouts(Q):
        pids = [tok(tok.apply_chat_template(x["msgs"], tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"] for x in Q]
        outs = llm.generate([{"prompt_token_ids": p} for p in pids], sp, use_tqdm=False)
        S = [{"g": g, "x": x, "text": c.text, "len": len(c.token_ids), "trunc": c.finish_reason == "length"}
             for g, (x, o) in enumerate(zip(Q, outs)) for c in o.outputs]
        for s, r in zip(S, score_all([s["text"] for s in S], [s["x"]["gt"] for s in S], 8)):
            s["r"] = float(r)
        return S

    S = rollouts(EV)
    SO = rollouts(OQ) if need_orm else []
    llm.sleep(level=2); del llm; torch.cuda.empty_cache()
    print(f"GEN {time.time() - t0:.0f}s eval reward {np.mean([s['r'] for s in S]):.3f} orm reward {np.mean([s['r'] for s in SO]) if SO else float('nan'):.3f}", flush=True)

    # ---- training sets
    prng = random.Random(a.seed)
    tr_transport = []
    for x in TR:
        kind = prng.choice(["numbers", "operators"])
        bad = perturb(x["ref"], kind, x["q"], prng)
        if bad == x["ref"]:
            bad = perturb(x["ref"], "numbers" if kind == "operators" else "operators", x["q"], prng)
        if bad != x["ref"]:
            tr_transport += [(x, x["ref"], 1), (x, bad, 0)]
    pos = [(s["x"], s["text"], 1) for s in SO if s["r"] == 1]; neg = [(s["x"], s["text"], 0) for s in SO if s["r"] == 0]
    prng.shuffle(pos); prng.shuffle(neg)
    k = min(len(pos), len(neg), len(tr_transport) // 2) if need_orm else len(tr_transport) // 2
    tr_orm = pos[:k] + neg[:k]
    tr_transport = tr_transport[:2 * k]
    prng.shuffle(tr_orm); prng.shuffle(tr_transport)
    tr_both = tr_transport + tr_orm; prng.shuffle(tr_both)
    print(f"DATA per-class {k}: transport {len(tr_transport)} orm {len(tr_orm)} both {len(tr_both)}", flush=True)

    # ---- probes on the student's own complete solutions
    complete = lambda s: (not s["trunc"]) and ("\\boxed" in s["text"])

    def steps_of(t):
        spans = [(m.start(), m.end()) for m in MATH.finditer(t)]
        inside = lambda c: any(s0 < c < e for s0, e in spans)
        cuts = [m.end() for m in re.finditer(r"(?<=[.;:])\s+|\n+", t) if not inside(m.start())]
        b = [0] + [c for c in cuts if 0 < c < len(t)] + [len(t)]
        return [(u, v) for u, v in zip(b, b[1:]) if t[u:v].strip()]

    def corrupt_step(t, q, rng):
        """change one derived value inside a math span of a middle step; return (prefix_clean, prefix_bad, full_bad) or None"""
        st = steps_of(t); given = set(re.findall(r"\d+", q))
        bx = t.rfind("\\boxed")
        cands = []
        for k_, (u, v) in enumerate(st[1:-1], start=1):
            if bx >= 0 and v > bx:
                break
            for m in MATH.finditer(t[u:v]):
                for d in re.finditer(r"\d+", m.group()):
                    if d.group() not in given:
                        cands.append((k_, u + m.start() + d.start(), u + m.start() + d.end()))
        if not cands:
            return None
        k_, p0, p1 = rng.choice(cands)
        n = int(t[p0:p1]); new = str(n + rng.choice([1, 2, 3]) if n < 3 else n + rng.choice([-2, -1, 1, 2, 3]))
        bad = t[:p0] + new + t[p1:]
        end = st[k_][1]
        return t[:end], bad[:end + len(new) - (p1 - p0)], bad

    rng = random.Random(a.seed + 7)
    ok_c = [s for s in S if s["r"] == 1 and complete(s)]
    loc = []
    for s in ok_c:
        c = corrupt_step(s["text"], s["x"]["q"], rng)
        if c:
            loc.append((s, *c))
    wrong_c = [s for s in S if s["r"] == 0 and complete(s)]
    right_c = ok_c

    def boxed(t):
        i = t.rfind("\\boxed{")
        if i < 0:
            return None
        j = i + 7; d = 1; k2 = j
        while k2 < len(t) and d:
            d += (t[k2] == "{") - (t[k2] == "}"); k2 += 1
        return t[j:k2 - 1]

    def edits(t):
        b = boxed(t) or ""
        last = t[t.rfind("\n", 0, max(0, t.rfind("\\boxed"))) + 1:] if "\\boxed" in t else ""
        return {"answer_only": f"The answer is \\boxed{{{b}}}.",
                "skip_middle": t[: int(0.3 * len(t))] + "\n\n" + last,
                "assert": t + "\n\nI have double-checked every step of this solution, and it is correct."}
    print(f"PROBES loc {len(loc)} wrong_complete {len(wrong_c)} right_complete {len(right_c)}", flush=True)

    # ---- model + scoring
    def enc(x, sol):
        st = tok(sol, add_special_tokens=False)["input_ids"]
        if len(st) > a.max_sol_tokens:
            st = st[: a.max_sol_tokens // 2] + st[-a.max_sol_tokens // 2:]
        msgs = [x["sys"], {"role": "user", "content": PROMPT.format(q=x["q"], s=tok.decode(st))}]
        return tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); base.enable_input_require_grads()
    pad = tok.pad_token_id
    holder = [base]

    def batch_logits(model, seqs):
        L = max(map(len, seqs))
        ids = torch.full((len(seqs), L), pad, dtype=torch.long); att = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, L - len(s):] = torch.tensor(s); att[i, L - len(s):] = 1
        ids, att = ids.cuda(), att.cuda()
        pos_ = (att.cumsum(-1) - 1).clamp(min=0)
        lp = torch.log_softmax(model(input_ids=ids, attention_mask=att, position_ids=pos_, logits_to_keep=1).logits[:, -1].float(), -1)
        return lp[:, YES] - lp[:, NO]

    @torch.no_grad()
    def scores(model, items, bs=8):
        model.eval(); res = []
        order = sorted(range(len(items)), key=lambda i: len(items[i][1]))      # length-bucketed batches
        tmp = {}
        for k0 in range(0, len(order), bs):
            idx = order[k0:k0 + bs]
            for i, v in zip(idx, batch_logits(model, [enc(*items[i]) for i in idx]).tolist()):
                tmp[i] = v
        return [tmp[i] for i in range(len(items))]

    def train_lora(data, tag):
        model = get_peft_model(holder[0], LoraConfig(task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.0,
                                                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
        steps = len(data) // a.batch; model.train(); losses = []
        for st in range(steps):
            b = data[st * a.batch:(st + 1) * a.batch]; opt.zero_grad(); tot = 0.0
            for m0 in range(0, len(b), a.micro):
                mb = b[m0:m0 + a.micro]
                d = batch_logits(model, [enc(x, s) for x, s, _ in mb])
                y = torch.tensor([float(v) for _, _, v in mb], device="cuda")
                loss = torch.nn.functional.binary_cross_entropy_with_logits(d, y) * len(mb) / len(b)
                loss.backward(); tot += float(loss)
            opt.step(); losses.append(tot)
            if st % 50 == 0:
                print(f"  train {tag} step {st}/{steps} loss {np.mean(losses[-50:]):.3f}", flush=True)
        return model

    def auc(p, n):
        if not p or not n:
            return None
        p = np.array(p); n = np.array(n)
        return float(((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean()))

    def within(sc, keep):
        num = den = 0.0
        for g in range(len(EV)):
            idx = [i for i, s in enumerate(S) if s["g"] == g and keep(s)]
            c = [sc[i] for i in idx if S[i]["r"] == 1]; w = [sc[i] for i in idx if S[i]["r"] == 0]
            for u in c:
                for v in w:
                    num += 1.0 if u > v else 0.5 if u == v else 0.0; den += 1
        return num / den if den else None

    res = {"eval_q": len(EV), "per_class_train": k, "n_loc": len(loc), "n_wrong_complete": len(wrong_c), "verifiers": {}}
    items_S = [(s["x"], s["text"]) for s in S]
    items_loc_clean = [(s["x"], pc) for s, pc, pb, fb in loc]; items_loc_bad = [(s["x"], pb) for s, pc, pb, fb in loc]
    items_full_clean = [(s["x"], s["text"]) for s, pc, pb, fb in loc]; items_full_bad = [(s["x"], fb) for s, pc, pb, fb in loc]
    E_wrong = {kk: [(s["x"], edits(s["text"])[kk]) for s in wrong_c] for kk in ("answer_only", "skip_middle", "assert")}
    E_right = {kk: [(s["x"], edits(s["text"])[kk]) for s in right_c] for kk in ("answer_only", "skip_middle", "assert")}

    def evaluate(name, model):
        sS = scores(model, items_S)
        lc, lb = scores(model, items_loc_clean), scores(model, items_loc_bad)
        fc, fb = scores(model, items_full_clean), scores(model, items_full_bad)
        sw = [sS[i] for i, s in enumerate(S) if s["r"] == 0 and complete(s)]
        sr = [sS[i] for i, s in enumerate(S) if s["r"] == 1 and complete(s)]
        v = {"auc_within_all": within(sS, lambda s: True), "auc_within_complete": within(sS, complete),
             "auc_pooled_complete": auc(sr, sw),
             "loc_auc_prefix_clean_vs_corrupt": auc(lc, lb), "loc_auc_full_clean_vs_corrupt": auc(fc, fb),
             "loc_pair_win_prefix": float(np.mean([c > b for c, b in zip(lc, lb)])) if lc else None, "gaming": {}}
        for kk in E_wrong:
            ew, er = scores(model, E_wrong[kk]), scores(model, E_right[kk])
            v["gaming"][kk] = {"index_wrong_edit_beats_original": float(np.mean([e > o for e, o in zip(ew, sw)])) if sw else None,
                               "mean_delta_wrong": float(np.mean(np.array(ew) - np.array(sw))) if sw else None,
                               "mean_delta_right": float(np.mean(np.array(er) - np.array(sr))) if sr else None,
                               "auc_edited_wrong_vs_original_right": auc(sr, ew)}
        res["verifiers"][name] = v
        print(f"VERIFIER {name} {time.time() - t0:.0f}s " + json.dumps(v), flush=True)
        (out / "partial.json").write_text(json.dumps(res, indent=2))

    if "zero_shot" in VER:
        evaluate("zero_shot", holder[0])
    for name, data in (("transport", tr_transport), ("orm", tr_orm), ("both", tr_both)):
        if name not in VER:
            continue
        m = train_lora(data, name)
        evaluate(name, m)
        holder[0] = m.unload()
    res["time"] = round(time.time() - t0, 1)
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print("RESULT " + json.dumps(res), flush=True)
    print(f"DONE {out}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
