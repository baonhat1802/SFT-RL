"""Transport the verifier, not the generator — pre-screen (no policy training).

Causal argument (transportability, ICM): the GENERATION mechanism P(next step | context, solver) differs between Luna and the 3B
student (Luna has private reasoning U), so imitating it does not transport. The VALIDITY mechanism P(step valid | context) is the
same for every solver, so it should transport. Test: learn a verifier ONLY from Luna's y* (valid) and same-style counterfactuals y~*
(1-2 derived values or +/- signs changed, invalid), then ask whether it ranks the STUDENT's own solutions correctly.

Verifier = base model + LoRA, prompt "Question / Proposed solution / Is this solution correct ...? Answer Yes or No",
score = log p(Yes) - log p(No) at the first answer token.
  zero_shot   adapter off                                         (does the base model already verify?)
  transport   LoRA trained on (y*, Yes) and (y~*, No)             (the idea)
  placebo     same texts, labels shuffled                          (control: training on the format only)
Student solutions: base model, 8 samples per held-out question (questions disjoint from the verifier training set).
Metrics: AUC correct-vs-wrong within mixed groups (all, and only complete + boxed solutions), pooled AUC, AUC of -length (anchor),
in-domain AUC on held-out (y*, y~*) pairs.

    PYTHONPATH=code python -m beat.diag_verifier --out runs/diag_verifier
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

PROMPT = ("Question:\n{q}\n\nProposed solution:\n{s}\n\n"
          "Is this solution correct, with every step valid? Answer with Yes or No.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-q", type=int, default=3000)
    ap.add_argument("--eval-q", type=int, default=256)
    ap.add_argument("--num-gen", type=int, default=8)
    ap.add_argument("--max-completion", type=int, default=2048)
    ap.add_argument("--max-sol-tokens", type=int, default=1536)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=61)
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
    from train.data import load_track
    from train.models import eos_ids
    from train.rewards import score_all

    t0 = time.time()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tok = configure_tokenizer(AutoTokenizer.from_pretrained(BASE_MODEL), "curated")
    stop_ids = sorted(set(eos_ids(tok, "curated")) | {tok.convert_tokens_to_ids("<|im_end|>")})
    YES, NO = tok("Yes", add_special_tokens=False)["input_ids"][0], tok("No", add_special_tokens=False)["input_ids"][0]

    rows = []
    for r in load_track("curated").itertuples():
        msgs = [{"role": m["role"], "content": m["content"]} for m in r.prompt]
        rows.append({"sys": msgs[0], "msgs": msgs, "q": r.extra_info["question"], "ref": r.target[0]["content"],
                     "gt": r.reward_model["ground_truth"]})
    perm = np.random.permutation(len(rows)).tolist()
    EV = [rows[i] for i in perm[:a.eval_q]]
    TR = [rows[i] for i in perm[a.eval_q:a.eval_q + a.train_q]]

    # ---- student solutions on held-out questions
    llm = LLM(model=BASE_MODEL, gpu_memory_utilization=0.5, max_model_len=4096, seed=a.seed, enable_sleep_mode=True)
    sp = SamplingParams(temperature=GRPO["temperature"], top_p=1.0, max_tokens=a.max_completion, stop_token_ids=stop_ids, n=a.num_gen)
    pids = [tok(tok.apply_chat_template(x["msgs"], tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"] for x in EV]
    outs = llm.generate([{"prompt_token_ids": p} for p in pids], sp, use_tqdm=False)
    S = []
    for g, (x, o) in enumerate(zip(EV, outs)):
        for c in o.outputs:
            S.append({"g": g, "text": c.text, "len": len(c.token_ids), "trunc": c.finish_reason == "length", "q": x["q"], "gt": x["gt"]})
    rew = score_all([s["text"] for s in S], [s["gt"] for s in S], 8)
    for s, r in zip(S, rew):
        s["r"] = float(r)
    llm.sleep(level=2); del llm; torch.cuda.empty_cache()
    G = np.array([s["r"] for s in S]).reshape(len(EV), a.num_gen)
    print(f"GEN {time.time() - t0:.0f}s reward {G.mean():.3f} mixed {int(((G.sum(1) > 0) & (G.sum(1) < a.num_gen)).sum())}", flush=True)

    # ---- verifier training data from y* / y~* only
    prng = random.Random(a.seed)
    data = []
    for x in TR:
        kind = prng.choice(["numbers", "operators"])
        bad = perturb(x["ref"], kind, x["q"], prng)
        if bad == x["ref"]:
            bad = perturb(x["ref"], "numbers" if kind == "operators" else "operators", x["q"], prng)
        if bad == x["ref"]:
            continue
        data.append((x, x["ref"], 1)); data.append((x, bad, 0))
    prng.shuffle(data)
    n_hold = 400
    hold, train = data[:n_hold], data[n_hold:]
    print(f"DATA train {len(train)} hold {len(hold)}", flush=True)

    def enc(x, sol):
        st = tok(sol, add_special_tokens=False)["input_ids"]
        if len(st) > a.max_sol_tokens:          # keep the end (the answer) and the start
            st = st[: a.max_sol_tokens // 2] + st[-a.max_sol_tokens // 2:]
        sol = tok.decode(st)
        msgs = [x["sys"], {"role": "user", "content": PROMPT.format(q=x["q"], s=sol)}]
        return tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); base.enable_input_require_grads()
    pad = tok.pad_token_id
    base_ref = [base]

    def batch_logits(model, seqs):
        L = max(map(len, seqs))
        ids = torch.full((len(seqs), L), pad, dtype=torch.long); att = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, L - len(s):] = torch.tensor(s); att[i, L - len(s):] = 1
        ids, att = ids.cuda(), att.cuda()
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        lg = model(input_ids=ids, attention_mask=att, position_ids=pos, logits_to_keep=1).logits[:, -1].float()
        lp = torch.log_softmax(lg, -1)
        return lp[:, YES] - lp[:, NO]

    @torch.no_grad()
    def scores(model, items, bs=8):
        model.eval(); res = []
        for k in range(0, len(items), bs):
            res += batch_logits(model, [enc(x, s) for x, s in items[k:k + bs]]).tolist()
        return res

    def train_lora(labels_shuffled):
        nonlocal_base = base_ref[0]
        model = get_peft_model(nonlocal_base, LoraConfig(task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.0,
                                                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        tr = list(train)
        if labels_shuffled:
            labs = [y for _, _, y in tr]; random.Random(a.seed + 1).shuffle(labs)
            tr = [(x, s, y) for (x, s, _), y in zip(tr, labs)]
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
        steps = int(len(tr) * a.epochs) // a.batch
        model.train(); losses = []
        for st in range(steps):
            b = tr[(st * a.batch) % len(tr):(st * a.batch) % len(tr) + a.batch]
            d = batch_logits(model, [enc(x, s) for x, s, _ in b])
            y = torch.tensor([float(v) for _, _, v in b], device="cuda")
            loss = torch.nn.functional.binary_cross_entropy_with_logits(d, y)
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss))
            if st % 50 == 0:
                print(f"  train{'(placebo)' if labels_shuffled else ''} step {st}/{steps} loss {np.mean(losses[-50:]):.3f}", flush=True)
        return model

    def auc_pairs(pos, neg):
        if not pos or not neg:
            return None
        return float(np.mean([1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg]))

    def within_auc(sc, keep):
        num = den = 0.0
        for g in range(len(EV)):
            idx = [i for i, s in enumerate(S) if s["g"] == g and keep(s)]
            c = [sc[i] for i in idx if S[i]["r"] == 1]; w = [sc[i] for i in idx if S[i]["r"] == 0]
            for u in c:
                for v in w:
                    num += 1.0 if u > v else 0.5 if u == v else 0.0; den += 1
        return num / den if den else None

    complete = lambda s: (not s["trunc"]) and ("\\boxed" in s["text"])
    items_S = [({"sys": EV[s["g"]]["sys"], "q": s["q"]}, s["text"]) for s in S]
    items_H = [(x, s) for x, s, _ in hold]; lab_H = [y for _, _, y in hold]
    res = {"eval_q": len(EV), "samples": len(S), "reward": float(G.mean()), "train_pairs": len(train) // 2,
           "anchor_auc_neg_length": within_auc([-s["len"] for s in S], lambda s: True),
           "anchor_auc_neg_length_complete": within_auc([-s["len"] for s in S], complete),
           "frac_complete": float(np.mean([complete(s) for s in S])), "verifiers": {}}

    def evaluate(name, model, adapter_off=False):
        if adapter_off:
            with model.disable_adapter() if hasattr(model, "disable_adapter") else torch.no_grad():
                sS, sH = scores(model, items_S), scores(model, items_H)
        else:
            sS, sH = scores(model, items_S), scores(model, items_H)
        res["verifiers"][name] = {
            "auc_within_all": within_auc(sS, lambda s: True),
            "auc_within_complete": within_auc(sS, complete),
            "auc_pooled_complete": auc_pairs([v for v, s in zip(sS, S) if complete(s) and s["r"] == 1],
                                             [v for v, s in zip(sS, S) if complete(s) and s["r"] == 0]),
            "auc_holdout_ystar_vs_cf": auc_pairs([v for v, y in zip(sH, lab_H) if y == 1], [v for v, y in zip(sH, lab_H) if y == 0]),
            "mean_score_student_correct": float(np.mean([v for v, s in zip(sS, S) if s["r"] == 1])) if any(s["r"] == 1 for s in S) else None,
            "mean_score_student_wrong": float(np.mean([v for v, s in zip(sS, S) if s["r"] == 0]))}
        print(f"VERIFIER {name} {time.time() - t0:.0f}s " + json.dumps(res["verifiers"][name]), flush=True)
        (out / "partial.json").write_text(json.dumps(res, indent=2))

    evaluate("zero_shot", base)
    m = train_lora(False)
    evaluate("transport", m)
    base_ref[0] = m.unload()               # LoRA removed (not merged) -> clean base for the placebo run
    m2 = train_lora(True)
    evaluate("placebo", m2)
    res["time"] = round(time.time() - t0, 1)
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print("RESULT " + json.dumps(res), flush=True)
    print(f"DONE {out}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
