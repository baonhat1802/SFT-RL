"""Pre-screen for R02-1 `r02_fork` (no training): can the y*-teacher LOCATE where a failed solution should be re-branched?

For questions whose 8 samples are all wrong, take one wrong solution, choose a fork position t with several locators, let the
STUDENT (no y* in its context) continue k times from prefix comp[:t], and measure how often a continuation is correct.

Locators (same k for each, so the comparison is at equal generation budget):
  start          t = 0: k fresh samples from the prompt (what more GRPO samples would give)
  random         uniform t in the allowed range
  entropy        t = argmax student entropy (uncertainty-based branching, as TreeAdv / LATR)
  teacher_first  first t with logp_T - logp_S < -tau (the teacher sees y*: TEACHER_TEMPLATE), fallback argmin
  teacher_min    t = argmin (logp_T - logp_S): strongest disagreement
  teacher_full_first / teacher_full_min   same, teacher context = HINT_TEMPLATE (full)
  depth_matched  the RELATIVE position teacher_first chose on ANOTHER case, applied here: same depth distribution as the
                 teacher, no information about this solution (DEEP-GRPO shows depth alone matters)

The idea survives only if a teacher locator rescues clearly more questions than `random` and `entropy`.

    PYTHONPATH=code python -m beat.diag_fork --prompts 384 --k 8 --out runs/diag_fork
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=int, default=384)
    ap.add_argument("--num-gen", type=int, default=8)
    ap.add_argument("--k", type=int, default=8, help="continuations per fork")
    ap.add_argument("--max-completion", type=int, default=2048)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--margin", type=int, default=16, help="no fork in the first/last `margin` tokens")
    ap.add_argument("--vllm-mem", type=float, default=0.6)
    ap.add_argument("--tok-budget", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from beat import fastlp
    from beat.loop import HINT_TEMPLATE, TEACHER_TEMPLATE
    from train.common import BASE_MODEL, GRPO, configure_tokenizer
    from train.data import load_track
    from train.models import eos_ids
    from train.rewards import score_all

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tok = configure_tokenizer(AutoTokenizer.from_pretrained(BASE_MODEL), "curated")
    stop_ids = sorted(set(eos_ids(tok, "curated")) | {tok.convert_tokens_to_ids("<|im_end|>")})

    rows = []
    for r in load_track("curated").itertuples():
        msgs = [{"role": m["role"], "content": m["content"]} for m in r.prompt]
        q, ref = r.extra_info["question"], r.target[0]["content"]
        enc = lambda user: tok(tok.apply_chat_template([msgs[0], {"role": "user", "content": user}], tokenize=False,
                                                       add_generation_prompt=True), add_special_tokens=False)["input_ids"]
        rows.append({"p_ids": tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
                                  add_special_tokens=False)["input_ids"],
                     "tp_ids": enc(TEACHER_TEMPLATE.format(question=q, reference=ref)),
                     "hp_ids": enc(HINT_TEMPLATE.format(question=q, reference=ref)),
                     "gt": r.reward_model["ground_truth"]})
    rows = [x for x in rows if len(x["p_ids"]) <= 1024 and len(x["hp_ids"]) <= 2048]

    llm = LLM(model=BASE_MODEL, gpu_memory_utilization=a.vllm_mem, max_model_len=2048 + a.max_completion, seed=a.seed,
              enable_prefix_caching=True, enable_sleep_mode=True)
    T = GRPO["temperature"]
    sp = SamplingParams(temperature=T, top_p=1.0, max_tokens=a.max_completion, stop_token_ids=stop_ids, n=a.num_gen)

    # 1) rollouts, keep all-fail questions (one non-truncated wrong solution each)
    t0 = time.time()
    idx = np.random.permutation(len(rows))[:a.prompts].tolist()
    outs = llm.generate([{"prompt_token_ids": rows[i]["p_ids"]} for i in idx], sp, use_tqdm=False)
    texts = [c.text for o in outs for c in o.outputs]
    rew = np.array(score_all(texts, [rows[i]["gt"] for i in idx for _ in range(a.num_gen)], 8)).reshape(len(idx), a.num_gen)
    cases = []
    for g, (i, o) in enumerate(zip(idx, outs)):
        if rew[g].sum() == 0:
            cands = [c for c in o.outputs if c.finish_reason == "stop" and len(c.token_ids) > 4 * a.margin]
            if cands:
                c = random.choice(cands)
                cases.append({"i": i, "comp": list(c.token_ids)})
    print(f"rollouts {time.time() - t0:.0f}s: reward {rew.mean():.3f}, all-fail {int((rew.sum(1) == 0).sum())}/{len(idx)}, "
          f"cases {len(cases)}", flush=True)
    llm.sleep(level=1)

    # 2) teacher / student per-token scores on the wrong solution
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.eval()

    class _Wrap:                         # fastlp expects a peft model; the base model has no adapter here
        def __init__(self, m): self.m = m
        def get_base_model(self): return self.m
    W = _Wrap(model)
    comps = [c["comp"] for c in cases]
    with torch.no_grad():
        lpS = fastlp.logps(W, tok.pad_token_id, [rows[c["i"]]["p_ids"] for c in cases], comps, False, 2 * a.tok_budget)
        lpT = fastlp.logps(W, tok.pad_token_id, [rows[c["i"]]["tp_ids"] for c in cases], comps, False, 2 * a.tok_budget)
        lpF = fastlp.logps(W, tok.pad_token_id, [rows[c["i"]]["hp_ids"] for c in cases], comps, False, 2 * a.tok_budget)
        ents = []
        for c in cases:                  # student entropy per position (one sample at a time, chunked head)
            ids = torch.tensor([rows[c["i"]]["p_ids"] + c["comp"]], device="cuda")
            hid = model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
            n, L = len(c["comp"]), ids.shape[1]
            h = hid[L - 1 - n:L - 1]
            e = []
            for s in range(0, n, 512):
                lg = model.lm_head(h[s:s + 512]).float()
                p = torch.softmax(lg, -1)
                e.append(-(p * torch.log_softmax(lg, -1)).sum(-1))
            ents.append(torch.cat(e).cpu())
    del model; torch.cuda.empty_cache()

    def first_below(d, lo, hi):
        for t in range(lo, hi):
            if d[t] < -a.tau:
                return t
        return lo + int(torch.argmin(d[lo:hi]))

    forks = {k: [] for k in ("start", "random", "entropy", "teacher_first", "teacher_min",
                             "teacher_full_first", "teacher_full_min")}
    for c, s, t_, f, e in zip(cases, lpS, lpT, lpF, ents):
        n = len(c["comp"]); lo, hi = a.margin, n - a.margin
        d = (t_ - s).cpu(); dF = (f - s).cpu()
        forks["start"].append(0)
        forks["random"].append(random.randrange(lo, hi))
        forks["entropy"].append(lo + int(torch.argmax(e[lo:hi])))
        forks["teacher_first"].append(first_below(d, lo, hi))
        forks["teacher_min"].append(lo + int(torch.argmin(d[lo:hi])))
        forks["teacher_full_first"].append(first_below(dF, lo, hi))
        forks["teacher_full_min"].append(lo + int(torch.argmin(dF[lo:hi])))

    rel = [t / len(c["comp"]) for c, t in zip(cases, forks["teacher_first"])]
    perm = list(range(len(cases))); random.shuffle(perm)
    forks["depth_matched"] = [min(max(int(rel[j] * len(c["comp"])), a.margin), len(c["comp"]) - a.margin - 1)
                              for c, j in zip(cases, perm)]

    # 3) k student continuations from every fork (no y* in the context)
    torch.cuda.empty_cache(); llm.wake_up()
    res = {}
    for name, ts in forks.items():
        prompts = [rows[c["i"]]["p_ids"] + c["comp"][:t] for c, t in zip(cases, ts)]
        spk = SamplingParams(temperature=T, top_p=1.0, max_tokens=a.max_completion, stop_token_ids=stop_ids, n=a.k)
        o2 = llm.generate([{"prompt_token_ids": p} for p in prompts], spk, use_tqdm=False)
        full = [tok.decode(c["comp"][:t]) + x.text for c, t, o in zip(cases, ts, o2) for x in o.outputs]
        r2 = np.array(score_all(full, [rows[c["i"]]["gt"] for c in cases for _ in range(a.k)], 8)).reshape(len(cases), a.k)
        pos = [t / len(c["comp"]) for c, t in zip(cases, ts)]
        boot = [r2[np.random.randint(0, len(cases), len(cases))].any(1).mean() for _ in range(1000)]
        res[name] = {"success": float(r2.mean()), "rescued": float(r2.any(1).mean()),
                     "rescued_ci90": [float(np.percentile(boot, 5)), float(np.percentile(boot, 95))],
                     "mixed_groups": float(((r2.sum(1) > 0) & (r2.sum(1) < a.k)).mean()),
                     "pos_mean": float(np.mean(pos)), "per_case_correct": r2.sum(1).tolist()}
        print(f"{name:20s} success {res[name]['success']:.3f}  rescued {res[name]['rescued']:.3f} "
              f"[{res[name]['rescued_ci90'][0]:.3f}, {res[name]['rescued_ci90'][1]:.3f}]  pos {res[name]['pos_mean']:.2f}",
              flush=True)
    (out / "results.json").write_text(json.dumps({"n_cases": len(cases), "args": vars(a), "results": res}, indent=2))
    print(f"DONE {out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
