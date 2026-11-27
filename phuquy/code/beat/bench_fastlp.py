"""Check beat/fastlp.py against the old loop.logps_of path: same log-probs, same gradients, then speed and peak memory.

    PYTHONPATH=code python -m beat.bench_fastlp --n 64 --budget 32768
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--budget", type=int, default=32768)
    ap.add_argument("--old-mb", type=int, default=4)
    ap.add_argument("--time-budgets", default="", help="comma list: only time the new path at these budgets")
    a = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from beat import fastlp
    from train.common import BASE_MODEL, LORA, configure_tokenizer
    from train.data import load_track

    random.seed(0); torch.manual_seed(0)
    tok = configure_tokenizer(AutoTokenizer.from_pretrained(BASE_MODEL), "curated")
    df = load_track("curated").head(a.n)
    ctxs, comps = [], []
    for k, r in enumerate(df.itertuples()):
        msgs = [{"role": m["role"], "content": m["content"]} for m in r.prompt]
        ctxs.append(tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
                        add_special_tokens=False)["input_ids"])
        ref = tok(r.target[0]["content"], add_special_tokens=False)["input_ids"]
        target = random.choice([64, 256, 600, 1200, a.max_len])          # mixed lengths, as in a real rollout
        comps.append((ref * (target // max(len(ref), 1) + 1))[:target])

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **{**LORA, "r": 32, "lora_alpha": 64}))
    with torch.no_grad():                         # non-zero LoRA B, so A also gets gradients
        for n_, p in model.named_parameters():
            if "lora_B" in n_:
                p.normal_(0, 1e-3)
    params = [p for p in model.parameters() if p.requires_grad]
    model.train()

    def old_logps(cs, ds, grad, mb):
        res = []
        for i in range(0, len(cs), mb):
            c_, d_ = cs[i:i + mb], ds[i:i + mb]
            seqs = [x + y for x, y in zip(c_, d_)]
            L, K = max(map(len, seqs)), max(map(len, d_)) + 1
            ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(seqs), L), dtype=torch.long)
            for j, s in enumerate(seqs):
                ids[j, L - len(s):] = torch.tensor(s); att[j, L - len(s):] = 1
            ids, att = ids.cuda(), att.cuda()
            pos = (att.cumsum(-1) - 1).clamp(min=0)
            with torch.set_grad_enabled(grad):
                logits = model(input_ids=ids, attention_mask=att, position_ids=pos, logits_to_keep=K).logits
                for j, d in enumerate(d_):
                    n = len(d)
                    res.append(torch.log_softmax(logits[j, K - 1 - n:K - 1].float(), -1)
                               .gather(-1, ids[j, L - n:][:, None]).squeeze(-1))
            del logits
        return res

    W = [torch.randn(len(d), device="cuda") for d in comps]       # a fixed per-token weighting for the test loss
    ntok = sum(map(len, comps))

    def run(kind):
        for p in params:
            p.grad = None
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        if kind == "old":
            for i in range(0, len(ctxs), a.old_mb):
                lp = old_logps(ctxs[i:i + a.old_mb], comps[i:i + a.old_mb], True, a.old_mb)
                (sum((w * x).sum() for w, x in zip(W[i:i + a.old_mb], lp)) / ntok).backward()
        else:
            for b in fastlp.token_batches([len(c) + len(d) for c, d in zip(ctxs, comps)], a.budget):
                lp = fastlp.logps(model, tok.pad_token_id, [ctxs[i] for i in b], [comps[i] for i in b], True, a.budget)
                (sum((W[i] * x).sum() for i, x in zip(b, lp)) / ntok).backward()
        torch.cuda.synchronize()
        dt, mem = time.time() - t0, torch.cuda.max_memory_allocated() / 2**30
        g = torch.cat([p.grad.float().flatten() for p in params])
        return g, dt, mem

    def run_grad(fn):
        for p in params:
            p.grad = None
        fn()
        return torch.cat([p.grad.float().flatten() for p in params])

    def old_grad(mb):
        def f():
            for i in range(0, len(ctxs), mb):
                lp = old_logps(ctxs[i:i + mb], comps[i:i + mb], True, mb)
                (sum((w * x).sum() for w, x in zip(W[i:i + mb], lp)) / ntok).backward()
        return f

    def new_grad():
        for bb in fastlp.token_batches([len(c) + len(d) for c, d in zip(ctxs, comps)], a.budget):
            lp = fastlp.logps(model, tok.pad_token_id, [ctxs[i] for i in bb], [comps[i] for i in bb], True, a.budget)
            (sum((W[i] * x).sum() for i, x in zip(bb, lp)) / ntok).backward()

    def stats(x, ref):
        d = torch.cat([(u - v).abs() for u, v in zip(x, ref)])
        return f"max {float(d.max()):.2e} mean {float(d.mean()):.2e}"

    if a.time_budgets:
        for bud in map(int, a.time_budgets.split(",")):
            a.budget = bud
            for p in params:
                p.grad = None
            torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
            new_grad(); torch.cuda.synchronize()
            print(f"TIME budget={bud}: {time.time() - t0:.1f}s peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
        return
    with torch.no_grad():
        ref = old_logps(ctxs, comps, False, 1)                                  # one sample at a time: no padding
        lo = old_logps(ctxs, comps, False, a.old_mb)
        ln = fastlp.logps(model, tok.pad_token_id, ctxs, comps, False, 2 * a.budget)
        ln1 = fastlp.logps(model, tok.pad_token_id, ctxs, comps, False, 1)       # new path, one sample at a time
    print(f"logp vs unpadded reference: old mb={a.old_mb} [{stats(lo, ref)}] | new packed [{stats(ln, ref)}] | "
          f"new unpadded [{stats(ln1, ref)}]")
    cg = lambda u, v: float(u @ v / (u.norm() * v.norm()))
    g_ref = run_grad(old_grad(1))
    g_old = run_grad(old_grad(a.old_mb))
    g_new = run_grad(new_grad)
    print(f"grad cos vs unpadded reference: old mb={a.old_mb} {cg(g_old, g_ref):.6f} | new {cg(g_new, g_ref):.6f} | "
          f"old vs new {cg(g_old, g_new):.6f}")

    def timed(fn):
        for p in params:
            p.grad = None
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        fn(); torch.cuda.synchronize()
        return time.time() - t0, torch.cuda.max_memory_allocated() / 2**30
    t_old, m_old = timed(old_grad(a.old_mb))
    t_new, m_new = timed(new_grad)
    print(f"fwd+bwd {len(ctxs)} samples ({ntok} completion tokens): old mb={a.old_mb} {t_old:.1f}s peak {m_old:.1f} GB | "
          f"new budget={a.budget} {t_new:.1f}s peak {m_new:.1f} GB | speed-up x{t_old / t_new:.2f}")


if __name__ == "__main__":
    main()
