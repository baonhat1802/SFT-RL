"""Gradient-accumulation check with the real Qwen2.5-3B through Unsloth (smoke step grad_real).

    PYTHONPATH=code python -m train.check_real

One SGD step (lr = 1) gives update = -gradient, which is linear in the loss. With 8 examples split into 4 micro-batches
m of 2, write g_m for the update from micro-batch m alone (loss averaged over its completion tokens T_m). Then
    correct accumulation (loss averaged over all T = sum T_m tokens):  sum_m (T_m / T) g_m
    buggy accumulation (average of per-micro-batch averages):          (1/4) sum_m g_m
We measure g_m directly, predict both, and check which one the real accumulated run matches. The micro-batches are
deliberately unbalanced (4 shortest examples first, 4 longest last) so the two predictions differ clearly.
Unsloth runs are not bit-reproducible, so the same configuration is also run twice to measure the noise floor.
Everything in fp32, so bf16 rounding does not enter.
"""
import unsloth  # noqa: F401  (must be imported before trl/transformers)

import gc

import torch

from .checks import _one_step_update
from .common import BASE_MODEL
from .data import sft_text_dataset
from .models import load_unsloth


def rel(a, b) -> float:
    return float((a - b).norm() / a.norm().clamp_min(1e-12))


def main():
    tok = load_unsloth(BASE_MODEL, "curated", seed=1, fast_inference=False)[1]
    ds = sft_text_dataset(tok, "curated", limit=400)

    def n_completion_tokens(r):
        return len(tok(text=r["prompt"] + r["completion"])["input_ids"]) - len(tok(text=r["prompt"])["input_ids"])

    lens = [n_completion_tokens(r) for r in ds]
    order = sorted(range(len(ds)), key=lambda i: lens[i])
    idx = order[:4] + order[-4:]                    # micro-batches: (s1,s2) (s3,s4) (l1,l2) (l3,l4)
    pick = ds.select(idx)
    T = [lens[idx[2 * m]] + lens[idx[2 * m + 1]] for m in range(4)]
    print(f"INFO completion tokens per micro-batch: {T}", flush=True)

    def update(data, per_device, accum):
        out = _one_step_update(lambda: load_unsloth(BASE_MODEL, "curated", seed=1, fast_inference=False,
                                                    dtype=torch.float32)[0],
                               tok, data, per_device=per_device, accum=accum, max_length=1024, use_cpu=False,
                               bf16=False)
        gc.collect()                       # many 3B models are loaded in turn: free each before the next
        torch.cuda.empty_cache()
        return out

    full = update(pick, 8, 1)
    full2 = update(pick, 8, 1)
    acc = update(pick, 2, 4)
    g = [update(pick.select([2 * m, 2 * m + 1]), 2, 1) for m in range(4)]
    pred_ok = sum((T[m] / sum(T)) * g[m] for m in range(4))
    pred_bug = sum(g) / 4

    noise = rel(full, full2)
    gap = rel(pred_ok, pred_bug)
    d_ok, d_bug = rel(acc, pred_ok), rel(acc, pred_bug)
    print(f"INFO noise floor (same config twice): {noise:.3e}", flush=True)
    print(f"INFO one batch vs predicted-correct: {rel(full, pred_ok):.3e}", flush=True)
    print(f"INFO predicted-correct vs predicted-bug (separation): {gap:.3e}", flush=True)
    print(f"INFO accumulated vs predicted-correct: {d_ok:.3e} | vs predicted-bug: {d_bug:.3e}", flush=True)

    discriminative = gap > 3 * max(noise, 1e-6)
    ok = discriminative and d_ok < d_bug / 2
    print(f"RESULT accumulated run matches {'correct' if d_ok < d_bug else 'BUGGY'} normalisation "
          f"(separation {gap:.1e} vs noise {noise:.1e}) -> {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
