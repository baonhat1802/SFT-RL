"""Checks run before any long job (PLAN section 2.4). Usable on CPU with a tiny model and on molab with the real one.

check_sft_grad_accum: one SGD step (lr=1) on the same examples must give the same update whether they come as one
batch of N or as N/k micro-batches accumulated k times. If the loss were averaged per micro-batch instead of per token
over the whole accumulated batch (the OpenRLHF bug in SFT-then-RL, arXiv 2604.23747), micro-batches with few completion
tokens would weigh as much as long ones and the two updates would differ.
"""
from __future__ import annotations

import tempfile


def _one_step_update(model_factory, tokenizer, dataset, per_device: int, accum: int, max_length: int, use_cpu: bool,
                     bf16: bool):
    import torch
    from torch.utils.data import SequentialSampler
    from trl import SFTConfig, SFTTrainer

    model = model_factory()
    before = {n: p.detach().float().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    with tempfile.TemporaryDirectory() as tmp:
        cfg = SFTConfig(output_dir=tmp, per_device_train_batch_size=per_device, gradient_accumulation_steps=accum,
                        max_steps=1, learning_rate=1.0, optim="sgd", lr_scheduler_type="constant", warmup_ratio=0.0,
                        weight_decay=0.0, max_grad_norm=0.0, completion_only_loss=True, max_length=max_length,
                        packing=False, report_to="none", save_strategy="no", logging_strategy="no",
                        use_cpu=use_cpu, bf16=bf16, seed=0, data_seed=0)
        trainer = SFTTrainer(model=model, args=cfg, processing_class=tokenizer, train_dataset=dataset)
        trainer._get_train_sampler = lambda *a, **k: SequentialSampler(trainer.train_dataset)
        trainer.train()
    after = {n: p.detach().float().cpu() for n, p in model.named_parameters() if p.requires_grad}  # trainer moved it to GPU
    return torch.cat([(after[n] - before[n]).flatten() for n in before])


def check_sft_grad_accum(model_factory, tokenizer, dataset, n: int = 8, micro: int = 2, max_length: int = 1024,
                         use_cpu: bool = True, tol: float = 1e-3, bf16: bool | None = None) -> float:
    """Return the relative difference between the two updates; raise if it exceeds tol.

    model_factory() must return a fresh, identically initialised model each call. `dataset` is a prompt-completion
    dataset (data.sft_text_dataset) with at least n rows of clearly different completion lengths.
    """
    ds = dataset.select(range(n))
    bf16 = (not use_cpu) if bf16 is None else bf16
    # bf16 rounding alone moves the update by ~1e-3 relative; a per-micro-batch loss average with these length
    # differences moves it by tens of percent. So: tol ~1e-4 in fp32, ~2e-2 in bf16.
    full = _one_step_update(model_factory, tokenizer, ds, per_device=n, accum=1, max_length=max_length, use_cpu=use_cpu,
                            bf16=bf16)
    acc = _one_step_update(model_factory, tokenizer, ds, per_device=micro, accum=n // micro, max_length=max_length,
                           use_cpu=use_cpu, bf16=bf16)
    rel = float((full - acc).norm() / full.norm().clamp_min(1e-12))
    if rel > tol:
        raise AssertionError(f"gradient accumulation changes the update: relative difference {rel:.2e} > {tol:.0e}")
    return rel
