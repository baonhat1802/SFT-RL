"""B1: SFT with LoRA on a track's reference solutions.

    PYTHONPATH=code python -m train.sft --track luffy8k --seed 1 --hub-repo <user>/capstone-luffy8k-sft-seed1

Loss is on the completion tokens only (prompt-completion format, see data.sft_text_dataset). At the end the adapter is
saved, and (unless --no-merge) a bf16 merged model is saved too: B3 (SFT->RL) starts RL from it with a fresh adapter.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=["luffy8k", "curated"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--init-model", default=None, help="default: Qwen/Qwen2.5-3B")
    ap.add_argument("--output-dir", default=None, help="default: runs/<track>_sft_seed<s>")
    ap.add_argument("--hub-repo", default=None, help="private HF repo for checkpoints; enables resume")
    ap.add_argument("--per-device-batch", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=-1, help="smoke tests only")
    ap.add_argument("--save-steps", type=int, default=25, help="~7 min of SFT on track A between checkpoints")
    ap.add_argument("--limit", type=int, default=None, help="use the first N rows (smoke tests only)")
    ap.add_argument("--no-unsloth", action="store_true", help="CPU/test path: transformers + peft")
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--report-to", default="none")
    # overrides used by code/beat (defaults keep the baseline config of common.py)
    ap.add_argument("--train-file", default=None, help="parquet with track columns; default tracks/<track>/train.parquet")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--lora-r", type=int, default=None)
    ap.add_argument("--lora-alpha", type=int, default=None)
    ap.add_argument("--run-name", default=None)
    return ap.parse_args()


def main():
    a = build_args()
    from .common import BASE_MODEL, MAX_SEQ_TOKENS, SFT, run_name
    from .models import load_plain, load_unsloth

    init = a.init_model or BASE_MODEL
    if a.no_unsloth:
        model, tok = load_plain(init, a.track, a.seed)
    else:
        lora = {k: v for k, v in (("r", a.lora_r), ("lora_alpha", a.lora_alpha)) if v is not None}
        model, tok = load_unsloth(init, a.track, a.seed, fast_inference=False, lora=lora)

    from trl import SFTConfig, SFTTrainer   # after unsloth

    from .data import sft_text_dataset
    from .hub import HubCheckpointCallback, ensure_repo, fetch_latest, upload_dir

    name = a.run_name or run_name(a.track, "sft", a.seed)
    out = Path(a.output_dir or f"runs/{name}")
    if SFT["effective_batch"] % a.per_device_batch:
        raise SystemExit("--per-device-batch must divide the effective batch (64)")

    cfg = SFTConfig(
        output_dir=str(out),
        run_name=name,
        seed=a.seed,
        num_train_epochs=SFT["num_train_epochs"],
        max_steps=a.max_steps,
        learning_rate=a.lr or SFT["learning_rate"],
        lr_scheduler_type=SFT["lr_scheduler_type"],
        lr_scheduler_kwargs=SFT["lr_scheduler_kwargs"],
        warmup_ratio=SFT["warmup_ratio"],
        weight_decay=SFT["weight_decay"],
        per_device_train_batch_size=a.per_device_batch,
        gradient_accumulation_steps=SFT["effective_batch"] // a.per_device_batch,
        max_length=MAX_SEQ_TOKENS,
        completion_only_loss=True,
        packing=False,
        bf16=not a.no_unsloth,
        logging_steps=1,
        save_strategy="steps",
        save_steps=a.save_steps,
        save_total_limit=2,
        report_to=a.report_to,
        optim="adamw_torch",            # no CPU-offloaded optimizer (SFT-then-RL bug 1)
    )
    callbacks, resume = [], None
    if a.hub_repo:
        ensure_repo(a.hub_repo)
        callbacks.append(HubCheckpointCallback(a.hub_repo))
        resume = fetch_latest(a.hub_repo, str(out))
        print(f"RESUME from {resume}" if resume else "RESUME none: starting fresh", flush=True)
    trainer = SFTTrainer(model=model, args=cfg, processing_class=tok,
                         train_dataset=sft_text_dataset(tok, a.track, a.limit, a.train_file), callbacks=callbacks)
    trainer.train(resume_from_checkpoint=resume)

    adapter_dir = out / "final_adapter"
    trainer.save_model(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    if a.hub_repo:
        upload_dir(a.hub_repo, str(adapter_dir), "final_adapter")
    if not a.no_merge:
        merged = out / "merged_bf16"
        if a.no_unsloth:
            model.merge_and_unload().save_pretrained(str(merged))
            tok.save_pretrained(str(merged))
        else:
            model.save_pretrained_merged(str(merged), tok, save_method="merged_16bit")
        if a.hub_repo:
            upload_dir(a.hub_repo, str(merged), "merged_bf16")
    print(f"done: {adapter_dir}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
