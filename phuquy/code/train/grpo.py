"""B2 (GRPO from the base model) and B3 (SFT->RL: GRPO from the merged B1 model), with LoRA.

    PYTHONPATH=code python -m train.grpo --track luffy8k --baseline grpo --seed 1 --hub-repo <user>/capstone-luffy8k-grpo-seed1
    PYTHONPATH=code python -m train.grpo --track luffy8k --baseline sft_rl --init-model <B1 merged_bf16 dir or repo> ...

Batch layout matches verl (sft_then_rl/exp_scripts/verl_rl.sh): every rollout samples 128 prompts x 8 completions =
1,024 completions and makes 2 optimizer steps of 512 completions each. In TRL this is generation_batch_size=1024 with
per_device_train_batch_size x gradient_accumulation_steps = 512; TRL derives steps_per_generation (in micro-batches).

Known difference from verl: with loss_type="dapo", TRL divides by the completion tokens of the whole 1,024-completion
generation batch, verl by those of the 512-completion mini-batch, so TRL's gradient is about half as large. Adam is
nearly invariant to a constant gradient scale, so the effect is small; it is the same for every baseline.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=["luffy8k", "curated"])
    ap.add_argument("--baseline", required=True, choices=["grpo", "sft_rl"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--init-model", default=None, help="grpo: default Qwen/Qwen2.5-3B; sft_rl: B1 merged_bf16")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--hub-repo", default=None, help="private HF repo for checkpoints; enables resume")
    # The full-batch out-of-memory was in Unsloth's log-prob chunking (fixed by unsloth_grpo_mini_batch), not in the
    # update micro-batch; 2-completion micro-batches then wasted ~135 s per rollout, so back to 8 (common.py).
    ap.add_argument("--per-device-batch", type=int, default=None, help="completions per micro-batch (default common)")
    ap.add_argument("--gpu-mem", type=float, default=0.5, help="vLLM share of GPU memory (Unsloth fast_inference)")
    ap.add_argument("--max-steps", type=int, default=-1, help="optimizer steps; smoke tests only")
    ap.add_argument("--save-steps", type=int, default=None, help="default 50 optimizer steps (= 25 rollouts)")
    ap.add_argument("--limit", type=int, default=None, help="use the first N prompts (smoke tests only)")
    ap.add_argument("--prompts-per-rollout", type=int, default=None, help="smoke tests only (default 128)")
    ap.add_argument("--no-unsloth", action="store_true", help="CPU/test path: transformers + peft, no vLLM")
    ap.add_argument("--report-to", default="none")
    # overrides used by code/beat (defaults keep the baseline config of common.py)
    ap.add_argument("--lora-r", type=int, default=None)
    ap.add_argument("--lora-alpha", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-completion", type=int, default=None, help="default common.MAX_COMPLETION_TOKENS (4,096)")
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--run-name", default=None)
    a = ap.parse_args()
    if a.baseline == "sft_rl" and not a.init_model:
        ap.error("--baseline sft_rl needs --init-model (the merged B1 model)")
    return a


def grpo_config(output_dir: str, run_name: str, seed: int, per_device_batch: int, max_steps: int = -1,
                prompts_per_rollout: int | None = None, use_vllm: bool = True, stop_ids=(), report_to="none",
                stop_text=(), max_completion: int | None = None, **overrides):
    """GRPOConfig for every RL baseline. Import trl only after unsloth (callers do)."""
    from trl import GRPOConfig

    from .common import GRPO, MAX_COMPLETION_TOKENS, MAX_PROMPT_TOKENS

    ppr = prompts_per_rollout or GRPO["prompts_per_rollout"]
    gen_batch = ppr * GRPO["num_generations"]                    # 1,024
    per_step = gen_batch // GRPO["steps_per_generation"]          # 512 completions per optimizer step
    if per_step % per_device_batch:
        raise ValueError(f"per_device_batch must divide {per_step}")
    kw = dict(
        output_dir=output_dir,
        run_name=run_name,
        seed=seed,
        num_train_epochs=GRPO["num_train_epochs"],
        max_steps=max_steps,
        learning_rate=GRPO["learning_rate"],
        lr_scheduler_type=GRPO["lr_scheduler_type"],
        warmup_ratio=0.0,
        weight_decay=0.0,
        optim="adamw_torch",
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=per_step // per_device_batch,
        generation_batch_size=gen_batch,
        num_generations=GRPO["num_generations"],
        num_iterations=1,
        beta=GRPO["beta"],
        epsilon=GRPO["epsilon"],
        epsilon_high=GRPO["epsilon_high"],
        scale_rewards=GRPO["scale_rewards"],
        loss_type=GRPO["loss_type"],
        temperature=GRPO["temperature"],
        top_p=GRPO["top_p"],
        mask_truncated_completions=GRPO["mask_truncated_completions"],
        max_prompt_length=MAX_PROMPT_TOKENS,
        max_completion_length=max_completion or MAX_COMPLETION_TOKENS,
        use_vllm=use_vllm,
        # vLLM stops only on what it is told: track B must stop on <|im_end|>, not the base model's <|endoftext|>
        generation_kwargs={"stop_token_ids": list(stop_ids)} if use_vllm else {"eos_token_id": list(stop_ids)},
        bf16=use_vllm,
        logging_steps=1,
        save_strategy="steps",
        save_steps=GRPO["save_every_optimizer_steps"],
        save_total_limit=2,
        report_to=report_to,
        log_completions=False,
    )
    import inspect
    if use_vllm:
        if "vllm_sampling_params" in inspect.signature(GRPOConfig.__init__).parameters:
            # Unsloth builds vLLM's SamplingParams itself and ignores TRL's generation_kwargs: track B then never stopped
            # on <|im_end|> (smoke: 100% of completions hit 4,096 tokens). Unsloth takes stop tokens here instead.
            # No fixed seed (it could make the 8 completions of a prompt identical) and no min_p (keeps verl's
            # temperature 1.0 / top_p 1.0 sampling). The stop token stays in the output so the model learns to stop.
            from vllm import SamplingParams
            kw["vllm_sampling_params"] = SamplingParams(
                temperature=GRPO["temperature"], top_p=GRPO["top_p"], top_k=-1, min_p=0.0,
                stop_token_ids=list(stop_ids), stop=list(stop_text), include_stop_str_in_output=True,
            )
    if use_vllm and "unsloth_grpo_mini_batch" in inspect.signature(GRPOConfig.__init__).parameters:
        # Unsloth computes log-probs over the whole 1,024-completion generation batch in chunks whose count it
        # autotunes from GPU memory, without subtracting what vLLM holds: it picked ~24 completions x 5,120 tokens per
        # chunk and ran out of memory. Set the chunk count by hand (common.GRPO_LOGPROB_CHUNKS). Memory/speed only;
        # the values computed are the same.
        from .common import GRPO_LOGPROB_CHUNKS
        kw["unsloth_grpo_mini_batch"] = min(GRPO_LOGPROB_CHUNKS, gen_batch)
    kw.update(overrides)
    return GRPOConfig(**kw)


def main():
    a = build_args()
    from .common import BASE_MODEL, run_name
    from .models import eos_ids, load_plain, load_unsloth

    init = a.init_model or BASE_MODEL
    if a.no_unsloth:
        model, tok = load_plain(init, a.track, a.seed)
    else:
        lora = {k: v for k, v in (("r", a.lora_r), ("lora_alpha", a.lora_alpha)) if v is not None}
        model, tok = load_unsloth(init, a.track, a.seed, fast_inference=True, gpu_memory_utilization=a.gpu_mem,
                                  lora=lora)

    from trl import GRPOTrainer   # after unsloth

    from .data import grpo_dataset
    from .hub import HubCheckpointCallback, ensure_repo, fetch_latest
    from .rewards import correctness_reward

    name = a.run_name or run_name(a.track, a.baseline, a.seed)
    out = Path(a.output_dir or f"runs/{name}")
    if a.per_device_batch is None:
        from .common import GRPO_MICRO_BATCH
        a.per_device_batch = GRPO_MICRO_BATCH
    cfg = grpo_config(output_dir=str(out), run_name=name, seed=a.seed, per_device_batch=a.per_device_batch,
                      max_steps=a.max_steps, prompts_per_rollout=a.prompts_per_rollout,
                      use_vllm=not a.no_unsloth, stop_ids=eos_ids(tok, a.track), report_to=a.report_to,
                      stop_text=[tok.eos_token], max_completion=a.max_completion,
                      **({"save_steps": a.save_steps} if a.save_steps else {}),
                      **({"learning_rate": a.lr} if a.lr else {}),
                      **({"num_train_epochs": a.epochs} if a.epochs else {}))
    callbacks, resume = [], None
    if a.hub_repo:
        ensure_repo(a.hub_repo)
        callbacks.append(HubCheckpointCallback(a.hub_repo))
        resume = fetch_latest(a.hub_repo, str(out))
        print(f"RESUME from {resume}" if resume else "RESUME none: starting fresh", flush=True)
    trainer = GRPOTrainer(model=model, processing_class=tok, reward_funcs=[correctness_reward], args=cfg,
                          train_dataset=grpo_dataset(a.track, a.limit, tok), callbacks=callbacks)
    if os.environ.get("CAPSTONE_PROFILE", "0") != "0":
        from .profiling import instrument
        instrument(trainer)
    trainer.train(resume_from_checkpoint=resume)

    adapter_dir = out / "final_adapter"
    trainer.save_model(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    if a.hub_repo:
        from .hub import upload_dir
        upload_dir(a.hub_repo, str(adapter_dir), "final_adapter")
    print(f"done: {adapter_dir}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
