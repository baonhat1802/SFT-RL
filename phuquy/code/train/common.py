"""Constants shared by every baseline. Change a value here and it changes for all runs; that is the point.

Every number below comes from PLAN_baselines_qwen25_3b.md section 4. Keep the two in sync.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Data root: code/data on the Mac; on molab the Kaggle download dir (set CAPSTONE_DATA).
DATA_ROOT = Path(os.environ.get("CAPSTONE_DATA", REPO_ROOT / "code" / "data"))

BASE_MODEL = "Qwen/Qwen2.5-3B"
TRACKS = ("luffy8k", "curated")

MAX_PROMPT_TOKENS = 1024
MAX_COMPLETION_TOKENS = 4096
MAX_SEQ_TOKENS = MAX_PROMPT_TOKENS + MAX_COMPLETION_TOKENS

LORA = dict(
    r=64,
    lora_alpha=64,
    lora_dropout=0.0,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# SFT: verl full fine-tune used lr 5e-5; LoRA uses x10 (LoRA Without Regret).
SFT = dict(
    num_train_epochs=3,
    learning_rate=5e-4,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr_rate": 0.1},
    warmup_ratio=0.1,
    effective_batch=64,          # sequences per optimizer step
    weight_decay=0.0,
)

# GRPO: one rollout = 128 prompts x 8 completions = 1,024 completions, split into 2 optimizer steps of 512
# (verl: train_batch_size=128, ppo_mini_batch_size=64, ppo_epochs=1). One epoch over track A's 7,999 prompts is
# ~62 rollouts = ~125 optimizer steps (track B: ~67 rollouts, ~133 steps).
GRPO = dict(
    # 1 epoch (decided 2026-11-12, option C): one full-size rollout takes ~317 s on molab, so 2 epochs would be ~11 h
    # per run. Same for every baseline; the algorithm (2 updates per rollout, as verl) is unchanged.
    num_train_epochs=1,
    learning_rate=1e-5,           # verl full fine-tune used 1e-6; x10 for LoRA
    lr_scheduler_type="constant",
    num_generations=8,
    prompts_per_rollout=128,
    steps_per_generation=2,
    beta=0.0,                     # no KL
    epsilon=0.2,
    epsilon_high=0.28,
    scale_rewards="none",         # verl norm_adv_by_std_in_grpo=false
    loss_type="dapo",             # token-level normalisation over the batch (verl token-mean)
    temperature=1.0,
    top_p=1.0,
    mask_truncated_completions=False,
    # a checkpoint (~490 MB: LoRA + optimizer) uploads to the HF Hub in ~10 s (measured 47 MB/s on molab), so save
    # often: a session that dies (Mac asleep, molab timeout) loses at most 10 optimizer steps (~20-35 min)
    save_every_optimizer_steps=10,
)

# Chat templates. Track A uses the LUFFY / SFT-then-RL template (system prompt baked in, answer opens with <think>);
# it ignores "system" messages, so passing the dataset's system message is harmless. Track B uses Qwen's ChatML with the
# SimpleRL system prompt that is already in the data.
# Copied verbatim from sft_then_rl/models/README.md (chat_template field) by code/train/luffy_chat_template.jinja.
LUFFY_CHAT_TEMPLATE = (Path(__file__).with_name("luffy_chat_template.jinja")).read_text()
# Qwen2.5's own ChatML template, copied verbatim from Qwen/Qwen2.5-3B tokenizer_config.json. Set explicitly because
# Unsloth loads its unsloth/Qwen2.5-3B mirror, whose base-model tokenizer ships without a chat template.
QWEN_CHAT_TEMPLATE = (Path(__file__).with_name("qwen25_chat_template.jinja")).read_text()

# End-of-answer token: <|endoftext|> for BOTH tracks. TRL recognises a single EOS id when it masks completions. In
# the GRPO smoke the base model, prompted in ChatML (track B), ended 127/128 answers with <|endoftext|> and none with
# <|im_end|>; with EOS = <|im_end|> TRL saw no end, counted the padding as completion (length 4,096, "100% truncated")
# and would have trained on pad tokens. So track B's reference answers end "...<|im_end|><|endoftext|>" (ChatML turn
# end, then EOS), RL stops on <|endoftext|>, and evaluation stops on either.
EOS_TOKEN = {"luffy8k": "<|endoftext|>", "curated": "<|endoftext|>"}


# Throughput settings (no effect on what is computed). Measured on molab (smoke step `profile`, one 1,024-completion
# rollout = 410 s): vLLM ran only 128 sequences at a time (Unsloth's default for this memory) -> 117 s generation;
# 2-completion micro-batches left the GPU idle -> ~135 s of updates; 4-completion log-prob chunks -> ~70 s.
VLLM_MAX_NUM_SEQS = 512
VLLM_MAX_NUM_BATCHED_TOKENS = 16384
GRPO_MICRO_BATCH = 8            # completions per micro-batch in the update
GRPO_LOGPROB_CHUNKS = 128       # Unsloth log-prob chunks per 1,024-completion rollout (8 completions each)


def configure_tokenizer(tokenizer, track: str):
    """Set the track's chat template and end-of-answer token. Call before any SFT/GRPO/eval use."""
    if track not in TRACKS:
        raise ValueError(f"unknown track {track!r}")
    tokenizer.chat_template = LUFFY_CHAT_TEMPLATE if track == "luffy8k" else QWEN_CHAT_TEMPLATE
    tokenizer.eos_token = EOS_TOKEN[track]
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.pad_token = "<|fim_pad|>"          # never the EOS, or the EOS would be masked out as padding
    return tokenizer


def run_name(track: str, baseline: str, seed: int) -> str:
    return f"{track}_{baseline}_seed{seed}"
