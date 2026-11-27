"""Load the base (or SFT-merged) model with a fresh LoRA adapter.

On the GPU (molab) this goes through Unsloth; `unsloth` must be imported before trl/transformers, which is why the
entry scripts import this module first. On a CPU-only machine (unit tests) `load_plain` gives the same LoRA through peft.
"""
from __future__ import annotations

from .common import (EOS_TOKEN, LORA, MAX_SEQ_TOKENS, VLLM_MAX_NUM_BATCHED_TOKENS, VLLM_MAX_NUM_SEQS,
                     configure_tokenizer)


def eos_ids(tokenizer, track: str) -> list[int]:
    """Token ids that end an answer for this track: the track's EOS, plus <|endoftext|> as a safety stop."""
    ids = {tokenizer.convert_tokens_to_ids(EOS_TOKEN[track]), tokenizer.convert_tokens_to_ids("<|endoftext|>")}
    return sorted(i for i in ids if i is not None and i >= 0)


def load_unsloth(init_model: str, track: str, seed: int, fast_inference: bool, gpu_memory_utilization: float = 0.6,
                 dtype=None, lora: dict | None = None):
    """lora: optional overrides of common.LORA (e.g. {"r": 32, "lora_alpha": 64})."""
    cfg = {**LORA, **(lora or {})}
    import unsloth  # noqa: F401  (patches trl/transformers; must come first)
    from unsloth import FastLanguageModel

    vllm_kw = {}
    if fast_inference:
        # forwarded by Unsloth to load_vllm; without them Unsloth picks 128 parallel sequences for this memory size
        vllm_kw = dict(max_num_seqs=VLLM_MAX_NUM_SEQS, max_num_batched_tokens=VLLM_MAX_NUM_BATCHED_TOKENS)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=init_model,
        max_seq_length=MAX_SEQ_TOKENS,
        load_in_4bit=False,                 # base weights stay bf16 for every baseline
        fast_inference=fast_inference,      # vLLM inside the process, shares weights with training
        max_lora_rank=cfg["r"],
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,                        # None = bf16 on this GPU; float32 only for the grad-accum check
        **vllm_kw,
    )
    configure_tokenizer(tokenizer, track)
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias=cfg["bias"],
        target_modules=cfg["target_modules"],
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )
    model.generation_config.eos_token_id = eos_ids(tokenizer, track)
    return model, tokenizer


def load_plain(init_model: str, track: str, seed: int, dtype=None):
    """CPU / test path: transformers + peft, same LoRA settings, no Unsloth, no vLLM."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    tokenizer = configure_tokenizer(AutoTokenizer.from_pretrained(init_model), track)
    model = AutoModelForCausalLM.from_pretrained(init_model, torch_dtype=dtype or torch.float32)
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **LORA))
    model.generation_config.eos_token_id = eos_ids(tokenizer, track)
    return model, tokenizer
