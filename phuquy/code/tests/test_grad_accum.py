"""SFT loss normalisation under gradient accumulation (bug 2 of SFT-then-RL), with a tiny Qwen2 + our LoRA.

Runs only where a CUDA GPU exists (molab): no model training on the Mac, not even a tiny one.
"""
import pytest
import torch
from conftest import tiny_qwen2

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="model checks run on the GPU machine only")

from train.checks import check_sft_grad_accum
from train.common import LORA
from train.data import sft_text_dataset


def _lora_factory(vocab_size):
    from peft import LoraConfig, get_peft_model

    def make():
        model = get_peft_model(tiny_qwen2(vocab_size, seed=0), LoraConfig(task_type="CAUSAL_LM", **{**LORA, "r": 8}))
        # LoRA B starts at zero, so A gets no gradient on step 1; give B a fixed non-zero start so both are tested
        import torch
        g = torch.Generator().manual_seed(1)
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.data = torch.randn(p.shape, generator=g) * 0.02
        return model

    return make


def _varied_rows(tok, n=8):
    ds = sft_text_dataset(tok, "curated", limit=400)
    lens = [len(tok(text=r["completion"])["input_ids"]) for r in ds]
    order = sorted(range(len(ds)), key=lambda i: lens[i])
    # alternate very short and very long completions so micro-batches differ a lot in token count
    pick = [order[i // 2] if i % 2 == 0 else order[-1 - i // 2] for i in range(n)]
    sub = ds.select(pick)
    sub_lens = [lens[i] for i in pick]
    assert max(sub_lens) > 4 * min(sub_lens)
    return sub


def test_sft_grad_accum_matches_full_batch(tokenizer_factory):
    tok = tokenizer_factory("curated")
    # fp32 on the GPU (bf16=False), so the tolerance can be tight: bf16 rounding alone gives ~3e-3
    rel = check_sft_grad_accum(_lora_factory(len(tok)), tok, _varied_rows(tok), n=8, micro=2, use_cpu=False, tol=1e-4,
                               bf16=False)
    print(f"relative difference {rel:.2e}")
