"""Tests run on CPU (Mac) and again on molab with the GPU stack. `code/` is put on sys.path so `import train` works
(the directory name `code` shadows Python's stdlib module of the same name, so it cannot be a package)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE = "Qwen/Qwen2.5-3B"   # only the tokenizer files are downloaded by these tests


@pytest.fixture(scope="session")
def tokenizer_factory():
    from transformers import AutoTokenizer

    from train.common import configure_tokenizer

    cache = {}

    def make(track):
        if track not in cache:
            cache[track] = configure_tokenizer(AutoTokenizer.from_pretrained(BASE), track)
        return cache[track]

    return make


def tiny_qwen2(vocab_size: int, seed: int = 0):
    """A 2-layer random Qwen2 with the real vocabulary: same code paths as Qwen2.5-3B, runs on CPU in seconds."""
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=vocab_size, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=8192,
                      tie_word_embeddings=True)
    return Qwen2ForCausalLM(cfg)
