"""The SFT examples must show the model exactly the prompt it gets in RL and eval, and train it to stop."""
import pytest

from train.data import grpo_dataset, sft_text_dataset

TRACKS = ["luffy8k", "curated"]


@pytest.mark.parametrize("track", TRACKS)
def test_sft_prompt_matches_rl_prompt(track, tokenizer_factory):
    tok = tokenizer_factory(track)
    sft = sft_text_dataset(tok, track, limit=20)
    rl = grpo_dataset(track, limit=20, tokenizer=tok)
    for s, r in zip(sft, rl):
        assert s["prompt"] == tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True)


@pytest.mark.parametrize("track", TRACKS)
def test_completion_ends_on_track_eos_once(track, tokenizer_factory):
    tok = tokenizer_factory(track)
    for ex in sft_text_dataset(tok, track, limit=50):
        c = ex["completion"]
        assert c.endswith(tok.eos_token)
        assert c.count(tok.eos_token) == 1, "end-of-answer token appears more than once"


def test_track_a_prompt_shape(tokenizer_factory):
    tok = tokenizer_factory("luffy8k")
    p = sft_text_dataset(tok, "luffy8k", limit=1)[0]["prompt"]
    assert p.startswith("Your task is to follow"), "bos_token must render as nothing, not 'None'"
    assert p.endswith("Assistant: <think>\n")


def test_track_b_prompt_shape(tokenizer_factory):
    tok = tokenizer_factory("curated")
    p = sft_text_dataset(tok, "curated", limit=1)[0]["prompt"]
    assert p.startswith("<|im_start|>system\nYou are a helpful assistant.<|im_end|>")
    assert "Please reason step by step, and put your final answer within \\boxed{}." in p
    assert p.endswith("<|im_start|>assistant\n")


@pytest.mark.parametrize("track", TRACKS)
def test_trl_tokenization_masks_only_the_completion(track, tokenizer_factory):
    """Mirror TRL 0.22's prompt-completion tokenisation: prompt ids must be a prefix, completion ends with EOS id."""
    tok = tokenizer_factory(track)
    for ex in sft_text_dataset(tok, track):          # every row: a few rows start with "\n" and used to break this
        p_ids = tok(text=ex["prompt"])["input_ids"]
        pc_ids = tok(text=ex["prompt"] + ex["completion"])["input_ids"]
        assert pc_ids[:len(p_ids)] == p_ids
        assert pc_ids[-1] == tok.eos_token_id
        assert len(p_ids) <= 1024                     # MAX_PROMPT_TOKENS: longer prompts are dropped
        assert len(pc_ids) - len(p_ids) <= 4096 + 8     # reference solutions were filtered to <= 4,096 tokens


@pytest.mark.parametrize("track", TRACKS)
def test_sft_and_grpo_keep_the_same_rows(track, tokenizer_factory):
    tok = tokenizer_factory(track)
    assert len(sft_text_dataset(tok, track)) == len(grpo_dataset(track, tokenizer=tok))


@pytest.mark.parametrize("track", TRACKS)
def test_template_set_even_if_tokenizer_has_none(track, tokenizer_factory):
    """Unsloth's unsloth/Qwen2.5-3B tokenizer ships without a chat template; configure_tokenizer must still set it."""
    import copy

    from train.common import configure_tokenizer

    ref = tokenizer_factory(track)
    bare = copy.deepcopy(ref)
    bare.chat_template = None
    configure_tokenizer(bare, track)
    msgs = grpo_dataset(track, limit=1)[0]["prompt"]
    assert bare.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) == \
        ref.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def test_track_b_answer_ends_with_turn_end_then_eos(tokenizer_factory):
    tok = tokenizer_factory("curated")
    assert tok.eos_token == "<|endoftext|>" and tok.pad_token != tok.eos_token
    for ex in sft_text_dataset(tok, "curated", limit=50):
        assert ex["completion"].endswith("<|im_end|><|endoftext|>")
