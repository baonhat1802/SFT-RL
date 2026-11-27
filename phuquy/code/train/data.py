"""Turn code/data/tracks/<track>/train.parquet (code/scripts/make_tracks.py) into TRL datasets.

SFT uses TRL's conversational prompt-completion format: "prompt" = the same chat messages the RL rollouts see, and
"completion" = the reference solution. TRL then trains on the completion tokens only, and the SFT model sees exactly the
prompt it will be evaluated and RL-trained with.
GRPO uses "prompt" plus "ground_truth" (passed to the reward function as a keyword argument).
"""
from __future__ import annotations

import pandas as pd
from datasets import Dataset

from .common import DATA_ROOT, MAX_PROMPT_TOKENS, TRACKS


def load_track(track: str, path: str | None = None) -> pd.DataFrame:
    """path: optional parquet with the same columns (e.g. data made by beat/gen_sft_data.py)."""
    if track not in TRACKS:
        raise ValueError(f"unknown track {track!r}")
    return pd.read_parquet(path or DATA_ROOT / "tracks" / track / "train.parquet")


def _messages(prompt) -> list[dict]:
    return [{"role": m["role"], "content": m["content"]} for m in prompt]


def prompt_fits(tokenizer, messages) -> bool:
    """Prompt within MAX_PROMPT_TOKENS once rendered. One luffy8k problem has a 1,788-token prompt; TRL would truncate
    it (SFT cuts the answer, GRPO cuts the question), so every baseline drops such rows instead."""
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    return len(ids) <= MAX_PROMPT_TOKENS


def sft_dataset(track: str, limit: int | None = None, path: str | None = None) -> Dataset:
    df = load_track(track, path)
    if limit:
        df = df.head(limit)
    return Dataset.from_list([
        {"prompt": _messages(r.prompt), "completion": [{"role": "assistant", "content": r.target[0]["content"]}]}
        for r in df.itertuples()
    ])


def sft_text_dataset(tokenizer, track: str, limit: int | None = None, path: str | None = None) -> Dataset:
    """Prompt-completion dataset rendered to text with the track's chat template (configure_tokenizer first).

    prompt     = template(prompt messages, add_generation_prompt=True), i.e. exactly what RL and eval feed the model
    completion = the rest of template(prompt + assistant answer): the answer plus the track's end-of-answer token
    """
    rows = []
    for ex in sft_dataset(track, limit, path):
        if not prompt_fits(tokenizer, ex["prompt"]):
            continue
        p = tokenizer.apply_chat_template(ex["prompt"], tokenize=False, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(ex["prompt"] + ex["completion"], tokenize=False)
        if not full.startswith(p):
            raise ValueError("chat template: prompt rendering is not a prefix of the full rendering")
        completion = full[len(p):].lstrip()
        # Some reference solutions start with "\n" (160 curated, 6 luffy8k). With the prompt ending in "\n", the two
        # newlines merge into one token and the prompt/completion token boundary shifts; drop the leading whitespace.
        # ChatML (track B) renders the answer as "...<|im_end|>\n": drop the "\n" and append the EOS, giving
        # "...<|im_end|><|endoftext|>" (see common.EOS_TOKEN). Ending on the EOS also stops TRL's add_eos step from
        # appending a second one. Track A's template already ends with the EOS.
        if completion.endswith("<|im_end|>\n"):
            completion = completion[:-1]
        if not completion.endswith(tokenizer.eos_token):
            completion += tokenizer.eos_token
        rows.append({"prompt": p, "completion": completion})
    return Dataset.from_list(rows)


def grpo_dataset(track: str, limit: int | None = None, tokenizer=None) -> Dataset:
    """With a tokenizer, rows whose prompt exceeds MAX_PROMPT_TOKENS are dropped (same rule as SFT)."""
    df = load_track(track)
    if limit:
        df = df.head(limit)
    return Dataset.from_list([
        {"prompt": _messages(r.prompt), "ground_truth": r.reward_model["ground_truth"]}
        for r in df.itertuples()
        if tokenizer is None or prompt_fits(tokenizer, _messages(r.prompt))
    ])
