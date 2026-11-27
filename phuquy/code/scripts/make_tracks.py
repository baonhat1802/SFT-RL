"""Build the two training tracks for the Qwen2.5-3B baselines (PLAN_baselines_qwen25_3b.md).

Track A "luffy8k": 8,000 problems sampled (seed 2026) from Elliott/Openr1-Math-46k-8192, keeping only problems whose
  DeepSeek-R1 trace is <= 4,096 Qwen2.5 tokens AND scores 1.0 under reward.py. LUFFY filtered with Math-Verify, which
  accepts a trace that boxes only the last of several required values (gold "7,18,29,40", box "40"); reward.py reads
  the last box only and rejects it (3.3% of traces). Long R1-style solutions, LUFFY prompt (system prompt + <think>).
Track B "curated": the team's BaoNhat/bridge-math-curated-teachers (8,519 MATH level 3-5 problems), SFT solutions from
  the format-fixed copy (code/scripts/fix_curated_sft.py). Short GPT-6 Luna solutions, SimpleRL prompt.

Both are written in one verl-ready schema (one row per problem):
  prompt        chat messages for RL rollouts
  reward_model  {"ground_truth": ..., "style": "rule"}; score with reward.py::compute_score for both tracks
  target        [{"role": "assistant", "content": <solution>}]  off-policy trace for LUFFY / SRFT / ReLIFT
  messages      [user, assistant] for SFT
  extra_info    {"index", "question", "answer", "source_id"}
Output: code/data/tracks/{luffy8k,curated}/train.parquet
Run from capstone/: ~/miniconda3/envs/capstone/bin/python code/scripts/make_tracks.py
"""
import json
import os

import sys

import numpy as np
import pandas as pd
from tokenizers import Tokenizer

sys.path.insert(0, "code/data/bridge_curated_teachers")
from reward import compute_score  # noqa: E402

SEED = 2026
N_A = 8000
MAX_TRACE_TOKENS = 4096
TOKENIZER = ".work/tok/qwen25_3b_tokenizer.json"   # Qwen/Qwen2.5-3B; identical to Qwen2.5-Math's
OUT = "code/data/tracks"


def user_text(prompt) -> str:
    return [m for m in prompt if m["role"] == "user"][-1]["content"]


def track_a(tok: Tokenizer) -> pd.DataFrame:
    d = pd.read_parquet("code/data/sft_then_rl/openr1_46k_8192.parquet")
    # same preprocessing as sft_then_rl/data/preprocess_openr1.py: "<think>\n" lives in the chat template
    traces = [t[0]["content"] for t in d["target"]]
    traces = [t[len("<think>\n"):] if t.startswith("<think>\n") else t for t in traces]
    n_tok = np.array([len(e.ids) for e in tok.encode_batch(traces)])
    short = np.flatnonzero(n_tok <= MAX_TRACE_TOKENS)
    gts = [r["ground_truth"] for r in d["reward_model"]]
    ok = np.array([compute_score(None, "<think>\n" + traces[i], gts[i]) == 1.0 for i in short])
    pool = short[ok]
    print(f"track A: <= {MAX_TRACE_TOKENS} tokens = {len(short)}; of those reward.py-verified = {len(pool)}")
    pick = np.sort(np.random.default_rng(SEED).choice(pool, size=N_A, replace=False))
    rows = []
    for k, i in enumerate(pick):
        r = d.iloc[i]
        q, gt = user_text(r["prompt"]), r["reward_model"]["ground_truth"]
        rows.append({
            "data_source": "luffy8k",
            "prompt": list(r["prompt"]),
            "reward_model": {"ground_truth": gt, "style": "rule"},
            "target": [{"role": "assistant", "content": traces[i]}],
            "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": traces[i]}],
            "extra_info": {"index": k, "question": q, "answer": gt, "source_id": int(i)},
            "trace_tokens": int(n_tok[i]),
            "orig_source": r["data_source"],
        })
    print(f"track A: sampled {N_A} from pool {len(pool)} / {len(d)}")
    return pd.DataFrame(rows)


def track_b(tok: Tokenizer) -> pd.DataFrame:
    sft = pd.read_parquet("code/data/bridge_curated_teachers_fixed/sft/train.parquet")
    rl = pd.read_parquet("code/data/bridge_curated_teachers/rl/train.parquet")
    assert list(sft["id"]) == list(rl["id"]), "SFT and RL rows must align"
    n_tok = [len(e.ids) for e in tok.encode_batch(list(sft["response"]))]
    rows = []
    for k, (s, r) in enumerate(zip(sft.itertuples(), rl.itertuples())):
        q = user_text(r.prompt)
        rows.append({
            "data_source": "curated",
            "prompt": list(r.prompt),
            "reward_model": dict(r.reward_model),          # ground_truth is reward.py's JSON answer spec
            "target": [{"role": "assistant", "content": s.response}],
            "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": s.response}],
            "extra_info": {"index": k, "question": s.question, "answer": s.answer, "source_id": s.id},
            "trace_tokens": n_tok[k],
            "orig_source": f"MATH/{s.subject}/L{s.level}",
        })
    print(f"track B: {len(rows)} rows")
    return pd.DataFrame(rows)


def main():
    tok = Tokenizer.from_file(TOKENIZER)
    for name, df in [("luffy8k", track_a(tok)), ("curated", track_b(tok))]:
        os.makedirs(f"{OUT}/{name}", exist_ok=True)
        df.to_parquet(f"{OUT}/{name}/train.parquet")
        t = df["trace_tokens"]
        print(f"  {name}: rows={len(df)} solution tokens median={int(t.median())} mean={int(t.mean())} max={int(t.max())}")
    with open(f"{OUT}/README.json", "w") as f:
        json.dump({"seed": SEED, "track_a_n": N_A, "max_trace_tokens": MAX_TRACE_TOKENS,
                   "tokenizer": "Qwen/Qwen2.5-3B"}, f, indent=2)


if __name__ == "__main__":
    main()
