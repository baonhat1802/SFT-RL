"""Build SFT data from the curated questions that the 3B model can actually imitate.

Why: SFT on the GPT-6 Luna solutions (median 115 tokens, skips steps) dropped the base model from 14.4 to 8.1 on the
team's table. Hypothesis: the style is too far from what a 3B model can follow. Two ways to make data in the model's
own style, both keeping only answers that the team's reward.py grades correct:

  self : the base model solves the question itself (rejection-sampling fine-tuning / STaR without hints)
  hint : the base model sees the Luna solution as a hint and writes its own full solution (STaR "rationalization");
         the SFT example pairs the ORIGINAL prompt (no hint) with that solution, so at test time no hint is needed

    PYTHONPATH=code python -m beat.gen_sft_data --mode hint --n 4 --max-tokens 512 --out runs/data/hint

Output: <out>/train.parquet with the track schema columns that train.data.sft_dataset reads (prompt, target, ...), one
example per question that got at least one usable answer, and <out>/stats.json.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

HINT_TEMPLATE = (
    "{question}\n\n"
    "Here is a short reference solution written by an expert. It may skip steps.\n"
    "<reference>\n{reference}\n</reference>\n\n"
    "Now write your own complete solution to the problem. Explain every step clearly, filling in any steps the "
    "reference skips. Do not mention the reference solution; write as if you solved it yourself."
    "\nPlease reason step by step, and put your final answer within \\boxed{{}}."
)
LEAK = re.compile(r"reference|provided solution|given solution|solution above|the expert|as shown above", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["self", "hint"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--workers", type=int, default=10, help="grading processes")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from train.common import configure_tokenizer
    from train.data import load_track
    from train.models import eos_ids
    from train.rewards import score_all

    df = load_track("curated")
    if a.limit:
        df = df.head(a.limit)
    tok = configure_tokenizer(AutoTokenizer.from_pretrained(a.model), "curated")

    def gen_prompt(r):
        msgs = [{"role": m["role"], "content": m["content"]} for m in r.prompt]
        if a.mode == "hint":
            q = r.extra_info["question"]
            msgs = [msgs[0], {"role": "user", "content": HINT_TEMPLATE.format(question=q,
                                                                               reference=r.target[0]["content"])}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    prompts = [gen_prompt(r) for r in df.itertuples()]
    sp = SamplingParams(n=a.n, temperature=a.temperature, top_p=1.0, max_tokens=a.max_tokens, seed=a.seed,
                        stop_token_ids=sorted(set(eos_ids(tok, "curated")) | {tok.convert_tokens_to_ids("<|im_end|>")}))
    llm = LLM(model=a.model, gpu_memory_utilization=0.85, max_model_len=2048 + a.max_tokens, seed=a.seed)
    outs = llm.generate(prompts, sp)

    texts, gts, meta = [], [], []
    for i, (r, o) in enumerate(zip(df.itertuples(), outs)):
        for c in o.outputs:
            texts.append(c.text)
            gts.append(r.reward_model["ground_truth"])
            meta.append((i, c.finish_reason == "length", len(c.token_ids)))
    scores = score_all(texts, gts, a.workers)

    rng = random.Random(a.seed)
    usable = {}
    n_correct = n_trunc = n_leak = 0
    for t, s, (i, trunc, ntok) in zip(texts, scores, meta):
        n_trunc += trunc
        if s != 1.0:
            continue
        n_correct += 1
        if trunc or "\\boxed" not in t:
            continue
        if a.mode == "hint" and LEAK.search(t):
            n_leak += 1
            continue
        usable.setdefault(i, []).append((t.strip(), ntok))
    rows, lens = [], []
    for i, cands in sorted(usable.items()):
        t, ntok = rng.choice(cands)
        r = df.iloc[i]
        rows.append({"data_source": r.data_source, "prompt": r.prompt, "reward_model": r.reward_model,
                     "target": [{"role": "assistant", "content": t}], "extra_info": r.extra_info})
        lens.append(ntok)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out / "train.parquet")
    lens.sort()
    stats = {"mode": a.mode, "questions": len(df), "samples": len(texts),
             "sample_accuracy_pct": round(100 * n_correct / len(texts), 1),
             "truncated_pct": round(100 * n_trunc / len(texts), 1), "leak_dropped": n_leak,
             "questions_with_example": len(rows), "coverage_pct": round(100 * len(rows) / len(df), 1),
             "example_tokens_median": lens[len(lens) // 2] if lens else None}
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    with open(out / "samples.jsonl", "w") as f:              # a few examples to read by eye
        for row in rows[:30]:
            f.write(json.dumps({"q": row["extra_info"]["question"], "a": row["target"][0]["content"]}) + "\n")
    print("STATS", json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
