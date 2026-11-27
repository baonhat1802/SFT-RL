"""Generate with vLLM and score like LUFFY's eval_scripts/generate_vllm.py, with prompts identical to training.

    PYTHONPATH=code python -m evaluation.run --model Qwen/Qwen2.5-3B --track curated \
        --data $CAPSTONE_DATA/sft_then_rl/dev.parquet --out runs/eval/base_curated_dev

Prompt per track (same as code/train): luffy8k -> LUFFY chat template, answer opens with "<think>\\n";
curated -> ChatML, system "You are a helpful assistant.", question + SimpleRL suffix. We do not use LUFFY's
`simplerl` template: it renders "within\\boxed{{}}" (no space, doubled braces), which differs from our training prompt.

Scoring follows LUFFY: for the think template, the text is "<think>\\n" + generation and only the part after
"</think>" is graded (no "</think>" -> wrong). Graded with Math-Verify (LUFFY's labeling_responses) and with LUFFY's
OAT grader (needs a LUFFY checkout: --luffy-eval-dir or $LUFFY_EVAL_DIR). Multiple-choice sets use the same graders.

all.parquet already repeats AIME24/AIME25/AMC 32 times, so --n 1 gives avg@32 on those and pass@1 elsewhere.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

SIMPLERL_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
SIMPLERL_SYSTEM = "You are a helpful assistant."
MATH_SOURCES = ["aime", "aime25", "amc", "math", "minerva", "olympiad_bench"]
OOD_SOURCES = ["arc_c", "gpqa", "mmlu_pro"]


def question_of(prompt) -> str:
    return [m for m in prompt if m["role"] == "user"][-1]["content"]


def messages_for(track: str, question: str) -> list[dict]:
    if track == "luffy8k":
        return [{"role": "user", "content": question}]          # the template supplies the system prompt
    return [{"role": "system", "content": SIMPLERL_SYSTEM}, {"role": "user", "content": question + SIMPLERL_SUFFIX}]


def norm_q(text: str) -> str:
    """Same normalisation as code/scripts/check_contamination.py, to match eval_exclude_union.tsv."""
    s = re.sub(r"\\[a-z]+", " ", text.lower())
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split()).split("this is the problem", 1)[-1].strip()


def graded_text(track: str, generation: str) -> str | None:
    if track != "luffy8k":
        return generation
    full = "<think>\n" + generation
    return full.split("</think>", 1)[1] if "</think>" in full else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local dir (base model, or a merged model)")
    ap.add_argument("--adapter", default=None, help="optional LoRA adapter dir, served through vLLM LoRA")
    ap.add_argument("--track", required=True, choices=["luffy8k", "curated"], help="selects the prompt format")
    ap.add_argument("--data", required=True, help="eval parquet (all.parquet or dev.parquet)")
    ap.add_argument("--sources", default=None, help="comma list of data_source values to keep")
    ap.add_argument("--exclude", default=None, help="tsv with eval_question_norm column; reported as a clean subset")
    ap.add_argument("--greedy", action="store_true", help="temperature 0 (BRIDGE anchor); default temperature 0.6")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--luffy-eval-dir", default=os.environ.get("LUFFY_EVAL_DIR"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import pandas as pd
    from math_verify import parse, verify
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from train.common import configure_tokenizer
    from train.models import eos_ids

    oat = None
    if a.luffy_eval_dir:
        sys.path.insert(0, a.luffy_eval_dir)
        from oat_math_grader import boxed_reward_fn as oat   # noqa: E402

    df = pd.read_parquet(a.data)
    if a.sources:
        df = df[df["data_source"].isin(a.sources.split(","))]
    df = df.reset_index(drop=True)
    excluded = set()
    if a.exclude:
        excluded = set(pd.read_csv(a.exclude, sep="\t")["eval_question_norm"].str[:150])

    tok = configure_tokenizer(AutoTokenizer.from_pretrained(a.adapter or a.model), a.track)
    prompts = [tok.apply_chat_template(messages_for(a.track, question_of(p)), tokenize=False,
                                       add_generation_prompt=True) for p in df["prompt"]]
    sp = SamplingParams(temperature=0.0 if a.greedy else 0.6, top_p=1.0, max_tokens=a.max_tokens,
                        stop_token_ids=sorted(set(eos_ids(tok, a.track)) | {tok.convert_tokens_to_ids("<|im_end|>")}),
                        seed=0)   # eval has no padding issue, so stop on either end token
    llm_kw = dict(model=a.model, gpu_memory_utilization=a.gpu_mem, max_model_len=1024 + a.max_tokens, seed=0)
    lora = None
    if a.adapter:
        from vllm.lora.request import LoRARequest
        llm_kw.update(enable_lora=True, max_lora_rank=64)
        lora = LoRARequest("adapter", 1, a.adapter)
    if os.environ.get("UNSLOTH_VLLM_NO_FLASHINFER", "0") != "0":
        # same as training: no FlashInfer (its JIT build fails on molab); FLASH_ATTN needs no compilation
        llm_kw["attention_backend"] = "FLASH_ATTN"
    try:
        llm = LLM(**llm_kw)
    except TypeError:                      # older vLLM without the attention_backend engine argument
        llm_kw.pop("attention_backend", None)
        llm = LLM(**llm_kw)
    outs = llm.generate(prompts, sp, lora_request=lora)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    per_source = defaultdict(lambda: {"mv": [], "oat": [], "clean_mv": [], "clean_oat": [], "len": [], "trunc": []})
    with open(out / "generations.jsonl", "w") as f:
        for i, o in enumerate(outs):
            src, gt = df["data_source"][i], df["reward_model"][i]["ground_truth"]
            gen = o.outputs[0].text
            txt = graded_text(a.track, gen)
            mv = ok_oat = False
            if txt is not None:
                try:
                    mv = bool(verify(parse("$" + gt + "$"), parse(txt)))
                except Exception:
                    mv = False
                if oat is not None:
                    try:
                        ok_oat = oat(txt, gt, fast=False)[1] == 1.0
                    except Exception:
                        ok_oat = False
            clean = norm_q(question_of(df["prompt"][i]))[:150] not in excluded
            r = per_source[src]
            r["mv"].append(mv)
            r["oat"].append(ok_oat)
            r["len"].append(len(o.outputs[0].token_ids))
            r["trunc"].append(o.outputs[0].finish_reason == "length")
            if clean:
                r["clean_mv"].append(mv)
                r["clean_oat"].append(ok_oat)
            f.write(json.dumps({"data_source": src, "answer": gt, "generation": gen, "mv": mv, "oat": ok_oat,
                                "clean": clean}) + "\n")

    def mean(x):
        return round(100 * sum(x) / len(x), 2) if x else None

    summary = {"model": a.model, "adapter": a.adapter, "track": a.track, "data": a.data, "greedy": a.greedy,
               "max_tokens": a.max_tokens, "per_source": {}}
    for src, r in sorted(per_source.items()):
        summary["per_source"][src] = {"n": len(r["mv"]), "math_verify": mean(r["mv"]),
                                      "oat": mean(r["oat"]) if oat else None,
                                      "math_verify_clean": mean(r["clean_mv"]), "n_clean": len(r["clean_mv"]),
                                      "mean_tokens": round(sum(r["len"]) / len(r["len"]), 1),
                                      "truncated_pct": mean(r["trunc"])}
    for group, srcs in [("math_avg", MATH_SOURCES), ("ood_avg", OOD_SOURCES)]:
        vals = [summary["per_source"][s]["math_verify"] for s in srcs if s in summary["per_source"]]
        summary[group] = round(sum(vals) / len(vals), 2) if vals else None
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
