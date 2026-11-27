"""Evaluate on the 5 math sets of the team's results table (BRIDGE / SimpleRL eval: AIME24, AMC23, MATH500, Minerva,
OlympiadBench) with the curated-track prompt, n samples per question, a token budget (default 512 like the table).

    PYTHONPATH=code python -m beat.eval_bridge --model Qwen/Qwen2.5-3B --max-tokens 512 --n 4 --out runs/eval/base512
    PYTHONPATH=code python -m beat.eval_bridge --model Qwen/Qwen2.5-3B --adapter runs/x/final_adapter ...

Score per set = mean over the n sample indices of pass@1 (Math-Verify, like code/evaluation/run.py); "±" = std over
the n sample indices. "TB" (average) = unweighted mean of the 5 sets, as in the table. truncated% = share of samples that
hit the token budget.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

SETS = ["aime24", "amc23", "math500", "minerva_math", "olympiadbench"]
SIMPLERL_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
SIMPLERL_SYSTEM = "You are a helpful assistant."


def load_sets(root: Path):
    rows = []
    for s in SETS:
        for line in open(root / s / "test.jsonl"):
            d = json.loads(line)
            q = d.get("problem") or d.get("question")
            if s == "olympiadbench":
                q = d["question"]
                ans = d["final_answer"][0] if isinstance(d["final_answer"], list) else d["final_answer"]
            elif s == "minerva_math":
                ans = d.get("answer")
                if ans is None:                          # SimpleRL's minerva file keeps the answer in the solution
                    ans = last_boxed(d["solution"])
            else:
                ans = d["answer"]
            ans = str(ans).strip().strip("$").strip()
            if ans.endswith(".0") and ans[:-2].lstrip("-").isdigit():
                ans = ans[:-2]                           # AMC answers stored as floats (27.0)
            rows.append({"set": s, "question": q, "answer": ans})
    return rows


def last_boxed(text: str) -> str | None:
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    depth, k = 0, j
    while k < len(text):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
        k += 1
    return None


def grade(gen: str, gt: str) -> bool:
    from math_verify import parse, verify
    try:
        return bool(verify(parse("$" + gt + "$"), parse(gen)))
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-lora-rank", type=int, default=64)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--budget-includes-prompt", action="store_true",
                    help="--max-tokens caps prompt + answer together (answer budget = max_tokens - prompt length)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from train.common import DATA_ROOT, configure_tokenizer
    from train.models import eos_ids

    rows = load_sets(Path(a.data_root) if a.data_root else DATA_ROOT / "bridge" / "eval_math")
    tok = configure_tokenizer(AutoTokenizer.from_pretrained(a.adapter or a.model), "curated")
    prompts = [tok.apply_chat_template([{"role": "system", "content": SIMPLERL_SYSTEM},
                                        {"role": "user", "content": r["question"] + SIMPLERL_SUFFIX}],
                                       tokenize=False, add_generation_prompt=True) for r in rows]
    sp = SamplingParams(n=a.n, temperature=a.temperature, top_p=1.0, max_tokens=a.max_tokens, seed=0,
                        stop_token_ids=sorted(set(eos_ids(tok, "curated")) | {tok.convert_tokens_to_ids("<|im_end|>")}))
    if a.budget_includes_prompt:
        import copy
        plens = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
        sps = []
        for n_p in plens:
            x = copy.deepcopy(sp)
            x.max_tokens = max(1, a.max_tokens - n_p)
            sps.append(x)
        print(f"prompt tokens: median {sorted(plens)[len(plens) // 2]}, max {max(plens)}", flush=True)
        sp = sps
    kw = dict(model=a.model, gpu_memory_utilization=a.gpu_mem, max_model_len=1024 + a.max_tokens, seed=0)
    lora = None
    if a.adapter:
        from vllm.lora.request import LoRARequest
        kw.update(enable_lora=True, max_lora_rank=a.max_lora_rank)
        lora = LoRARequest("adapter", 1, a.adapter)
    import os
    if os.environ.get("UNSLOTH_VLLM_NO_FLASHINFER", "0") != "0":      # molab Blackwell: FlashInfer JIT fails
        kw["attention_backend"] = "FLASH_ATTN"
    try:
        llm = LLM(**kw)
    except TypeError:
        kw.pop("attention_backend", None)
        llm = LLM(**kw)
    outs = llm.generate(prompts, sp, lora_request=lora)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    acc = {s: [[] for _ in range(a.n)] for s in SETS}
    trunc = {s: [] for s in SETS}
    toks = {s: [] for s in SETS}
    with open(out / "generations.jsonl", "w") as f:
        for r, o in zip(rows, outs):
            for k, c in enumerate(o.outputs):
                ok = grade(c.text, r["answer"])
                acc[r["set"]][k].append(ok)
                trunc[r["set"]].append(c.finish_reason == "length")
                toks[r["set"]].append(len(c.token_ids))
                f.write(json.dumps({"set": r["set"], "k": k, "answer": r["answer"], "ok": ok,
                                    "truncated": c.finish_reason == "length", "generation": c.text}) + "\n")

    def pct(x):
        return 100 * sum(x) / len(x)

    summ = {"model": a.model, "adapter": a.adapter, "max_tokens": a.max_tokens, "n": a.n,
            "budget_includes_prompt": a.budget_includes_prompt,
            "temperature": a.temperature, "sets": {}}
    per_k_avg = [statistics.mean(pct(acc[s][k]) for s in SETS) for k in range(a.n)]
    for s in SETS:
        v = [pct(acc[s][k]) for k in range(a.n)]
        summ["sets"][s] = {"n_q": len(acc[s][0]), "mean": round(statistics.mean(v), 1),
                           "std": round(statistics.stdev(v), 1) if a.n > 1 else 0.0,
                           "truncated_pct": round(pct(trunc[s]), 1),
                           "mean_tokens": round(statistics.mean(toks[s]), 1)}
    dev = ["math500", "minerva_math", "olympiadbench"]       # explore metric (research/explore.md): no AIME/AMC
    per_k_dev = [statistics.mean(pct(acc[s][k]) for s in dev) for k in range(a.n)]
    summ["DEV"] = round(statistics.mean(per_k_dev), 1)
    summ["DEV_std"] = round(statistics.stdev(per_k_dev), 1) if a.n > 1 else 0.0
    summ["TB"] = round(statistics.mean(per_k_avg), 1)
    summ["TB_std"] = round(statistics.stdev(per_k_avg), 1) if a.n > 1 else 0.0
    all_tr = [t for s in SETS for t in trunc[s]]
    summ["truncated_pct"] = round(pct(all_tr), 1)
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    line = " | ".join(f"{s} {summ['sets'][s]['mean']}±{summ['sets'][s]['std']}" for s in SETS)
    print(f"RESULT {a.out}: {line} | TB {summ['TB']}±{summ['TB_std']} | DEV {summ['DEV']}±{summ['DEV_std']} | cut {summ['truncated_pct']}%", flush=True)


if __name__ == "__main__":
    main()
