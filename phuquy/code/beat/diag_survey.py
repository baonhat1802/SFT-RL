"""Reading survey (no training): the student's own solutions side by side with Luna's y*, for manual reading.

300 random questions x 8 plain samples. Picks 15 all-fail, 15 hard (1-3/8) and 10 medium (4-7/8) questions. For each picked question it
also records how surprising each step of y* is to the student (mean NLL of the step's tokens given the question and the earlier steps,
from vLLM prompt log-probs), to point at the places where Luna jumps.
Writes survey.json and survey.md (question, answer, y* with per-step surprisal, then the student's samples with their rewards).

    PYTHONPATH=code python -m beat.diag_survey --out runs/diag_survey
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=300)
    ap.add_argument("--num-gen", type=int, default=8)
    ap.add_argument("--max-completion", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=53)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import numpy as np
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from train.common import BASE_MODEL, GRPO, configure_tokenizer
    from train.data import load_track
    from train.models import eos_ids
    from train.rewards import score_all

    random.seed(a.seed); np.random.seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tok = configure_tokenizer(AutoTokenizer.from_pretrained(BASE_MODEL), "curated")
    stop_ids = sorted(set(eos_ids(tok, "curated")) | {tok.convert_tokens_to_ids("<|im_end|>")})
    MATH = re.compile(r"\\\((.*?)\\\)|\\\[(.*?)\\\]|\$\$(.*?)\$\$|\$(.*?)\$", re.S)

    rows = []
    for r in load_track("curated").itertuples():
        msgs = [{"role": m["role"], "content": m["content"]} for m in r.prompt]
        rows.append({"q": r.extra_info["question"], "ref": r.target[0]["content"], "ans": str(r.extra_info["answer"]),
                     "gt": r.reward_model["ground_truth"], "src": r.extra_info.get("source_id"),
                     "p": tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]})
    Q = [rows[i] for i in np.random.permutation(len(rows))[:a.questions].tolist()]

    llm = LLM(model=BASE_MODEL, gpu_memory_utilization=0.85, max_model_len=4096, seed=a.seed, enable_prefix_caching=True)
    sp = SamplingParams(temperature=GRPO["temperature"], top_p=1.0, max_tokens=a.max_completion, stop_token_ids=stop_ids, n=a.num_gen)
    outs = llm.generate([{"prompt_token_ids": x["p"]} for x in Q], sp, use_tqdm=False)
    rew = score_all([c.text for o in outs for c in o.outputs], [x["gt"] for x, o in zip(Q, outs) for _ in o.outputs], 8)
    k = 0
    for x, o in zip(Q, outs):
        x["samples"] = [{"text": c.text, "r": float(rew[k + j]), "len": len(c.token_ids), "trunc": c.finish_reason == "length"}
                        for j, c in enumerate(o.outputs)]
        x["n_ok"] = int(sum(s["r"] for s in x["samples"])); k += len(o.outputs)
    af = [x for x in Q if x["n_ok"] == 0]; hd = [x for x in Q if 1 <= x["n_ok"] <= 3]; md = [x for x in Q if 4 <= x["n_ok"] <= 7]
    pick = af[:15] + hd[:15] + md[:10]
    print(f"GEN mean {np.mean([x['n_ok'] for x in Q]) / a.num_gen:.3f} allfail {len(af)} hard {len(hd)} medium {len(md)}", flush=True)

    # per-step surprisal of y* under the student
    def steps_of(ref):
        spans = [(m.start(), m.end()) for m in MATH.finditer(ref)]
        inside = lambda c: any(s < c < e for s, e in spans)
        cuts = [m.end() for m in re.finditer(r"(?<=[.;:])\s+|\n+", ref) if not inside(m.start())]
        b = [0] + [c for c in cuts if 0 < c < len(ref)] + [len(ref)]
        return [ref[u:v] for u, v in zip(b, b[1:]) if ref[u:v]]

    reqs = []
    for x in pick:
        x["steps"] = steps_of(x["ref"])
        x["step_tok"] = [tok(t, add_special_tokens=False)["input_ids"] for t in x["steps"]]
        reqs.append({"prompt_token_ids": x["p"] + [t for s in x["step_tok"] for t in s]})
    lp = llm.generate(reqs, SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0), use_tqdm=False)
    for x, o in zip(pick, lp):
        pl = o.prompt_logprobs
        pos = len(x["p"])
        x["step_nll"] = []
        for s in x["step_tok"]:
            vals = []
            for j, t in enumerate(s):
                d = pl[pos + j]
                vals.append(-d[t].logprob if d is not None and t in d else float("nan"))
            pos += len(s)
            x["step_nll"].append(float(np.nanmean(vals)) if vals else float("nan"))

    keep = ["q", "ans", "ref", "src", "n_ok", "steps", "step_nll", "samples"]
    (out / "survey.json").write_text(json.dumps([{k: x[k] for k in keep} for x in pick], indent=1, ensure_ascii=False))
    md_lines = []
    for n, x in enumerate(pick):
        md_lines += [f"\n\n# Q{n} · {x['n_ok']}/8 đúng · đáp án `{x['ans']}`\n", x["q"], "\n## Luna (NLL trung bình mỗi bước)\n"]
        md_lines += [f"- [{v:.2f}] {s.strip()}" for s, v in zip(x["steps"], x["step_nll"])]
        ok = [s for s in x["samples"] if s["r"] == 1][:2]; bad = [s for s in x["samples"] if s["r"] == 0][:2]
        for tag, L in (("ĐÚNG", ok), ("SAI", bad)):
            for s in L:
                md_lines += [f"\n## Model — {tag} ({s['len']} token{', bị cắt' if s['trunc'] else ''})\n", s["text"]]
    (out / "survey.md").write_text("\n".join(md_lines))
    print(f"DONE {out}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
