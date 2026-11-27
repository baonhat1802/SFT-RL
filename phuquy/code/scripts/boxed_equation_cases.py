"""Build test cases: do evaluation graders accept a boxed equation such as \\boxed{x=-5}?

Source: the curated-teacher SFT split (code/data/bridge_curated_teachers). 778 teacher solutions box an
equation ("p+q=118", "x=-5"). If a student learns that habit, benchmark graders may mark correct answers wrong.

Groups written to .work/boxed_eq/cases.jsonl:
  eq        the 778 solutions as written (boxed content contains "=")
  eq_fixed  the same solutions with the boxed content replaced by the canonical answer (harness check: ~100%)
  control   300 random solutions whose boxed content has no "=" (baseline agreement of each grader)
Run from capstone/: conda run -n capstone python code/scripts/boxed_equation_cases.py
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, "code/data/bridge_curated_teachers")
from reward import final_boxed_answer  # noqa: E402

OUT = ".work/boxed_eq/cases.jsonl"


def replace_last_box(text: str, new: str) -> str:
    i = text.rfind("\\boxed{")
    j, depth = i + 7, 1
    while depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[:i + 7] + new + "}" + text[j:]


d = pd.read_parquet("code/data/bridge_curated_teachers/sft/train.parquet")
d["boxed"] = d["response"].map(final_boxed_answer)
has_eq = d["boxed"].str.contains("=", regex=False)
eq, control = d[has_eq], d[~has_eq].sample(300, random_state=0)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    for group, rows, fix in [("eq", eq, False), ("eq_fixed", eq, True), ("control", control, False)]:
        for r in rows.itertuples():
            resp = replace_last_box(r.response, r.answer) if fix else r.response
            f.write(json.dumps({"group": group, "id": r.id, "boxed": final_boxed_answer(resp), "answer": r.answer,
                                "ground_truth": r.reward_model["ground_truth"], "response": resp}) + "\n")
print(f"eq={len(eq)} eq_fixed={len(eq)} control={len(control)} -> {OUT}")
