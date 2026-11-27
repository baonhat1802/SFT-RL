"""Repair two formatting defects in the curated-teacher SFT split, keeping the original file untouched.

1. Boxed equations (778 rows), e.g. \\boxed{p+q=118}. Rewrite as p+q=\\boxed{118}: the sentence still reads the same,
   but the box holds only the value, which every benchmark grader extracts. SimpleRL's grader rejected 42% of the
   original boxes (code/scripts/boxed_equation_grade.py).
2. Math delimiters that close with the wrong token, e.g. \\(\\boxed{2}.\\] or a display block closed by a bare "]" line,
   or a block left open at the end of the text (output ends in a lone "\\" where "\\]" was cut off). Replace the wrong
   closer with the one matching the opener; close blocks still open at the end.

Every changed row must still score 1.0 under the team's reward.py; otherwise the original text is kept and flagged.
Input : code/data/bridge_curated_teachers/sft/train.parquet
Output: code/data/bridge_curated_teachers_fixed/sft/train.parquet (+ columns response_original, fixes)
Run from capstone/: ~/miniconda3/envs/capstone/bin/python code/scripts/fix_curated_sft.py
"""
import hashlib
import json
import os
import sys
from collections import Counter

import pandas as pd

SRC = "code/data/bridge_curated_teachers"
DST = "code/data/bridge_curated_teachers_fixed"
sys.path.insert(0, SRC)
from reward import compute_score, final_boxed_answer  # noqa: E402

OPEN = {"\\(": "\\)", "\\[": "\\]"}
CLOSE = {"\\)", "\\]"}


def fix_delimiters(text: str) -> tuple[str, int]:
    """Make every closer match the most recent opener. Skips the LaTeX line break '\\\\'."""
    out, stack, fixes, i = [], [], 0, 0
    while i < len(text):
        two = text[i:i + 2]
        if two == "\\\\":
            out.append(two)
            i += 2
            continue
        if two in OPEN:
            stack.append(two)
            out.append(two)
            i += 2
            continue
        if two in CLOSE:
            if not stack:              # orphan closer, e.g. a stray "\]" after "\(\boxed{5}\)."
                fixes += 1
                while out and out[-1] in (" ", "\n"):
                    out.pop()
                i += 2
                continue
            want = OPEN[stack.pop()]
            fixes += want != two
            out.append(want)
            i += 2
            continue
        # display block closed by a bare "]" on its own line
        if text[i] == "]" and stack and stack[-1] == "\\[":
            line_start = text.rfind("\n", 0, i) + 1
            line_end = text.find("\n", i)
            line_end = len(text) if line_end < 0 else line_end
            if text[line_start:line_end].strip() == "]":
                stack.pop()
                out.append("\\]")
                fixes += 1
                i += 1
                continue
        out.append(text[i])
        i += 1
    # blocks still open at the end: the teacher output often ends in a lone "\" where "\]" was cut off
    res = "".join(out)
    while stack:
        closer = OPEN[stack.pop()]
        res = res + closer[1:] if res.endswith("\\") and not res.endswith("\\\\") else res.rstrip() + "\n" + closer
        fixes += 1
    return res, fixes


def last_box_span(text: str):
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + 7, 1
    while j < len(text) and depth:
        if text[j] == "\\":
            j += 2
            continue
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return (i, j) if depth == 0 else None


def top_level_split(inner: str):
    """Split at the last '=' outside braces; None if there is none."""
    depth, pos = 0, None
    for k, ch in enumerate(inner):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == "=" and depth == 0 and inner[k - 1:k] not in ("<", ">", "!") and inner[k + 1:k + 2] != "=":
            pos = k
    return None if pos is None else (inner[:pos].rstrip(), inner[pos + 1:].strip())


def fix_boxed_equation(text: str):
    span = last_box_span(text)
    if not span:
        return text, False
    i, j = span
    parts = top_level_split(text[i + 7:j - 1])
    if not parts or not parts[0] or not parts[1]:
        return text, False
    lhs, rhs = parts
    return text[:i] + lhs + "=\\boxed{" + rhs + "}" + text[j:], True


def main():
    d = pd.read_parquet(f"{SRC}/sft/train.parquet")
    new_resp, fix_log, counts = [], [], Counter()
    for r in d.itertuples():
        text, fixes = r.response, []
        t2, n = fix_delimiters(text)
        if n:
            text, fixes = t2, fixes + [f"delimiters:{n}"]
        t3, changed = fix_boxed_equation(text)
        if changed:
            text, fixes = t3, fixes + ["boxed_equation"]
        if fixes and compute_score(None, text, r.reward_model["ground_truth"]) != 1.0:
            counts["reverted_reward_fail"] += 1
            text, fixes = r.response, ["REVERTED"]
        for f in fixes:
            counts[f.split(":")[0]] += 1
        new_resp.append(text)
        fix_log.append(json.dumps(fixes))

    out = d.copy()
    out["response_original"] = d["response"]
    out["response"] = new_resp
    out["fixes"] = fix_log
    out["messages"] = [[*m[:-1], {**m[-1], "content": t}] for m, t in zip(d["messages"], new_resp)]
    out["completion"] = [[{**c[0], "content": t}] for c, t in zip(d["completion"], new_resp)]
    out["solution_sha256"] = [hashlib.sha256(t.encode()).hexdigest() for t in new_resp]

    still_eq = sum("=" in (final_boxed_answer(t) or "") for t in new_resp)
    still_bad = sum(fix_delimiters(t)[1] > 0 for t in new_resp)
    os.makedirs(f"{DST}/sft", exist_ok=True)
    out.to_parquet(f"{DST}/sft/train.parquet")
    print("rows:", len(out), "| changed:", int((out["response"] != out["response_original"]).sum()))
    print("fix counts:", dict(counts))
    print("after fix: boxed still containing '=':", still_eq, "| rows with mismatched delimiters:", still_bad)


if __name__ == "__main__":
    main()
