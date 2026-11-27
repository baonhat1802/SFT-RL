"""Build the hill-climbing dev set: MATH-500 + Minerva + OlympiadBench from the LUFFY eval file,
minus every question that near-duplicates a problem of either training track
(code/data/tracks/eval_exclude_union.tsv, built from code/scripts/check_contamination.py runs).
AIME24/25, AMC and the OOD sets are kept for final evaluation only.
Run from capstone/: conda run -n research python code/scripts/make_dev_split.py
"""
import re

import pandas as pd

DATA = "code/data/sft_then_rl"
DEV_SOURCES = ["math", "minerva", "olympiad_bench"]


def norm(prompt) -> str:
    s = [m for m in prompt if m["role"] == "user"][-1]["content"].lower()
    s = re.sub(r"\\[a-z]+", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split()).split("this is the problem", 1)[-1].strip()


ev = pd.read_parquet(f"{DATA}/all.parquet")
bad = set(pd.read_csv("code/data/tracks/eval_exclude_union.tsv", sep="\t")["eval_question_norm"])
dev = ev[ev["data_source"].isin(DEV_SOURCES)].copy()
dev["_q"] = [norm(p) for p in dev["prompt"]]
dropped = dev["_q"].map(lambda q: q[:150] in bad)
dev = dev[~dropped].drop(columns="_q")
dev.to_parquet(f"{DATA}/dev.parquet")
print(f"dropped {int(dropped.sum())} contaminated; dev = {len(dev)}", dev["data_source"].value_counts().to_dict())
