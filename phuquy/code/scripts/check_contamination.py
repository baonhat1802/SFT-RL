"""Check overlap between the LUFFY-setup training set and its evaluation set.

For each eval question, find the training question that shares the largest fraction of its 13-grams
(after normalising text). A single shared 13-gram is not evidence: competition problems share boilerplate
("m and n are relatively prime positive integers", asymptote labels). We report:
  exact  = identical normalised text
  >=50%  = one training question covers at least half of the eval question's 13-grams (near duplicate)
Run from capstone/: conda run -n research python code/scripts/check_contamination.py
"""
import os
import re
from collections import Counter, defaultdict

import pandas as pd

DATA = "code/data/sft_then_rl"
N = 13
NEAR = 0.5


def user_text(prompt) -> str:
    msgs = [m for m in prompt if m["role"] == "user"]
    return msgs[-1]["content"] if msgs else ""


def norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\\[a-z]+", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = " ".join(s.split())
    return s.split("this is the problem", 1)[-1].strip()


def ngrams(s: str):
    w = s.split()
    return {" ".join(w[i:i + N]) for i in range(len(w) - N + 1)}


# TRAIN=<parquet> checks another training file (e.g. code/data/tracks/curated/train.parquet); output gets its name
TRAIN = os.environ.get("TRAIN", f"{DATA}/openr1_46k_8192.parquet")
train = pd.read_parquet(TRAIN)
ev = pd.read_parquet(f"{DATA}/all.parquet")
if "extra_info" in train and isinstance(train["extra_info"].iloc[0], dict) and "question" in train["extra_info"].iloc[0]:
    train_q = [norm(x["question"]) for x in train["extra_info"]]
else:
    train_q = [norm(user_text(p)) for p in train["prompt"]]
train_exact = set(train_q)
index = defaultdict(list)
for i, q in enumerate(train_q):
    for g in ngrams(q):
        index[g].append(i)

ev = ev.assign(q=[norm(user_text(p)) for p in ev["prompt"]]).drop_duplicates(["data_source", "q"])
print(f"train questions: {len(train_q)} (unique {len(train_exact)})")
print(f"{'benchmark':16s} {'n':>6s} {'exact':>6s} {'>=50%':>6s}")
flagged = []
for src, grp in ev.groupby("data_source"):
    exact = near = 0
    for q in grp["q"]:
        grams = ngrams(q)
        best, frac = None, 0.0
        if grams:
            c = Counter(i for g in grams for i in index.get(g, ()))
            if c:
                best, k = c.most_common(1)[0]
                frac = k / len(grams)
        if q in train_exact:
            exact += 1
        if q in train_exact or frac >= NEAR:
            near += 1
            flagged.append((src, frac, q[:150], train_q[best][:150] if best is not None else ""))
    print(f"{src:16s} {len(grp):6d} {exact:6d} {near:6d}  ({100 * near / len(grp):.1f}%)")
print("\nflagged (eval | closest train):")
for src, frac, q, t in flagged:
    print(f"- [{src} {frac:.0%}] {q}\n    train: {t}")

out = f"{DATA}/contaminated.tsv" if "TRAIN" not in os.environ else \
    os.path.join(os.path.dirname(TRAIN), "contaminated_vs_eval.tsv")
pd.DataFrame(flagged, columns=["data_source", "overlap", "eval_question_norm", "closest_train_norm"]).to_csv(out, sep="\t", index=False)
print(f"\nwrote {len(flagged)} flagged questions to {out}")
