"""Score .work/boxed_eq/cases.jsonl with one grader and write per-case results.

Graders (each must run in its own env because SimpleRL's latex2sympy2 pins antlr 4.7.2):
  team_reward   code/data/bridge_curated_teachers/reward.py::compute_score (training reward of the team)   env capstone
  luffy_mv      LUFFY eval: math_verify parse(response) vs parse("$gold$")                               env capstone
  luffy_oat     LUFFY eval --add_oat_evaluate: oat_math_grader.boxed_reward_fn                          env capstone
  simplerl      SimpleRL eval: parser.extract_answer(resp, "math") + grader.math_equal(pred, strip(gold)) env capstone-simplerl-eval
Run from capstone/:
  conda run -n capstone python code/scripts/boxed_equation_grade.py team_reward luffy_mv luffy_oat
  conda run -n capstone-simplerl-eval python code/scripts/boxed_equation_grade.py simplerl
"""
import json
import signal
import sys
from collections import defaultdict

import os
CASES = os.environ.get("CASES", ".work/boxed_eq/cases.jsonl")
OUTDIR = os.path.dirname(CASES)


class Timeout(Exception):
    pass


def with_timeout(fn, seconds=10):
    def handler(signum, frame):
        raise Timeout()
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        return fn()
    except Exception:
        return False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def make_grader(name):
    if name == "team_reward":
        sys.path.insert(0, "code/data/bridge_curated_teachers")
        from reward import compute_score
        return lambda c: compute_score(None, c["response"], c["ground_truth"]) == 1.0
    if name == "luffy_mv":
        from math_verify import parse, verify
        return lambda c: bool(verify(parse("$" + c["answer"] + "$"), parse(c["response"])))
    if name == "luffy_oat":
        sys.path.insert(0, "third_party/LUFFY/eval_scripts")
        from oat_math_grader import boxed_reward_fn
        return lambda c: boxed_reward_fn(c["response"], c["answer"], fast=False)[1] == 1.0
    if name == "simplerl":
        sys.path.insert(0, "third_party/simpleRL_eval")
        from grader import math_equal
        from parser import extract_answer, strip_string
        return lambda c: bool(math_equal(extract_answer(c["response"], "math"), strip_string(c["answer"])))
    raise SystemExit(f"unknown grader {name}")


cases = [json.loads(line) for line in open(CASES)]
for name in sys.argv[1:]:
    grade = make_grader(name)
    hits, totals, rows = defaultdict(int), defaultdict(int), []
    for c in cases:
        ok = bool(with_timeout(lambda: grade(c)))
        hits[c["group"]] += ok
        totals[c["group"]] += 1
        rows.append({"group": c["group"], "id": c["id"], "boxed": c["boxed"], "answer": c["answer"], name: ok})
    with open(f"{OUTDIR}/{name}.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(name, {g: f"{hits[g]}/{totals[g]} ({100 * hits[g] / totals[g]:.1f}%)" for g in totals}, flush=True)
