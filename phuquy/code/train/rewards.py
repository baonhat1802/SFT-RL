"""The common training reward: the team's reward.py (BaoNhat/bridge-math-curated-teachers @5756b56) for both tracks.

It reads the last \\boxed{} of the completion and checks it against the answer specification (JSON for the curated
track, a plain answer string for luffy8k; reward.py accepts both). Binary: 1.0 correct, 0.0 otherwise.
"""
from __future__ import annotations

import importlib.util
import os

from .common import DATA_ROOT

_TEAM_REWARD = os.environ.get("CAPSTONE_REWARD_PY", str(DATA_ROOT / "bridge_curated_teachers" / "reward.py"))
_spec = importlib.util.spec_from_file_location("team_reward", _TEAM_REWARD)
team_reward = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(team_reward)


def _text(completion) -> str:
    # TRL passes conversational completions as [{"role": "assistant", "content": ...}], plain ones as str
    if isinstance(completion, list):
        return completion[-1]["content"]
    return completion


_DEBUG_LEFT = [int(os.environ.get("CAPSTONE_DEBUG_COMPLETIONS", "0"))]
_IM_END, _ENDOFTEXT = 151645, 151643          # Qwen2.5 token ids


def _debug_print(completions, completion_ids):
    """Smoke-test aid (CAPSTONE_DEBUG_COMPLETIONS=N): on the first call, show how completions end."""
    if not _DEBUG_LEFT[0] or completion_ids is None:
        return
    n = len(completion_ids)
    has_im = sum(_IM_END in ids for ids in completion_ids)
    has_eot = sum(_ENDOFTEXT in ids for ids in completion_ids)
    lens = sorted(len(ids) for ids in completion_ids)
    print(f"DEBUG_COMPLETION batch={n} contains<|im_end|>={has_im} contains<|endoftext|>={has_eot} "
          f"len min/median/max={lens[0]}/{lens[n // 2]}/{lens[-1]}", flush=True)
    for c, ids in list(zip(completions, completion_ids))[:_DEBUG_LEFT[0]]:
        print(f"DEBUG_COMPLETION len={len(ids)} last_ids={list(ids[-6:])} tail={_text(c)[-160:]!r}", flush=True)
    _DEBUG_LEFT[0] = 0


def _score_one(args) -> float:
    text, gt = args
    try:
        return float(team_reward.compute_score(None, text, gt))
    except Exception:
        return 0.0


_POOL = [None]
_WORKERS = int(os.environ.get("CAPSTONE_REWARD_WORKERS", "1"))
_PROFILE = os.environ.get("CAPSTONE_PROFILE", "0") != "0"
_COMPARED = [False]


def _pool(n: int):
    """Worker processes for grading (Math-Verify is CPU-bound; molab has 4 CPUs). Spawned, not forked: the parent
    holds CUDA and vLLM state. Each worker imports this module and therefore reward.py once."""
    if _POOL[0] is None:
        import multiprocessing as mp
        _POOL[0] = mp.get_context("spawn").Pool(n)
    return _POOL[0]


def score_all(texts, gts, workers: int = 1) -> list[float]:
    jobs = list(zip(texts, gts))
    if workers <= 1:
        return [_score_one(j) for j in jobs]
    return _pool(workers).map(_score_one, jobs, chunksize=max(1, len(jobs) // (workers * 8)))


def correctness_reward(prompts, completions, ground_truth, completion_ids=None, **kwargs) -> list[float]:
    """TRL GRPO reward function. `ground_truth` comes from the dataset column of the same name."""
    import time
    _debug_print(completions, completion_ids)
    texts = [_text(c) for c in completions]
    t0 = time.perf_counter()
    out = score_all(texts, ground_truth, _WORKERS)
    if _PROFILE:
        from . import profiling
        profiling.add(f"reward.serial_or_{_WORKERS}_workers", time.perf_counter() - t0)
        if not _COMPARED[0]:                     # once: time the same batch on 4 worker processes, check equality
            _COMPARED[0] = True
            _pool(4)                             # start-up cost outside the timing
            t1 = time.perf_counter()
            par = score_all(texts, ground_truth, 4)
            profiling.add("reward.compare_4_workers", time.perf_counter() - t1)
            print(f"PROFILE reward: {len(texts)} completions, parallel result identical: {par == out}", flush=True)
    return out
