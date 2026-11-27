"""Where does a GRPO rollout spend its time? (smoke step `profile`, enabled with CAPSTONE_PROFILE=1)

Wraps trainer methods and the vLLM engine's generate with wall-clock timers and prints `PROFILE ...` lines at exit.
TRL's own profiling only reports to W&B/MLflow, which these runs do not use.
"""
from __future__ import annotations

import atexit
import time
from collections import defaultdict

STATS: dict[str, list] = defaultdict(lambda: [0.0, 0])      # name -> [seconds, calls]
EXTRA: dict[str, object] = {}

TRAINER_METHODS = [
    "_generate_and_score_completions",     # whole rollout: generation + rewards + old log-probs
    "_generate",
    "_generate_single_turn",
    "_calculate_rewards",
    "_get_per_token_logps_and_entropies",
    "_get_per_token_logps",
    "training_step",
    "compute_loss",
]


def _wrap(obj, name: str, label: str | None = None, count_tokens: bool = False) -> bool:
    f = getattr(obj, name, None)
    if f is None or not callable(f) or getattr(f, "_capstone_wrapped", False):
        return False

    def timed(*args, **kwargs):
        t0 = time.perf_counter()
        out = f(*args, **kwargs)
        s = STATS[label or name]
        s[0] += time.perf_counter() - t0
        s[1] += 1
        if count_tokens:
            try:
                STATS["vllm.output_tokens"][0] += sum(len(c.token_ids) for o in out for c in o.outputs)
                STATS["vllm.requests"][0] += len(out)
            except Exception:
                pass
        return out

    timed._capstone_wrapped = True
    try:
        setattr(obj, name, timed)
        return True
    except Exception:
        return False


def instrument(trainer) -> None:
    for name in TRAINER_METHODS:
        _wrap(trainer, name, f"trainer.{name}")
    model = getattr(trainer, "model", None)
    for holder in (trainer, model):
        for attr in ("llm", "vllm_engine"):
            eng = getattr(holder, attr, None) if holder is not None else None
            if eng is not None and _wrap(eng, "generate", "vllm.generate", count_tokens=True):
                try:
                    cfg = eng.llm_engine.vllm_config
                    EXTRA["max_num_seqs"] = cfg.scheduler_config.max_num_seqs
                    EXTRA["max_num_batched_tokens"] = cfg.scheduler_config.max_num_batched_tokens
                    EXTRA["lora"] = cfg.lora_config is not None
                    EXTRA["gpu_memory_utilization"] = cfg.cache_config.gpu_memory_utilization
                except Exception as e:                       # noqa: BLE001
                    EXTRA["config_error"] = repr(e)[:120]
    if model is not None:
        _wrap(model, "fast_generate", "model.fast_generate", count_tokens=True)
    atexit.register(report)


def add(name: str, seconds: float, calls: int = 1) -> None:
    s = STATS[name]
    s[0] += seconds
    s[1] += calls


def report() -> None:
    for name, (sec, calls) in sorted(STATS.items(), key=lambda kv: -kv[1][0]):
        if name in ("vllm.output_tokens", "vllm.requests"):
            continue
        print(f"PROFILE {name}: {sec:.1f} s in {calls} calls", flush=True)
    toks, gen = STATS["vllm.output_tokens"][0], STATS["vllm.generate"][0] or STATS["model.fast_generate"][0]
    if toks:
        print(f"PROFILE vllm: {int(toks)} output tokens, {int(STATS['vllm.requests'][0])} requests, "
              f"{toks / max(gen, 1e-9):.0f} tokens/s", flush=True)
    if EXTRA:
        print(f"PROFILE vllm config: {EXTRA}", flush=True)
