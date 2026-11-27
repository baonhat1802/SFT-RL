"""Run ONE R01 job on molab end to end, resumable across sessions through a private HF repo.

    PYTHONPATH=code python -m beat.molab_run --job metaopsd_h0_s1 --hf-user <user>

Repo `<hf-user>/capstone-r01-<job>` holds ckpt/ (pushed by beat.loop every few rollouts), final_adapter/, and
eval/summary.json (@512) + eval2048/summary.json. A job whose two summaries exist on the Hub is skipped; a job with
a checkpoint resumes from it; a job with a final adapter but no evaluation is only evaluated.
Needs CAPSTONE_DATA (the Kaggle dataset dir, from `python -m train.smoke data`) and HF_TOKEN in the environment.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def stream(cmd, env):
    print("$ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    p = subprocess.Popen(cmd, env=env, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         bufsize=1)
    for line in p.stdout:
        if line.startswith(("STEP", "START", "RESUME", "RESULT", "DONE", "Traceback", "hub ", "no hub")) \
                or "Error" in line:
            print(line.rstrip()[:600], flush=True)
    return p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--jobs", default=str(ROOT / "research/rounds/R01/jobs.txt"))
    ap.add_argument("--hf-user", default=os.environ.get("HF_USER"))
    ap.add_argument("--work", default=os.environ.get("CAPSTONE_WORK", str(Path.home() / "capstone")))
    ap.add_argument("--loop-args", default="--max-completion 2048 --micro-batch 8 --vllm-mem 0.35")
    ap.add_argument("--eval-tokens", default="512 2048")
    a = ap.parse_args()

    spec = None
    for line in open(a.jobs):
        f = line.split()
        if f and not f[0].startswith("#") and f[0] == a.job:
            spec = f
    if spec is None:
        sys.exit(f"job {a.job} not in {a.jobs}")
    name, idea, rest = spec[0], spec[1], spec[2:]
    repo = f"{a.hf_user}/capstone-r01-{name}"
    out = Path(a.work) / "r01" / name
    out.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    api.create_repo(repo, private=True, exist_ok=True)
    have = {f for f in api.list_repo_files(repo)}
    evals = {T: ("eval" if T == "512" else f"eval{T}") for T in a.eval_tokens.split()}
    if all(f"{d}/summary.json" in have for d in evals.values()):
        print(f"{name}: đã xong trên Hub ({repo}), bỏ qua", flush=True)
        return

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "code")
    env.setdefault("UNSLOTH_VLLM_NO_FLASHINFER", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    if "final_adapter/adapter_config.json" not in have:
        cmd = [sys.executable, "-m", "beat.loop", "--idea", idea, "--seed", "1", "--out", str(out), "--hub-repo", repo,
               *shlex.split(a.loop_args), *rest]
        code = stream(cmd, env)
        if code:
            sys.exit(f"train {name} exit {code} — chạy lại notebook để tiếp từ checkpoint")
    elif not (out / "final_adapter" / "adapter_config.json").exists():
        snapshot_download(repo, allow_patterns=["final_adapter/*"], local_dir=str(out))

    for T, d in evals.items():
        if f"{d}/summary.json" in have:
            continue
        cmd = [sys.executable, "-m", "beat.eval_bridge", "--model", "Qwen/Qwen2.5-3B", "--adapter",
               str(out / "final_adapter"), "--max-lora-rank", "32", "--max-tokens", T, "--n", "4", "--out", str(out / d)]
        if stream(cmd, env):
            sys.exit(f"eval {name} @{T} failed")
        api.upload_file(path_or_fileobj=str(out / d / "summary.json"), path_in_repo=f"{d}/summary.json", repo_id=repo,
                        commit_message=f"eval @{T}")
    print(f"XONG {name} → {repo}", flush=True)


if __name__ == "__main__":
    main()
