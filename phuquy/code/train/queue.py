"""Run every baseline in priority order, resuming across molab sessions (PLAN_baselines_qwen25_3b.md section 1).

    PYTHONPATH=code python -m train.queue            # run everything not yet done
    PYTHONPATH=code python -m train.queue --list     # show the jobs, no network
    PYTHONPATH=code python -m train.queue --status   # what is done / running / waiting (needs HF_USER, HF_TOKEN)
    PYTHONPATH=code python -m train.queue --only=luffy8k-sft          # one notebook per baseline (train only)
    PYTHONPATH=code python -m train.queue --only=<eval keys> --no-wait   # evaluate what is trained, skip the rest

Jobs: B1 SFT, B2 GRPO, B3 SFT->RL for both tracks, and an evaluation of every model incl. the base (B0); priority
order in all_jobs(). B3 and the evaluations wait for the jobs they need.
Each job keeps its state on the Hugging Face Hub, in a private repo <HF_USER>/capstone-<track>-<baseline>-seed1:
  - a job whose final artefact is on the Hub is skipped;
  - an unfinished training job resumes from its newest Hub checkpoint (train/hub.py);
  - eval results are uploaded under eval/ in the same repo.
So when a session dies, re-running the notebook continues where it stopped. The queue stops at the first failure.

Several notebooks can run this queue at once (molab allows several GPU sessions per account). A worker claims a job by
writing lock.json (worker id + heartbeat) to the job's Hub repo, refreshes it every 5 minutes while the job runs, and
skips jobs locked by someone else. A lock without a heartbeat for 20 minutes is stale (that session died) and the job
can be taken over; it resumes from its Hub checkpoint. Jobs only start once the jobs they depend on are done.
A new notebook installs packages and downloads data itself (smoke steps `install` and `data`).
"""
from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import uuid

from . import smoke
from .smoke import PY, WORK, block, child_env, cleanup_leftovers, run, tail

WORKER = f"{socket.gethostname()}-{uuid.uuid4().hex[:6]}"
LOCK_FILE = "lock.json"
HEARTBEAT_S = 300            # refresh the lock every 5 minutes
STALE_S = 20 * 60            # a lock older than this belongs to a dead session

SEED = 1
BASE = "Qwen/Qwen2.5-3B"
TRACKS = ["luffy8k", "curated"]


def repo(track: str, baseline: str) -> str:
    return f"{os.environ.get('HF_USER', '<HF_USER>')}/capstone-{track}-{baseline}-seed{SEED}"


def hub_file(repo_id: str, path: str) -> bool:
    from huggingface_hub import HfApi
    try:
        return HfApi().file_exists(repo_id, path)
    except Exception:
        return False


def fetch(repo_id: str, sub: str) -> str:
    """Download <repo_id>/<sub>/ to WORK/models/<repo name>/<sub> and return that path."""
    from huggingface_hub import snapshot_download
    local = WORK / "models" / repo_id.split("/")[-1]
    snapshot_download(repo_id, allow_patterns=[f"{sub}/*"], local_dir=str(local))
    return str(local / sub)


# ------------------------------------------------------------------ jobs

def train_sft(track):
    r = repo(track, "sft")
    return dict(
        key=f"{track}-sft", deps=[],
        name=f"{track}/B1 SFT", repo=r,
        done=lambda: hub_file(r, "final_adapter/adapter_config.json") and hub_file(r, "merged_bf16/config.json"),
        cmd=lambda: [PY, "-m", "train.sft", "--track", track, "--seed", str(SEED), "--hub-repo", r],
    )


def train_grpo(track, baseline):
    r = repo(track, baseline)

    def cmd():
        c = [PY, "-m", "train.grpo", "--track", track, "--baseline", baseline, "--seed", str(SEED), "--hub-repo", r]
        if baseline == "sft_rl":
            c += ["--init-model", fetch(repo(track, "sft"), "merged_bf16")]
        return c

    label = {"grpo": "B2 GRPO", "sft_rl": "B3 SFT->RL"}[baseline]
    deps = [f"{track}-sft"] if baseline == "sft_rl" else []
    return dict(key=f"{track}-{baseline}", deps=deps, name=f"{track}/{label}", repo=r,
                done=lambda: hub_file(r, "final_adapter/adapter_config.json"), cmd=cmd)


def evaluate(track, baseline):
    """Full evaluation (9 sets in all.parquet, temperature 0.6, 8,192 tokens) of one model with its track's prompt."""
    r = repo(track, baseline)
    out = WORK / "runs" / "eval" / f"{track}-{baseline}"

    def cmd():
        data = child_env()["CAPSTONE_DATA"]
        if baseline == "base":
            model, adapter = BASE, None
        elif baseline == "sft":
            model, adapter = fetch(repo(track, "sft"), "merged_bf16"), None
        elif baseline == "grpo":
            model, adapter = BASE, fetch(r, "final_adapter")
        else:                                            # sft_rl: LoRA trained on top of the merged SFT model
            model, adapter = fetch(repo(track, "sft"), "merged_bf16"), fetch(r, "final_adapter")
        c = [PY, "-m", "evaluation.run", "--model", model, "--track", track,
             "--data", f"{data}/sft_then_rl/all.parquet", "--exclude", f"{data}/tracks/eval_exclude_union.tsv",
             "--out", str(out)]
        return c + (["--adapter", adapter] if adapter else [])

    def after():
        from huggingface_hub import HfApi
        from .hub import ensure_repo
        ensure_repo(r)
        HfApi().upload_folder(repo_id=r, folder_path=str(out), path_in_repo="eval", commit_message="eval")

    label = {"base": "B0 gốc", "sft": "B1", "grpo": "B2", "sft_rl": "B3"}[baseline]
    deps = [] if baseline == "base" else [f"{track}-{baseline}"] + ([f"{track}-sft"] if baseline == "sft_rl" else [])
    return dict(key=f"{track}-eval-{baseline}", deps=deps, name=f"{track}/đánh giá {label}", repo=r,
                lock_file="eval.lock.json",                 # same repo as the training job, separate lock
                done=lambda: hub_file(r, "eval/summary.json"), cmd=cmd, after=after, summary=out / "summary.json")


def all_jobs():
    """Priority order = what a free worker picks first. The longest chain (SFT -> SFT->RL, longest on track A, whose
    SFT model writes R1-length answers) goes first; evaluations, which are short, fill in around it."""
    jobs = [train_sft(t) for t in TRACKS] + [train_grpo(t, "grpo") for t in TRACKS] + \
        [train_grpo(t, "sft_rl") for t in TRACKS]
    jobs += [evaluate(t, "base") for t in TRACKS]
    for b in ["sft", "grpo", "sft_rl"]:
        jobs += [evaluate(t, b) for t in TRACKS]
    return jobs


# ------------------------------------------------------------------ driver

def results_table() -> str:
    """Math average and OOD average for every evaluated model (read from the Hub)."""
    from huggingface_hub import hf_hub_download
    rows = []
    for t in TRACKS:
        for b in ["base", "sft", "grpo", "sft_rl"]:
            r = repo(t, b)
            if not hub_file(r, "eval/summary.json"):
                continue
            s = json.loads(open(hf_hub_download(r, "eval/summary.json")).read())
            ps = s["per_source"]
            cells = " | ".join(str(ps.get(k, {}).get("math_verify")) for k in
                               ["aime", "aime25", "amc", "math", "minerva", "olympiad_bench", "arc_c", "gpqa", "mmlu_pro"])
            rows.append(f"| {t} | {b} | {s.get('math_avg')} | {s.get('ood_avg')} | {cells} |")
    if not rows:
        return ""
    head = ("| bộ | baseline | TB toán | TB ngoài phân phối | AIME24 | AIME25 | AMC | MATH-500 | Minerva | Olympiad | "
            "ARC-c | GPQA | MMLU-Pro |\n|" + "---|" * 13)
    return "\n### Kết quả (Math-Verify, bộ chuẩn)\n\n" + head + "\n" + "\n".join(rows) + "\n"


# --------------------------------------------------------------- locking

def _lock_path(job) -> str:
    return job.get("lock_file", LOCK_FILE)


def read_lock(job) -> dict | None:
    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(job["repo"], _lock_path(job), force_download=True)
        return json.loads(open(path).read())
    except Exception:
        return None


def write_lock(job, release: bool = False) -> None:
    from huggingface_hub import HfApi
    from .hub import ensure_repo
    ensure_repo(job["repo"])
    if release:
        try:
            HfApi().delete_file(_lock_path(job), job["repo"], commit_message=f"release {job['key']}")
        except Exception:
            pass
        return
    body = json.dumps({"worker": WORKER, "job": job["key"], "heartbeat": time.time()}).encode()
    HfApi().upload_file(path_or_fileobj=body, path_in_repo=_lock_path(job), repo_id=job["repo"],
                        commit_message=f"lock {job['key']} by {WORKER}")


def locked_by_other(job) -> bool:
    lock = read_lock(job)
    return bool(lock) and lock.get("worker") != WORKER and time.time() - lock.get("heartbeat", 0) < STALE_S


def claim(job) -> bool:
    """Write our lock, wait, read it back: if another worker wrote at the same moment, only one id survives."""
    if locked_by_other(job):
        return False
    write_lock(job)
    time.sleep(15)
    lock = read_lock(job)
    return bool(lock) and lock.get("worker") == WORKER


class Heartbeat:
    def __init__(self, job):
        self.job, self.stop = job, threading.Event()
        self.thread = threading.Thread(target=self._beat, daemon=True)

    def _beat(self):
        while not self.stop.wait(HEARTBEAT_S):
            try:
                write_lock(self.job)
            except Exception:
                pass

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        write_lock(self.job, release=True)


# --------------------------------------------------------------- driver

def prepare() -> None:
    """A fresh notebook (its own machine on molab) needs packages and data: run those smoke steps when missing."""
    for key in ("install", "data"):
        done = WORK / "reports" / f"{key}.md"
        stale = key == "install" and done.exists() and smoke.INSTALL_SPEC not in done.read_text()
        if done.exists() and "**PASS**" in done.read_text() and not stale:
            continue
        print(f"⏳ chuẩn bị máy: {key} …", flush=True)
        body = getattr(smoke, f"step_{key}")()
        (WORK / "reports").mkdir(parents=True, exist_ok=True)
        done.write_text(f"### {key}\n\n{body}\n")
        if "**PASS**" not in body:
            print(f"**FAIL** chuẩn bị máy ({key}):\n{body}", flush=True)
            sys.exit(1)
        print(f"✓ {key}", flush=True)


def run_job(job) -> None:
    print(f"\n⏳ {job['name']} (máy {WORKER}) …", flush=True)
    t0 = time.time()
    with Heartbeat(job):
        code, out = run(job["cmd"](), job["key"])
        if code == 0 and job.get("after"):
            job["after"]()
    minutes = round((time.time() - t0) / 60)
    if code != 0:
        print(f"**FAIL** {job['name']} sau {minutes} phút (exit {code}). Sửa xong chạy lại notebook: việc này chạy "
              f"tiếp từ checkpoint." + block(tail(out, 40)), flush=True)
        sys.exit(1)
    extra = ""
    if job.get("summary") and job["summary"].exists():
        s = json.loads(job["summary"].read_text())
        extra = f" · TB toán {s.get('math_avg')} · TB ngoài phân phối {s.get('ood_avg')}"
    print(f"✓ {job['name']}: xong sau {minutes} phút{extra}", flush=True)


TOTAL_STEPS = {"luffy8k-sft": 375, "curated-sft": 400, "luffy8k-grpo": 125, "curated-grpo": 133,
               "luffy8k-sft_rl": 125, "curated-sft_rl": 133}      # optimizer steps (SFT 3 epochs, GRPO 1 epoch)


def status() -> str:
    """One table of every job: done, running (which worker, heartbeat age, latest checkpoint), stale, waiting, to do."""
    from huggingface_hub import HfApi
    api = HfApi()
    jobs = all_jobs()
    done = {j["key"] for j in jobs if j["done"]()}
    rows = []
    for j in jobs:
        ckpt = ""
        if j["key"] in TOTAL_STEPS:
            try:
                names = [f.path for f in api.list_repo_tree(j["repo"])]
                steps = sorted(int(n.split("-")[1]) for n in names if n.startswith("checkpoint-"))
                if steps:
                    ckpt = f"{steps[-1]}/{TOTAL_STEPS[j['key']]}"
            except Exception:
                pass
        if j["key"] in done:
            state = "✓ xong"
        else:
            lock = read_lock(j)
            if lock and time.time() - lock.get("heartbeat", 0) < STALE_S:
                state = f"⏳ đang chạy · {lock['worker']} · nhịp sống {int((time.time() - lock['heartbeat']) / 60)} phút trước"
            elif lock:
                state = "⚠️ máy giữ việc đã im quá 20 phút: máy khác sẽ nhận lại"
            elif any(d not in done for d in j["deps"]):
                state = f"chờ {', '.join(d for d in j['deps'] if d not in done)}"
            else:
                state = "chưa ai nhận"
        rows.append(f"| {j['name']} | {state} | {ckpt} |")
    return "| việc | trạng thái | checkpoint mới nhất |\n|---|---|---|\n" + "\n".join(rows)


def main():
    if "--status" in sys.argv:
        print(status())
        print(results_table())
        return
    if "--list" in sys.argv:
        for i, j in enumerate(all_jobs(), 1):
            print(f"{i:2d}. {j['name']}  ->  {j['repo']}  (cần: {', '.join(j['deps']) or 'không'})")
        return
    if not os.environ.get("HF_USER") or not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_USER and HF_TOKEN must be set (the notebook sets them)")
    print(cleanup_leftovers(), flush=True)
    prepare()
    everything = all_jobs()
    jobs = everything
    only = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--only=")), None)
    if only:                                     # one notebook per baseline: run just these jobs, in this order
        keys = only.split(",")
        by_key = {j["key"]: j for j in everything}
        unknown = [k for k in keys if k not in by_key]
        if unknown:
            raise SystemExit(f"unknown job(s) {unknown}; known: {sorted(by_key)}")
        jobs = [by_key[k] for k in keys]
    print(f"Máy này: `{WORKER}`. Nhiều notebook có thể chạy hàng đợi này cùng lúc; mỗi việc chỉ một máy nhận.", flush=True)
    while True:
        done = {j["key"] for j in everything if j["done"]()}
        if all(j["key"] in done for j in jobs):
            break
        started = False
        for job in jobs:
            if job["key"] in done or any(d not in done for d in job["deps"]):
                continue
            if claim(job):
                run_job(job)
                started = True
                break
        if not started and "--no-wait" in sys.argv:
            # evaluation notebook: evaluate what is trained, report the rest, do not wait for training
            skipped = [j["name"] for j in jobs if j["key"] not in done]
            print(f"Bỏ qua (model chưa train xong hoặc máy khác đang làm): {', '.join(skipped)}", flush=True)
            break
        if not started:
            waiting = [j["name"] for j in jobs if j["key"] not in done]
            need = sorted({d for j in jobs if j["key"] not in done for d in j["deps"] if d not in done})
            print(f"… chưa chạy được: {', '.join(waiting)}. Đang chờ {', '.join(need) or 'máy khác đang giữ việc'} "
                  f"xong; kiểm lại sau 5 phút.", flush=True)
            time.sleep(300)
    print(results_table() + "\n**Hàng đợi xong.**", flush=True)


if __name__ == "__main__":
    main()
