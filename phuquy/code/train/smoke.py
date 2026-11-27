"""Environment and pipeline checks on molab (PLAN_baselines_qwen25_3b.md section 2), one step at a time.

    PYTHONPATH=code python -m train.smoke --list          # steps as JSON, read by notebooks/00_env_smoke.py
    PYTHONPATH=code python -m train.smoke <step>

The notebook only pulls this repo and calls these steps, so changes land with a `git pull`; the notebook itself does
not need re-uploading. Every step writes a markdown report to $CAPSTONE_WORK/reports/<step>.md and prints it.
Heavy libraries are imported inside the child commands, never here: `install` runs before they exist.

Environment (set by the notebook): CAPSTONE_WORK, HF_TOKEN, HF_USER, GITHUB_TOKEN + GIT_ASKPASS, LUFFY_EVAL_DIR.
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("CAPSTONE_WORK", str(Path.home() / "capstone")))
PY = sys.executable
KAGGLE_DATASET = "nguyenlamphuquy/gstar-capstone-sft-rl-math"
BRIDGE_MATH500_BASE = 32.4

STEPS = [
    ("all", "Chạy tất cả từ đầu đến cuối (dừng ở bước đầu tiên FAIL), khoảng 2–3 giờ"),
    ("install", "Cài thư viện (unsloth, vllm, trl 0.22.2, transformers 4.57.6, …)"),
    ("gpu", "Kiểm GPU và phiên bản thư viện"),
    ("data", "Tải dữ liệu từ Kaggle"),
    ("tests", "Unit test: template, reward, cộng dồn gradient (model tí hon)"),
    ("grad_real", "Cộng dồn gradient với Qwen2.5-3B thật qua Unsloth"),
    ("sft_smoke", "Chạy thử SFT vài bước, mỗi bộ"),
    ("grpo_smoke", "Chạy thử GRPO 2 bước, mỗi bộ + kiểm token dừng"),
    ("resume", "Lưu checkpoint lên HF rồi chạy tiếp (giả lập hết phiên)"),
    ("anchor", "Điểm neo: Qwen2.5-3B gốc, MATH-500, greedy, 4.096 token (đối chiếu BRIDGE 32,4)"),
    ("speed", "Đo tốc độ ở batch thật (1–2 giờ)"),
    ("profile", "Đo từng phần của một lượt sinh GRPO (sinh, chấm, cập nhật), khoảng 10 phút"),
    ("report", "Ghi experiments/env.txt và smoke_speed.json vào repo"),
]


# ---------------------------------------------------------------- helpers

def child_env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_DIR / "code")
    env.setdefault("HF_HOME", str(WORK / "hf"))
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("UNSLOTH_VLLM_STANDBY", "1")
    env["WANDB_DISABLED"] = "true"
    # molab ships TensorFlow/Keras 3; transformers 4.57 then tries its TF path and fails. We only use PyTorch.
    env["USE_TF"] = "0"
    env["TRANSFORMERS_NO_TF"] = "1"
    env["USE_FLAX"] = "0"
    # FlashInfer JIT-compiles CUDA kernels for sm_120a and fails on molab's CUDA 13 headers ("nv/target: No such
    # file"). On Blackwell Unsloth re-enables FlashInfer itself, overriding VLLM_USE_FLASHINFER_SAMPLER; its own switch
    # UNSLOTH_VLLM_NO_FLASHINFER=1 blocks FlashInfer and makes vLLM use FLASH_ATTN + the PyTorch sampler (no compile).
    env["UNSLOTH_VLLM_NO_FLASHINFER"] = "1"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    data_file = WORK / "data_path.txt"
    if data_file.exists():
        env["CAPSTONE_DATA"] = data_file.read_text().strip()
    return env


def run(cmd: list[str], log: str, extra_env: dict | None = None) -> tuple[int, str]:
    """Run a child command from the repo root, log to WORK/logs/<log>.log, return (exit code, full output)."""
    logs = WORK / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / f"{log}.log"
    env = {**child_env(), **(extra_env or {})}
    t0 = time.time()
    with open(path, "w") as f:
        proc = subprocess.Popen(cmd, cwd=str(REPO_DIR), env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
        # heartbeat: every 60 s print the newest log line, so the notebook shows progress during long steps
        while proc.poll() is None:
            time.sleep(5)
            if time.time() - t0 >= 60 and int(time.time() - t0) % 60 < 5:
                last = next((l for l in reversed(path.read_text(errors="replace").splitlines()) if l.strip()), "")
                print(f"  … {int(time.time() - t0) // 60} phút · `{log}` · {last.strip()[-160:]}", flush=True)
    return proc.returncode, path.read_text(errors="replace")


def tail(text: str, n: int = 40) -> str:
    return "\n".join(text.splitlines()[-n:])


def train_logs(text: str) -> list[dict]:
    """The dict lines transformers' Trainer prints, e.g. {'loss': 1.2, 'epoch': 0.1}."""
    out = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("{'") and s.endswith("}"):
            try:
                out.append(ast.literal_eval(s))
            except Exception:
                pass
    return out


def verdict(ok: bool) -> str:
    return "**PASS**" if ok else "**FAIL**"


def block(text: str) -> str:
    return f"\n```\n{text}\n```\n"


# ---------------------------------------------------------------- steps

# Unsloth's own GRPO notebook (Qwen2.5_(3B)-GRPO.ipynb) pins vllm 0.15.1 + transformers 4.57.6 + trl 0.22.2 (--no-deps).
# An unpinned vllm pulls a release that needs transformers 5 (ImportError: PreTrainedConfig). Bump INSTALL_SPEC whenever
# the pins change: step_all re-runs `install` when its last report does not mention the current spec.
# Steps whose result depends on how training examples are built. Bump DATA_FORMAT when that changes (e.g. the EOS):
# step_all re-runs these steps when their last report was made with another format.
DATA_FORMAT = "eos-endoftext-both-tracks"
FORMAT_STEPS = ("tests", "sft_smoke")
# Throughput settings (common.VLLM_MAX_NUM_SEQS, GRPO_MICRO_BATCH, GRPO_LOGPROB_CHUNKS); re-run `profile` when they change.
PERF_CONFIG = "seqs512-btok16k-mb8-chunks128"
PERF_STEPS = ("profile",)
INSTALL_SPEC = "vllm==0.15.1 transformers==4.57.6 trl==0.22.2 torchao<0.18 no-broken-flash-attn-v2"


def step_install() -> str:
    cmds = [
        [PY, "-m", "pip", "install", "-q", "--upgrade", "uv"],
        [PY, "-m", "uv", "pip", "install", "--python", PY, "unsloth", "vllm==0.15.1"],
        [PY, "-m", "uv", "pip", "install", "--python", PY, "transformers==4.57.6"],
        [PY, "-m", "uv", "pip", "install", "--python", PY, "--no-deps", "trl==0.22.2"],
        # vllm 0.15.1 pins torch 2.9; torchao >= 0.18 imports torch.nn.functional.ScalingType (torch >= 2.10).
        # Same fix as Unsloth's notebook; peft >= 0.19 needs torchao >= 0.16.
        [PY, "-m", "uv", "pip", "install", "--python", PY, "--no-deps", "torchao>=0.16.0,<0.18.0"],
        [PY, "-m", "uv", "pip", "install", "--python", PY, "math-verify==0.8.0", "sympy==1.14.0",
         "latex2sympy2_extended==1.10.2", "antlr4-python3-runtime==4.13.2", "pylatexenc", "kagglehub",
         "huggingface_hub", "pytest", "pandas", "pyarrow"],
    ]
    lines = []
    for i, cmd in enumerate(cmds):
        code, out = run(cmd, f"install_{i}")
        lines.append(f"- exit {code}: `{' '.join(cmd[3:8])}`")
        if code:
            return "\n".join(lines) + "\n\n" + verdict(False) + block(tail(out))
    # A `flash_attn` module without `flash_attn.ops` makes vLLM's loader fail ("No module named 'flash_attn.ops'"):
    # vLLM imports flash_attn.ops whenever a `flash_attn` module is importable. The module may come from a package not
    # named flash-attn (e.g. a FlashAttention-4 / CuTe build that only ships flash_attn/cute), so find the owners and
    # remove them when ops is missing. vLLM ships its own flash-attention; Unsloth trains with xformers here.
    probe = (
        "import importlib.metadata as md, importlib.util as u, json\n"
        "owners = md.packages_distributions().get('flash_attn', [])\n"
        "spec = u.find_spec('flash_attn')\n"
        "try:\n    ops = u.find_spec('flash_attn.ops') is not None if spec else False\n"
        "except Exception as e:\n    ops = False\n"
        "print('FA_PROBE ' + json.dumps({'present': spec is not None, 'ops': ops, 'owners': owners,"
        " 'where': list(spec.submodule_search_locations or []) if spec else []}))\n"
    )
    _, out = run([PY, "-c", probe], "install_probe_flash_attn")
    info = json.loads(next((l[9:] for l in out.splitlines() if l.startswith("FA_PROBE ")), "{}") or "{}")
    lines.append(f"- flash_attn trước khi sửa: `{info}`")
    if info.get("present") and not info.get("ops"):
        for owner in info.get("owners") or ["flash-attn"]:
            code, out = run([PY, "-m", "uv", "pip", "uninstall", "--python", PY, owner], f"install_rm_{owner}")
            lines.append(f"- exit {code}: `uv pip uninstall {owner}`" + (block(tail(out, 8)) if code else ""))
        for where in info.get("where") or []:
            if Path(where).exists() and not (Path(where) / "ops").exists():
                shutil.rmtree(where, ignore_errors=True)       # leftover namespace dir with no owner
                lines.append(f"- xoá thư mục còn sót: `{where}`")
        _, out = run([PY, "-c", probe], "install_probe_flash_attn_after")
        after = json.loads(next((l[9:] for l in out.splitlines() if l.startswith("FA_PROBE ")), "{}") or "{}")
        lines.append(f"- flash_attn sau khi sửa: `{after}`")
        if after.get("present") and not after.get("ops"):
            return "\n".join(lines) + "\n\n" + verdict(False) + "\n\nVẫn còn flash_attn thiếu ops."
    _, freeze = run([PY, "-m", "pip", "freeze"], "install_freeze")
    keep = ("unsloth", "trl", "transformers", "vllm", "torch", "triton", "peft", "accelerate", "math-verify",
            "math_verify", "datasets", "flash")
    env_lines = sorted(l for l in freeze.splitlines() if l.lower().startswith(keep))
    (REPO_DIR / "experiments").mkdir(exist_ok=True)
    (REPO_DIR / "experiments" / "env.txt").write_text("\n".join(env_lines) + "\n")
    return "\n".join(lines) + "\n\n" + verdict(True) + f"\n\nspec: `{INSTALL_SPEC}`" + \
        "\n\nPhiên bản (ghi vào `experiments/env.txt`):" + \
        block("\n".join(env_lines))


def step_gpu() -> str:
    code, out = run([PY, "-c", (
        "import torch, vllm, trl, transformers, unsloth;"
        "print('torch', torch.__version__, 'cuda', torch.version.cuda);"
        "print('gpu', torch.cuda.get_device_name(0), 'capability', torch.cuda.get_device_capability(0));"
        "print('bf16', torch.cuda.is_bf16_supported());"
        "print('vllm', vllm.__version__, 'trl', trl.__version__, 'transformers', transformers.__version__)")],
        "gpu")
    _, smi = run(["nvidia-smi"], "gpu_smi")
    return verdict(code == 0) + block(tail(out, 15)) + block(tail(smi, 20))


def step_data() -> str:
    code, out = run([PY, "-c", f"import kagglehub; print('PATH=' + kagglehub.dataset_download('{KAGGLE_DATASET}'))"],
                    "data")
    path = next((l[5:] for l in out.splitlines() if l.startswith("PATH=")), None)
    need = ["tracks/luffy8k/train.parquet", "tracks/curated/train.parquet", "sft_then_rl/all.parquet",
            "sft_then_rl/dev.parquet", "tracks/eval_exclude_union.tsv", "bridge_curated_teachers/reward.py"]
    missing = [n for n in need if not path or not (Path(path) / n).exists()]
    if path and not missing:
        (WORK / "data_path.txt").write_text(path)
    return verdict(code == 0 and not missing) + f"\n\ndata: `{path}` · thiếu: {missing or 'không'}" + block(tail(out, 15))


def step_tests() -> str:
    code, out = run([PY, "-m", "pytest", "code/tests", "-q", "-s"], "tests")
    return verdict(code == 0) + block(tail(out, 40))


def step_grad_real() -> str:
    code, out = run([PY, "-m", "train.check_real"], "grad_real")
    res = [l for l in out.splitlines() if l.startswith(("RESULT", "INFO"))] or ["không thấy dòng RESULT"]
    return verdict(code == 0) + "\n\n" + "\n".join(f"- {l}" for l in res) + block(tail(out, 30))


def step_sft_smoke() -> str:
    rows, errs = [], []
    ok = True
    for track, steps, limit, bs in [("curated", 10, 640, 8), ("luffy8k", 3, 192, 4)]:
        code, out = run([PY, "-m", "train.sft", "--track", track, "--max-steps", str(steps), "--limit", str(limit),
                         "--per-device-batch", str(bs), "--report-to", "none", "--no-merge",
                         "--output-dir", str(WORK / "runs" / f"smoke_sft_{track}")], f"sft_smoke_{track}")
        losses = [d["loss"] for d in train_logs(out) if "loss" in d]
        ok &= code == 0
        rows.append(f"| {track} | {code} | {losses[:1]} → {losses[-1:]} |")
        if code:
            errs.append(f"\n**{track}** (cuối log):" + block(tail(out, 40)))
    return verdict(ok) + "\n\n| bộ | exit | loss đầu → cuối |\n|---|---|---|\n" + "\n".join(rows) + \
        f"\n\nLog: `{WORK / 'logs'}/sft_smoke_*.log`" + "".join(errs)


def step_grpo_smoke() -> str:
    rows, errs = [], []
    ok = True
    for track in ["curated", "luffy8k"]:
        code, out = run([PY, "-m", "train.grpo", "--track", track, "--baseline", "grpo", "--max-steps", "2",
                         "--limit", "32", "--prompts-per-rollout", "16", "--per-device-batch", "8",
                         "--report-to", "none", "--output-dir", str(WORK / "runs" / f"smoke_grpo_{track}")],
                        f"grpo_smoke_{track}", extra_env={"CAPSTONE_DEBUG_COMPLETIONS": "2"})
        dbg = [l.strip() for l in out.splitlines() if "DEBUG_COMPLETION" in l]
        if dbg:
            errs.append(f"\n**{track}**, lời giải sinh ra (lần chấm đầu):" + block("\n".join(dbg)))
        logs = [d for d in train_logs(out) if "reward" in d]
        last = logs[-1] if logs else {}
        clipped = last.get("completions/clipped_ratio")
        ok &= code == 0 and (track != "curated" or (clipped is not None and clipped < 0.5))
        rows.append(f"| {track} | {code} | {last.get('reward')} | {last.get('completions/mean_length')} | {clipped} |")
        if code:
            errs.append(f"\n**{track}** (cuối log):" + block(tail(out, 40)))
    return verdict(ok) + "\n\n| bộ | exit | reward | độ dài TB | tỉ lệ bị cắt độ dài |\n|---|---|---|---|---|\n" + \
        "\n".join(rows) + "\n\nTỉ lệ bị cắt độ dài của bộ `curated` phải thấp (model dừng ở `<|im_end|>`). " \
        "Gần 1,0 nghĩa là vLLM không nhận `stop_token_ids`: sửa trước khi chạy dài." + "".join(errs)


def step_resume() -> str:
    user = os.environ.get("HF_USER")
    if not user or not os.environ.get("HF_TOKEN"):
        return verdict(False) + "\n\nThiếu HF_USER hoặc HF_TOKEN trong notebook."
    repo = f"{user}/capstone-smoke-resume"
    out_dir = WORK / "runs" / "smoke_resume"
    base = [PY, "-m", "train.grpo", "--track", "curated", "--baseline", "grpo", "--limit", "32",
            "--prompts-per-rollout", "16", "--per-device-batch", "8", "--save-steps", "1", "--report-to", "none",
            "--hub-repo", repo, "--output-dir", str(out_dir)]
    c1, _o1 = run(base + ["--max-steps", "2"], "resume_first")
    shutil.rmtree(out_dir, ignore_errors=True)          # a new session has no local files
    c2, o2 = run(base + ["--max-steps", "3"], "resume_second")
    resumed = "RESUME from" in o2 and "checkpoint-2" in o2
    return verdict(c1 == 0 and c2 == 0 and resumed) + \
        f"\n\nlần 1 exit {c1} · lần 2 exit {c2} · chạy tiếp từ checkpoint: {resumed}\n\nRepo thử: `{repo}` " \
        "(private, xoá được sau khi qua)" + (block(tail(_o1, 25)) if c1 else "") + block(tail(o2, 25))


def step_anchor() -> str:
    """Base Qwen2.5-3B on MATH-500, greedy, 4,096 tokens, curated (SimpleRL-style ChatML) prompt.

    BRIDGE reports 32.4 for this model but does not publish its evaluation prompt; our first run gave 58.4 (Math-Verify)
    and 55.4 (OAT), close to each other and plausible for Qwen2.5 base with a chat template (Qwen reports 42.6 with
    4-shot prompting). So the BRIDGE number is reported for comparison only. PASS = the evaluation runs and few answers
    hit the length cap; the report shows graded samples so the grader can be audited by eye.
    """
    data = child_env().get("CAPSTONE_DATA")
    if not data:
        return verdict(False) + "\n\nChưa có dữ liệu: chạy bước `data`."
    out = WORK / "runs" / "anchor_base_math500"
    code, log = run([PY, "-m", "evaluation.run", "--model", "Qwen/Qwen2.5-3B", "--track", "curated",
                     "--data", f"{data}/sft_then_rl/all.parquet", "--sources", "math", "--greedy",
                     "--max-tokens", "4096", "--out", str(out)], "anchor")
    summ = json.loads((out / "summary.json").read_text())["per_source"]["math"] if (out / "summary.json").exists() else {}
    score, trunc = summ.get("math_verify"), summ.get("truncated_pct")
    ok = code == 0 and score is not None and trunc is not None and trunc < 10
    samples = []
    gen = out / "generations.jsonl"
    if gen.exists():
        rows = [json.loads(l) for l in gen.read_text().splitlines()]
        for want in (True, False):
            for r in [r for r in rows if r["mv"] == want][:3]:
                samples.append(f"[chấm {'ĐÚNG' if want else 'SAI'}] gold={r['answer']!r} · cuối lời giải: "
                               f"{r['generation'][-140:]!r}")
    return verdict(ok) + f"\n\nMATH-500: Math-Verify **{score}**, OAT {summ.get('oat')}, bị cắt độ dài {trunc}% " \
        f"· BRIDGE báo {BRIDGE_MATH500_BASE} (prompt không công bố, chỉ để đối chiếu)" + \
        ("\n\nMẫu để kiểm bộ chấm:" + block("\n".join(samples)) if samples else "") + \
        (block(tail(log, 20)) if code else "")


def step_speed() -> str:
    report = {}
    for track, steps_total in [("luffy8k", 125), ("curated", 133)]:   # 1 epoch
        t0 = time.time()
        code, out = run([PY, "-m", "train.grpo", "--track", track, "--baseline", "grpo", "--max-steps", "4",
                         "--report-to", "none",
                         "--output-dir", str(WORK / "runs" / f"speed_grpo_{track}")], f"speed_grpo_{track}")
        logs = train_logs(out)
        rt = next((d["train_runtime"] for d in logs if "train_runtime" in d), None)
        sec = rt / 4 if rt else None
        report[f"grpo_{track}"] = {
            "exit": code, "wall_s_incl_load": round(time.time() - t0), "train_runtime_s": rt,
            "sec_per_optimizer_step": sec,
            "mean_completion_tokens": [d.get("completions/mean_length") for d in logs if "completions/mean_length" in d],
            "est_hours_full_run": round(sec * steps_total / 3600, 1) if sec else None,
        }
        if code:
            report[f"grpo_{track}"]["log_tail"] = tail(out, 30)
    code, out = run([PY, "-m", "train.sft", "--track", "luffy8k", "--max-steps", "20", "--per-device-batch", "4",
                     "--report-to", "none", "--no-merge", "--output-dir", str(WORK / "runs" / "speed_sft_luffy8k")],
                    "speed_sft_luffy8k")
    rt = next((d["train_runtime"] for d in train_logs(out) if "train_runtime" in d), None)
    report["sft_luffy8k"] = {"exit": code, "train_runtime_s": rt,
                             "est_hours_full_run": round(rt / 20 * 375 / 3600, 1) if rt else None}
    (REPO_DIR / "experiments").mkdir(exist_ok=True)
    (REPO_DIR / "experiments" / "smoke_speed.json").write_text(json.dumps(report, indent=2))
    ok = all(v["exit"] == 0 for v in report.values())
    return verdict(ok) + "\n\nƯớc lượng giờ cho một lần chạy đầy đủ (GRPO 2 epoch: bộ A 250, bộ B 266 bước cập nhật; " \
        "SFT bộ A 375 bước). Bước đầu gồm cả khởi động vLLM nên số hơi cao." + block(json.dumps(report, indent=2))


def step_profile() -> str:
    """One full-size rollout (1,024 completions) on track B with timers on generation, grading and updates."""
    code, out = run([PY, "-m", "train.grpo", "--track", "curated", "--baseline", "grpo", "--max-steps", "2",
                     "--report-to", "none",
                     "--output-dir", str(WORK / "runs" / "profile_grpo_curated")], "profile_grpo_curated",
                    extra_env={"CAPSTONE_PROFILE": "1"})
    prof = [l.strip() for l in out.splitlines() if l.startswith("PROFILE")]
    timeouts = out.count("Timeout during comparison")
    rt = next((d["train_runtime"] for d in train_logs(out) if "train_runtime" in d), None)
    return verdict(code == 0 and bool(prof)) + f"\n\ntrain_runtime {rt} s cho 1 lượt sinh (2 bước cập nhật) · " \
        f"bộ chấm timeout {timeouts} lần" + block("\n".join(prof) or "không có dòng PROFILE") + \
        (block(tail(out, 30)) if code else "")


def step_report() -> str:
    git = ["git", "-C", str(REPO_DIR), "-c", "user.name=capstone-molab",
           "-c", "user.email=capstone-molab@users.noreply.github.com"]
    files = [f for f in ["experiments/env.txt", "experiments/smoke_speed.json"] if (REPO_DIR / f).exists()]
    if not files:
        return verdict(False) + "\n\nChưa có file báo cáo: chạy `install` và `speed` trước."
    run(git + ["add", *files], "report_add")
    c1, o1 = run(git + ["commit", "-m", "molab smoke: environment versions and speed"], "report_commit")
    c2, o2 = run(git + ["push"], "report_push")
    return verdict(c2 == 0) + f"\n\ncommit exit {c1} · push exit {c2}" + block(tail(o1, 5) + "\n" + tail(o2, 5))


def cleanup_leftovers() -> str:
    """Kill training / vLLM processes left by an earlier run (an interrupted notebook leaves its children running and
    holding GPU memory), and show where the last unfinished step stopped."""
    me = os.getpid()
    ps = subprocess.run(["ps", "-eo", "pid,etime,args"], capture_output=True, text=True).stdout.splitlines()[1:]
    pattern = ("-m train.grpo", "-m train.sft", "-m train.check_real", "-m evaluation.run", "-m train.smoke",
               "vllm", "EngineCore")
    killed = []
    for line in ps:
        parts = line.split(None, 2)
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        pid, etime, args = int(parts[0]), parts[1], parts[2]
        if pid in (me, os.getppid()) or not any(p in args for p in pattern):
            continue
        try:
            os.kill(pid, 9)
            killed.append(f"{pid} ({etime}) {args[:120]}")
        except Exception:
            pass
    md = "### Dọn tiến trình còn sót\n\n" + (block("\n".join(killed)) if killed else "Không có.\n")
    logs = sorted((WORK / "logs").glob("*.log"), key=lambda p: p.stat().st_mtime) if (WORK / "logs").exists() else []
    if killed and logs:
        md += f"\nLog mới nhất trước khi dọn (`{logs[-1].name}`):" + block(tail(logs[-1].read_text(errors="replace"), 30))
    time.sleep(5 if killed else 0)
    _, smi = run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"], "cleanup_smi")
    return md + f"\nGPU đang dùng: `{smi.strip()}`\n"


def step_all() -> str:
    """Every step in order; skip steps that already passed; stop at the first FAIL (later steps depend on it)."""
    clean = cleanup_leftovers()
    print(clean, flush=True)
    parts = [clean]
    for key, label in STEPS:
        if key == "all":
            continue
        done = WORK / "reports" / f"{key}.md"
        stale = (key == "install" and done.exists() and INSTALL_SPEC not in done.read_text()) or \
            (key in FORMAT_STEPS and done.exists() and DATA_FORMAT not in done.read_text()) or \
            (key in PERF_STEPS and done.exists() and PERF_CONFIG not in done.read_text())
        if done.exists() and "**PASS**" in done.read_text() and not stale:
            # already passed in an earlier run (e.g. before the session ended): skip; delete the report to redo it
            note = f"### {key}: đã PASS ở lần chạy trước, bỏ qua\n"
            parts.append(note)
            print(note, flush=True)
            continue
        print(f"\n⏳ Đang chạy `{key}`: {label} …\n", flush=True)
        t0 = time.time()
        body = globals()[f"step_{key}"]()
        md = f"### {key} ({round(time.time() - t0)} s): {label}\n\n{body}\n\n<sub>format: {DATA_FORMAT} · perf: {PERF_CONFIG}</sub>\n"
        (WORK / "reports").mkdir(parents=True, exist_ok=True)
        (WORK / "reports" / f"{key}.md").write_text(md)
        parts.append(md)
        print(md, flush=True)
        if "**PASS**" not in body:
            parts.append(f"**Dừng ở bước `{key}`.** Gửi log phía trên để sửa, rồi chạy lại từ bước này.")
            break
    else:
        parts.append("**Tất cả các bước PASS.**")
    return "\n".join(parts)


# ---------------------------------------------------------------- entry

def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m train.smoke --list | <step>")
    if sys.argv[1] == "--list":
        print(json.dumps(STEPS, ensure_ascii=False))
        return
    step = sys.argv[1]
    fn = globals().get(f"step_{step}")
    if fn is None:
        raise SystemExit(f"unknown step {step!r}; known: {[s for s, _ in STEPS]}")
    t0 = time.time()
    body = fn()
    md = f"### {step} ({round(time.time() - t0)} s)\n\n{body}\n"
    (WORK / "reports").mkdir(parents=True, exist_ok=True)
    (WORK / "reports" / f"{step}.md").write_text(md)
    if step == "all":
        print(body.splitlines()[-1], flush=True)   # the step reports were already printed as they finished
    else:
        print(md, flush=True)


if __name__ == "__main__":
    main()
