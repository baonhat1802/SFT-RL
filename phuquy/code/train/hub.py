"""Keep checkpoints on the Hugging Face Hub so a run survives the end of a molab session.

On every Trainer save, checkpoint-<step>/ (LoRA adapter + optimizer + scheduler + RNG + trainer_state) is uploaded to a
private model repo; only the newest KEEP checkpoints are kept there. At start, the newest checkpoint is downloaded
into output_dir and training resumes from it.
"""
from __future__ import annotations

import re
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from transformers import TrainerCallback

KEEP = 2
_CKPT = re.compile(r"^checkpoint-(\d+)$")
# PEFT writes a model card (README.md) whose base_model is the init model. For SFT->RL that is a local path
# (.../merged_bf16), which the Hub rejects ("Invalid metadata in README.md") and the whole upload fails. Resume and
# evaluation never read the card, so it is not uploaded.
_SKIP = ["README.md"]


def _remote_checkpoints(api: HfApi, repo_id: str) -> list[str]:
    try:
        names = {f.path.split("/")[0] for f in api.list_repo_tree(repo_id, recursive=False)}
    except Exception:
        return []
    return sorted((n for n in names if _CKPT.match(n)), key=lambda n: int(_CKPT.match(n).group(1)))


def ensure_repo(repo_id: str) -> None:
    HfApi().create_repo(repo_id, private=True, exist_ok=True, repo_type="model")


def _step(name: str) -> int:
    return int(_CKPT.match(name).group(1))


def fetch_latest(repo_id: str, output_dir: str) -> str | None:
    """Return the newest checkpoint, local or remote, as a local path; None if there is none.

    A local checkpoint newer than the Hub's (same session, its upload failed) is used as is and uploaded now, so the
    Hub catches up. Trainer writes a checkpoint completely before on_save runs, so a local one is never half-written.
    """
    api = HfApi()
    remote = _remote_checkpoints(api, repo_id)
    out = Path(output_dir)
    local = sorted((d.name for d in out.glob("checkpoint-*") if d.is_dir() and _CKPT.match(d.name)
                    and (d / "trainer_state.json").exists()), key=_step) if out.exists() else []
    if local and (not remote or _step(local[-1]) > _step(remote[-1])):
        name = local[-1]
        api.upload_folder(repo_id=repo_id, folder_path=str(out / name), path_in_repo=name,
                          commit_message=f"{name} (local, uploaded on resume)", ignore_patterns=_SKIP)
        return str(out / name)
    if not remote:
        return None
    snapshot_download(repo_id, allow_patterns=[f"{remote[-1]}/*"], local_dir=output_dir)
    return str(out / remote[-1])


class HubCheckpointCallback(TrainerCallback):
    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        self.api = HfApi()

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        name = f"checkpoint-{state.global_step}"
        local = Path(args.output_dir) / name
        if not local.exists():
            return
        self.api.upload_folder(repo_id=self.repo_id, folder_path=str(local), path_in_repo=name,
                               commit_message=f"{name}", ignore_patterns=_SKIP)
        for old in _remote_checkpoints(self.api, self.repo_id)[:-KEEP]:
            self.api.delete_folder(path_in_repo=old, repo_id=self.repo_id, commit_message=f"drop {old}")


def upload_dir(repo_id: str, folder: str, path_in_repo: str) -> None:
    HfApi().upload_folder(repo_id=repo_id, folder_path=folder, path_in_repo=path_in_repo,
                          commit_message=f"upload {path_in_repo}", ignore_patterns=_SKIP)
