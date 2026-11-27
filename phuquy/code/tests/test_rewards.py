"""The training reward must accept every reference solution and reject the same solution with a wrong answer."""
import pytest

from train.data import load_track
from train.rewards import correctness_reward


def _break_answer(text: str) -> str:
    i = text.rfind("\\boxed{")
    j, depth = i + 7, 1
    while depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[:i] + "\\boxed{987654321}" + text[j:]


@pytest.mark.parametrize("track", ["luffy8k", "curated"])
def test_reward_on_reference_and_broken(track):
    df = load_track(track).sample(50, random_state=0)
    sols = [t[0]["content"] for t in df["target"]]
    gts = [r["ground_truth"] for r in df["reward_model"]]
    conv = [[{"role": "assistant", "content": s}] for s in sols]
    assert correctness_reward(None, conv, gts) == [1.0] * 50
    broken = [[{"role": "assistant", "content": _break_answer(s)}] for s in sols]
    assert correctness_reward(None, broken, gts) == [0.0] * 50
    assert correctness_reward(None, ["no box here"] * 2, gts[:2]) == [0.0, 0.0]
