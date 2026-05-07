from __future__ import annotations

import pytest

try:
    import ray  # noqa: F401
    from verl.experimental.agent_loop.code_localization_loop import CodeLocalizationAgentLoop
except ModuleNotFoundError as exc:
    CodeLocalizationAgentLoop = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

pytestmark = pytest.mark.skipif(
    CodeLocalizationAgentLoop is None,
    reason=f"code localization agent loop dependencies are unavailable: {_IMPORT_ERROR}",
)


def _make_loop(
    *,
    reward_metric: str = "acc_at_k",
    reward_acc_at_k: int | None = None,
    reward_acc_at_k_from_gt: bool = False,
    top_k_predictions: int = 5,
) -> CodeLocalizationAgentLoop:
    loop = object.__new__(CodeLocalizationAgentLoop)
    loop.reward_metric = reward_metric
    loop.reward_acc_at_k = reward_acc_at_k
    loop.reward_acc_at_k_from_gt = reward_acc_at_k_from_gt
    loop.top_k_predictions = top_k_predictions
    return loop


def test_navigator_acc_at_k_reward_uses_fixed_k() -> None:
    loop = _make_loop(reward_acc_at_k=2, top_k_predictions=1)

    result = loop._compute_navigator_acc_at_k_reward(
        predicted_locations=["pkg/other.py:g", "pkg/a.py:f"],
        ground_truth_locations=["pkg/a.py:f"],
        total_tool_calls=4,
        successful_tool_calls=3,
    )

    assert result["reward_metric"] == "acc_at_k"
    assert result["k"] == 2
    assert result["acc_at_k"] == 1.0
    assert result["tool_success_rate"] == 0.75
    assert result["score"] == 1.75
    assert loop._prediction_limit_for_reward(["pkg/a.py:f"]) == 2


def test_navigator_acc_at_k_reward_can_use_ground_truth_size_for_k() -> None:
    loop = _make_loop(reward_acc_at_k_from_gt=True, top_k_predictions=1)
    ground_truth = ["pkg/a.py:f", "pkg/b.py:g", "pkg/c.py:h"]

    result = loop._compute_navigator_acc_at_k_reward(
        predicted_locations=["pkg/z.py:q", "pkg/y.py:r", "pkg/b.py:g"],
        ground_truth_locations=ground_truth,
    )

    assert result["k"] == 3
    assert result["k_from_gt"] is True
    assert result["acc_at_k"] == 0.0
    assert loop._prediction_limit_for_reward(ground_truth) == 3

    covered = loop._compute_navigator_acc_at_k_reward(
        predicted_locations=["pkg/c.py:h", "pkg/a.py:f", "pkg/b.py:g"],
        ground_truth_locations=ground_truth,
    )
    assert covered["acc_at_k"] == 1.0
    assert covered["f1_dice_score"] == 1.0


def test_inspector_acc_at_k_reward_uses_relevant_ground_truth_files() -> None:
    loop = _make_loop(reward_acc_at_k_from_gt=True, top_k_predictions=1)

    result = loop._compute_inspector_acc_at_k_reward(
        predicted_locations=["pkg/b.py:g", "pkg/a.py:f"],
        ground_truth_locations=["pkg/a.py:f", "pkg/b.py:g"],
        explored_files={"pkg/a.py"},
    )

    assert result["discard"] is False
    assert result["f_relevant_size"] == 1
    assert result["k"] == 1
    assert result["acc_at_k"] == 0.0


def test_inspector_acc_at_k_reward_discards_empty_relevant_ground_truth() -> None:
    loop = _make_loop(reward_acc_at_k=2)

    result = loop._compute_inspector_acc_at_k_reward(
        predicted_locations=["pkg/a.py:f"],
        ground_truth_locations=["pkg/b.py:g"],
        explored_files={"pkg/a.py"},
    )

    assert result["discard"] is True
    assert result["score"] is None
    assert result["f_relevant_size"] == 0


def test_reward_metric_and_k_parsing() -> None:
    loop = _make_loop()

    assert loop._normalize_reward_metric("acc@k") == "acc_at_k"
    assert loop._normalize_reward_metric("top-k-accuracy") == "acc_at_k"
    assert loop._parse_reward_acc_at_k("3") == 3
    assert loop._parse_reward_acc_at_k("null") is None
    assert loop._parse_reward_acc_at_k("gt") is None
    assert loop.reward_acc_at_k_from_gt is True
    assert loop._parse_bool("false") is False
    assert loop._parse_bool("true") is True

    with pytest.raises(ValueError):
        loop._normalize_reward_metric("recall")
    with pytest.raises(ValueError):
        loop._parse_reward_acc_at_k("0")
