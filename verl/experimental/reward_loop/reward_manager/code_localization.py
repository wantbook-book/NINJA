"""
Reward manager for code localization multi-agent RL training.

Handles:
- Navigator reward: DICE(S_pred*, S_gt) + ToolSuccessRate
- Inspector reward: DICE(L_pred*, F_relevant) + ToolSuccessRate
- Discarding Inspector trajectories where F_relevant is empty
"""

import inspect
import logging
import os

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score.code_localization import compute_score as default_compute_score

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("code_localization")
class CodeLocalizationRewardManager(RewardManagerBase):
    """Reward manager for code localization.

    Computes DICE + ToolSuccessRate rewards for Navigator or Inspector
    trajectories. Inspector trajectories with empty F_relevant are discarded
    (reward_score=None).
    """

    def __init__(self, config, tokenizer, compute_score=None, **kwargs):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or default_compute_score
        self.is_async = inspect.iscoroutinefunction(self.compute_score)

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]

        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        if tool_extra_fields is not None:
            extra_info.update(tool_extra_fields.items())

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        if self.is_async:
            result = await self.compute_score(
                data_source="code_localization",
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source="code_localization",
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                ),
            )

        reward_extra_info = {}
        if isinstance(result, dict):
            score = result.get("score")
            discard = result.get("discard", False)
            reward_extra_info = {k: v for k, v in result.items()}

            if discard or score is None:
                # Inspector trajectory with empty F_relevant → discard
                score = 0.0
                reward_extra_info["discarded"] = True
                logger.info("Discarding Inspector trajectory (F_relevant is empty)")
        else:
            score = float(result) if result is not None else 0.0
            reward_extra_info["acc"] = score

        return {"reward_score": score, "reward_extra_info": reward_extra_info}
