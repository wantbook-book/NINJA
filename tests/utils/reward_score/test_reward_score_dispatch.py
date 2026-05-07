# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl.utils.reward_score import default_compute_score, get_default_compute_score
from verl.utils.reward_score.code_localization import compute_score as code_localization_compute_score


def test_default_compute_score_supports_code_localization():
    result = default_compute_score(
        data_source="code_localization",
        solution_str="<result>{\"suspicious\": [{\"location\": \"foo.py:bar\"}]}</result>",
        ground_truth='["foo.py:bar"]',
        extra_info={"role": "navigator", "confirmed_suspicious": ["foo.py:bar"]},
    )

    assert isinstance(result, dict)
    assert result["score"] == 1.0
    assert result["dice"] == 1.0


def test_default_compute_score_supports_code_localization_with_list_ground_truth():
    result = default_compute_score(
        data_source="code_localization",
        solution_str="<result>{\"suspicious\": [{\"location\": \"foo.py:bar\"}]}</result>",
        ground_truth=["foo.py:bar"],
        extra_info={"role": "navigator", "confirmed_suspicious": ["foo.py:bar"]},
    )

    assert isinstance(result, dict)
    assert result["score"] == 1.0
    assert result["dice"] == 1.0


def test_get_default_compute_score_uses_code_localization_dispatch():
    compute_score = get_default_compute_score("code_localization")

    assert compute_score is code_localization_compute_score
