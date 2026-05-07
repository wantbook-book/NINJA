from __future__ import annotations

import sys
import types
import unittest


datasets_stub = types.ModuleType("datasets")
datasets_stub.Dataset = object
sys.modules["datasets"] = datasets_stub

verl_stub = types.ModuleType("verl")
verl_utils_stub = types.ModuleType("verl.utils")
verl_hdfs_stub = types.ModuleType("verl.utils.hdfs_io")
verl_hdfs_stub.copy = lambda *args, **kwargs: None
verl_hdfs_stub.makedirs = lambda *args, **kwargs: None
sys.modules["verl"] = verl_stub
sys.modules["verl.utils"] = verl_utils_stub
sys.modules["verl.utils.hdfs_io"] = verl_hdfs_stub

tools_stub = types.ModuleType("tools")
repo_search_pkg_stub = types.ModuleType("tools.RepoSearch")
preprocess_data_stub = types.ModuleType("tools.RepoSearch.preprocess_data")
preprocess_data_stub.filter_out_test_files = lambda structure: structure
preprocess_data_stub.show_project_structure = lambda structure: ""
preprocess_data_stub.filter_none_python = lambda structure: structure
sys.modules["tools"] = tools_stub
sys.modules["tools.RepoSearch"] = repo_search_pkg_stub
sys.modules["tools.RepoSearch.preprocess_data"] = preprocess_data_stub

from examples.data_preprocess.swebench_tool_agent_loop_reposearch import (
    _build_output_data,
    _resolve_ground_truth,
    _resolve_ground_truth_details,
)


class SwebenchToolAgentLoopRepoSearchTest(unittest.TestCase):
    def test_resolve_ground_truth_collapses_nested_class_function_to_outer_method(self) -> None:
        project_structure = {
            "astropy": {
                "wcs": {
                    "wcs.py": {
                        "classes": [],
                        "functions": [],
                        "text": [
                            "class WCS:",
                            "    def _array_converter(self):",
                            "        def _return_list_of_arrays():",
                            "            return []",
                            "        def _return_single_array():",
                            "            return None",
                            "        return _return_single_array()",
                        ],
                    }
                }
            }
        }

        example = {
            "patch": "",
            "edit_functions": [
                "astropy/wcs/wcs.py:WCS._array_converter._return_list_of_arrays",
                "astropy/wcs/wcs.py:WCS._array_converter._return_single_array",
            ],
        }

        self.assertEqual(
            _resolve_ground_truth(example, project_structure),
            ["astropy/wcs/wcs.py:WCS._array_converter"],
        )

    def test_resolve_ground_truth_collapses_nested_file_function_to_outer_function(self) -> None:
        project_structure = {
            "requests": {
                "models.py": {
                    "classes": [],
                    "functions": [],
                    "text": [
                        "def generate():",
                        "    def iter_content():",
                        "        return b''",
                        "    return iter_content()",
                    ],
                }
            }
        }

        example = {
            "patch": "",
            "edit_functions": ["requests/models.py:iter_content"],
        }

        self.assertEqual(
            _resolve_ground_truth(example, project_structure),
            ["requests/models.py:generate"],
        )

    def test_resolve_ground_truth_requires_exact_file_match(self) -> None:
        project_structure = {
            "repo_root": {
                "requests": {
                    "models.py": {
                        "classes": [],
                        "functions": [],
                        "text": [
                            "def generate():",
                            "    return b''",
                        ],
                    }
                }
            }
        }

        example = {
            "patch": "",
            "edit_functions": ["requests/models.py:generate"],
        }

        self.assertEqual(_resolve_ground_truth(example, project_structure), [])

        details = _resolve_ground_truth_details(example, project_structure)
        self.assertEqual(
            details["edit_function_resolution_issues"],
            [
                {
                    "location": "requests/models.py:generate",
                    "file_path": "requests/models.py",
                    "reason": "exact_file_path_not_found",
                }
            ],
        )

    def test_build_output_data_stores_ground_truth_details_in_reward_model(self) -> None:
        project_record = {
            "astropy": {
                "wcs": {
                    "wcs.py": {
                        "classes": [],
                        "functions": [],
                        "text": [
                            "class WCS:",
                            "    def _array_converter(self):",
                            "        def _return_list_of_arrays():",
                            "            return []",
                            "        return _return_list_of_arrays()",
                        ],
                    }
                }
            }
        }
        example = {
            "instance_id": "astropy__astropy-12907",
            "repo": "astropy/astropy",
            "base_commit": "abc123",
            "problem_statement": "Fix nested WCS array conversion behavior.",
            "patch": "",
            "edit_functions": [
                "astropy/wcs/wcs.py:WCS._array_converter._return_list_of_arrays",
            ],
        }
        details = _resolve_ground_truth_details(example, project_record)

        data = _build_output_data(
            example,
            split="test",
            original_index=7,
            data_source="mock/source",
            project_record=project_record,
            ground_truth_details=details,
        )

        self.assertEqual(data["reward_model"]["ground_truth"], ["astropy/wcs/wcs.py:WCS._array_converter"])
        self.assertEqual(data["reward_model"]["ground_truth_source"], "edit_functions")
        self.assertEqual(
            data["reward_model"]["edit_function_locations"],
            ["astropy/wcs/wcs.py:WCS._array_converter._return_list_of_arrays"],
        )
        self.assertEqual(data["extra_info"]["instance_id"], "astropy__astropy-12907")
        self.assertEqual(data["extra_info"]["split"], "test")
        self.assertTrue(data["extra_info"]["need_tools_kwargs"])


if __name__ == "__main__":
    unittest.main()
