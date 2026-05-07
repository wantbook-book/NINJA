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

import ast
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.repo_search_tools import (
    get_code_of_class_function,
    get_code_of_file_function,
    get_file_classes,
    get_file_functions,
    get_methods_of_class,
)


def _build_mock_repo_structure(file_name: str, source: str) -> dict:
    lines = source.splitlines()
    tree = ast.parse(source)
    classes = []
    functions = []
    class_method_names = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = []
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append(
                        {
                            "name": child.name,
                            "start_line": child.lineno,
                            "end_line": child.end_lineno,
                            "text": lines[child.lineno - 1 : child.end_lineno],
                        }
                    )
                    class_method_names.add(child.name)
            classes.append(
                {
                    "name": node.name,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "text": lines[node.lineno - 1 : node.end_lineno],
                    "methods": methods,
                }
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name not in class_method_names:
            functions.append(
                {
                    "name": node.name,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "text": lines[node.lineno - 1 : node.end_lineno],
                }
            )

    root, leaf = file_name.split("/", 1)
    return {
        root: {
            leaf: {
                "classes": classes,
                "functions": functions,
                "text": lines,
            }
        }
    }


@pytest.fixture
def mock_repo_file(tmp_path, monkeypatch):
    source = '''"""module doc"""

from pkg import thing


def top(a: int, b=1, *, c: str = "x") -> str:
    """Top level function."""
    return str(a)


async def async_top(x):
    """Async top level function."""
    return x


@decorator
class Demo(Base):
    """Demo class."""

    @classmethod
    def make(cls, value: int) -> "Demo":
        """Build a demo."""
        return cls()

    async def run(self, *args, **kwargs):
        """Run the demo."""
        return None


class Plain:
    pass
'''
    instance_id = "demo-instance"
    file_name = "demo/example.py"
    structure = _build_mock_repo_structure(file_name, source)
    payload = {"structure": structure}
    payload_path = tmp_path / f"{instance_id}.json"
    payload_path.write_text(json.dumps(payload))
    monkeypatch.setenv("PROJECT_FILE_LOC", str(tmp_path))
    return {"instance_id": instance_id, "file_name": file_name}


def test_get_methods_of_class_returns_signatures_only(mock_repo_file):
    result = json.loads(
        get_methods_of_class(mock_repo_file["file_name"], "Demo", mock_repo_file["instance_id"])
    )

    assert list(result.keys()) == ["methods"]
    assert result["methods"] == [
        "def make(cls, value: int) -> 'Demo'",
        "async def run(self, *args, **kwargs)",
    ]


def test_get_file_functions_returns_only_module_level_functions(mock_repo_file):
    result = json.loads(get_file_functions(mock_repo_file["file_name"], mock_repo_file["instance_id"]))

    assert list(result.keys()) == ["functions"]
    assert result["functions"] == [
        "def top(a: int, b = 1, *, c: str = 'x') -> str",
        "async def async_top(x)",
    ]


def test_get_file_classes_returns_only_module_level_classes(mock_repo_file):
    result = json.loads(get_file_classes(mock_repo_file["file_name"], mock_repo_file["instance_id"]))

    assert result["file_name"] == mock_repo_file["file_name"]
    assert [item["name"] for item in result["classes"]] == ["Demo", "Plain"]
    assert result["classes"][0]["signature"] == "class Demo(Base)"
    assert result["classes"][0]["docstring"] == "Demo class."
    assert result["classes"][0]["decorators"] == ["decorator"]
    assert result["classes"][1]["signature"] == "class Plain"
    assert result["classes"][1]["docstring"] is None


def test_get_code_of_file_function_returns_function_source(mock_repo_file):
    result = get_code_of_file_function(mock_repo_file["file_name"], "top", mock_repo_file["instance_id"])

    assert result == '\n'.join(
        [
            'def top(a: int, b=1, *, c: str = "x") -> str:',
            '    """Top level function."""',
            "    return str(a)",
        ]
    )


def test_get_code_of_class_function_returns_method_source(mock_repo_file):
    result = get_code_of_class_function(mock_repo_file["file_name"], "Demo", "make", mock_repo_file["instance_id"])

    assert result == '\n'.join(
        [
            '    def make(cls, value: int) -> "Demo":',
            '        """Build a demo."""',
            "        return cls()",
        ]
    )


def test_repo_search_functions_return_json_errors_for_invalid_targets(mock_repo_file):
    missing_file = json.loads(get_file_functions("demo/missing.py", mock_repo_file["instance_id"]))
    missing_class = json.loads(
        get_methods_of_class(mock_repo_file["file_name"], "MissingClass", mock_repo_file["instance_id"])
    )
    missing_function = get_code_of_file_function(mock_repo_file["file_name"], "missing_func", mock_repo_file["instance_id"])
    missing_method = get_code_of_class_function(
        mock_repo_file["file_name"], "Demo", "missing_method", mock_repo_file["instance_id"]
    )

    assert "wrong file name" in missing_file["error"]
    assert "wrong file name or class name" in missing_class["error"]
    assert "wrong file name or function name" in missing_function
    assert "wrong file name or class name or function name" in missing_method
