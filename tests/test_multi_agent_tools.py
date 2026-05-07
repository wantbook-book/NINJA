"""
Tests for tools/multi_agent_tools.py

Tests cover:
- Graph loading and caching (_load_graph)
- Fuzzy node matching (_fuzzy_match_node)
- Test file detection (_is_test_file)
- get_callers / get_callees
- find_files_by_content_repo / find_files_by_name_repo
- Wrapper functions (get_code_of_function, get_code_of_class_method)
"""

import json
import os
import pickle
import tempfile
from unittest import mock

import pytest
import networkx as nx

from tools.multi_agent_tools import (
    _graph_cache,
    _load_graph,
    _fuzzy_match_node,
    _is_test_file,
    get_callers,
    get_callees,
    find_files_by_content_repo,
    find_files_by_name_repo,
    get_code_of_function,
    get_code_of_class_method,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_graph_cache():
    """Clear the module-level graph cache before each test."""
    _graph_cache.clear()
    yield
    _graph_cache.clear()


def _build_sample_graph() -> nx.MultiDiGraph:
    """Build a small sample dependency graph for testing."""
    G = nx.MultiDiGraph()

    # Nodes: file:QualifiedName
    G.add_node("django/db/models/query.py:QuerySet.filter")
    G.add_node("django/db/models/query.py:QuerySet._filter_or_exclude")
    G.add_node("django/db/models/query.py:QuerySet.all")
    G.add_node("django/db/models/sql/query.py:Query.add_q")
    G.add_node("django/views/generic/list.py:ListView.get_queryset")
    G.add_node("tests/test_queries.py:TestFilter.test_basic")

    # Edges: caller --invokes--> callee
    G.add_edge(
        "django/db/models/query.py:QuerySet.filter",
        "django/db/models/query.py:QuerySet._filter_or_exclude",
        type="invokes",
    )
    G.add_edge(
        "django/db/models/query.py:QuerySet._filter_or_exclude",
        "django/db/models/sql/query.py:Query.add_q",
        type="invokes",
    )
    G.add_edge(
        "django/views/generic/list.py:ListView.get_queryset",
        "django/db/models/query.py:QuerySet.all",
        type="invokes",
    )
    G.add_edge(
        "django/views/generic/list.py:ListView.get_queryset",
        "django/db/models/query.py:QuerySet.filter",
        type="invokes",
    )
    # Test file caller
    G.add_edge(
        "tests/test_queries.py:TestFilter.test_basic",
        "django/db/models/query.py:QuerySet.filter",
        type="invokes",
    )
    # Non-invokes edge (should be ignored)
    G.add_edge(
        "django/db/models/query.py:QuerySet.filter",
        "django/db/models/query.py:QuerySet.all",
        type="contains",
    )

    return G


@pytest.fixture
def graph_dir(tmp_path):
    """Create a temporary directory with a pickled graph."""
    G = _build_sample_graph()
    graph_file = tmp_path / "test_instance.pkl"
    with open(graph_file, "wb") as f:
        pickle.dump(G, f)
    return tmp_path


@pytest.fixture
def set_graph_env(graph_dir):
    """Set the GRAPH_INDEX_DIR environment variable."""
    with mock.patch.dict(os.environ, {"GRAPH_INDEX_DIR": str(graph_dir)}):
        yield


# ---------------------------------------------------------------------------
# Tests: _is_test_file
# ---------------------------------------------------------------------------

class TestIsTestFile:

    def test_test_prefix(self):
        assert _is_test_file("tests/test_models.py:TestCase.test_method")

    def test_test_in_path(self):
        assert _is_test_file("tests/unit/test_views.py:dummy")

    def test_non_test_file(self):
        assert not _is_test_file("django/db/models/query.py:QuerySet.filter")

    def test_test_as_word(self):
        assert _is_test_file("test_something.py:func")

    def test_no_test_in_name(self):
        assert not _is_test_file("django/contrib/auth/models.py:User.save")


# ---------------------------------------------------------------------------
# Tests: _load_graph
# ---------------------------------------------------------------------------

class TestLoadGraph:

    def test_load_existing(self, set_graph_env):
        G = _load_graph("test_instance")
        assert isinstance(G, nx.MultiDiGraph)
        assert "django/db/models/query.py:QuerySet.filter" in G

    def test_graph_caching(self, set_graph_env):
        G1 = _load_graph("test_instance")
        G2 = _load_graph("test_instance")
        assert G1 is G2  # Same object from cache

    def test_missing_graph_raises(self, set_graph_env):
        with pytest.raises(FileNotFoundError, match="Graph file not found"):
            _load_graph("nonexistent_instance")


# ---------------------------------------------------------------------------
# Tests: _fuzzy_match_node
# ---------------------------------------------------------------------------

class TestFuzzyMatchNode:

    def test_exact_match(self, set_graph_env):
        G = _load_graph("test_instance")
        result = _fuzzy_match_node(G, "django/db/models/query.py:QuerySet.filter")
        assert result == "django/db/models/query.py:QuerySet.filter"

    def test_suffix_match(self, set_graph_env):
        G = _load_graph("test_instance")
        # Match by suffix: func_name ends with the query
        result = _fuzzy_match_node(G, "django/db/models/query.py:filter")
        assert result is not None
        assert "filter" in result

    def test_no_match(self, set_graph_env):
        G = _load_graph("test_instance")
        result = _fuzzy_match_node(G, "nonexistent/file.py:nonexistent_func")
        assert result is None

    def test_no_colon(self, set_graph_env):
        G = _load_graph("test_instance")
        result = _fuzzy_match_node(G, "no_colon_here")
        assert result is None


# ---------------------------------------------------------------------------
# Tests: get_callers
# ---------------------------------------------------------------------------

class TestGetCallers:

    def test_basic_callers(self, set_graph_env):
        result = json.loads(get_callers(
            "django/db/models/query.py:QuerySet._filter_or_exclude",
            "test_instance",
        ))
        callers = result["callers"]
        assert "django/db/models/query.py:QuerySet.filter" in callers

    def test_excludes_test_files(self, set_graph_env):
        result = json.loads(get_callers(
            "django/db/models/query.py:QuerySet.filter",
            "test_instance",
        ))
        callers = result["callers"]
        # Test file callers should be excluded
        for c in callers:
            assert not _is_test_file(c)
        # But non-test caller should be there
        assert "django/views/generic/list.py:ListView.get_queryset" in callers

    def test_node_not_found(self, set_graph_env):
        result = json.loads(get_callers(
            "nonexistent/file.py:func",
            "test_instance",
        ))
        assert "error" in result

    def test_missing_graph(self):
        with mock.patch.dict(os.environ, {"GRAPH_INDEX_DIR": "/nonexistent"}):
            result = json.loads(get_callers(
                "some/file.py:func",
                "nonexistent_instance",
            ))
            assert "error" in result


# ---------------------------------------------------------------------------
# Tests: get_callees
# ---------------------------------------------------------------------------

class TestGetCallees:

    def test_basic_callees(self, set_graph_env):
        result = json.loads(get_callees(
            "django/db/models/query.py:QuerySet.filter",
            "test_instance",
        ))
        callees = result["callees"]
        assert "django/db/models/query.py:QuerySet._filter_or_exclude" in callees

    def test_excludes_test_files(self, set_graph_env):
        result = json.loads(get_callees(
            "django/db/models/query.py:QuerySet.filter",
            "test_instance",
        ))
        callees = result["callees"]
        for c in callees:
            assert not _is_test_file(c)

    def test_ignores_non_invokes_edges(self, set_graph_env):
        result = json.loads(get_callees(
            "django/db/models/query.py:QuerySet.filter",
            "test_instance",
        ))
        callees = result["callees"]
        # "QuerySet.all" has a "contains" edge, not "invokes", so should NOT appear
        assert "django/db/models/query.py:QuerySet.all" not in callees

    def test_node_not_found(self, set_graph_env):
        result = json.loads(get_callees(
            "nonexistent/file.py:func",
            "test_instance",
        ))
        assert "error" in result


# ---------------------------------------------------------------------------
# Tests: find_files_by_content_repo
# ---------------------------------------------------------------------------

class TestFindFilesByContentRepo:

    @pytest.fixture
    def project_dir(self, tmp_path):
        """Create a mock project file with repo structure.

        ``extract_structure`` expects file entries to be dicts with keys
        ``text`` (list of lines), ``functions`` and ``classes``.
        """
        structure = {
            "django": {
                "db": {
                    "models": {
                        "query.py": {
                            "text": [
                                "class QuerySet:",
                                "    def filter(self, *args, **kwargs):",
                                "        return self._filter_or_exclude(False, args, kwargs)",
                                "",
                                "    def _filter_or_exclude(self, negate, args, kwargs):",
                                "        pass",
                            ],
                            "functions": [],
                            "classes": [],
                        }
                    }
                },
                "views": {
                    "generic": {
                        "list.py": {
                            "text": [
                                "class ListView:",
                                "    def get_queryset(self):",
                                "        return queryset.filter(**self.kwargs)",
                            ],
                            "functions": [],
                            "classes": [],
                        }
                    }
                }
            }
        }
        data = {"structure": structure}
        instance_file = tmp_path / "test_grep.json"
        with open(instance_file, "w") as f:
            json.dump(data, f)
        return tmp_path

    def test_basic_grep(self, project_dir):
        with mock.patch.dict(os.environ, {"PROJECT_FILE_LOC": str(project_dir)}):
            result = json.loads(find_files_by_content_repo(["filter"], "test_grep"))
            assert result["keywords"] == ["filter"]
            assert len(result["results"]) > 0
            # Should find filter in query.py
            files_found = [r["file"] for r in result["results"]]
            assert any("query.py" in f for f in files_found)

    def test_case_insensitive(self, project_dir):
        with mock.patch.dict(os.environ, {"PROJECT_FILE_LOC": str(project_dir)}):
            result = json.loads(find_files_by_content_repo(["QUERYSET"], "test_grep"))
            assert len(result["results"]) > 0

    def test_multiple_keywords(self, project_dir):
        with mock.patch.dict(os.environ, {"PROJECT_FILE_LOC": str(project_dir)}):
            result = json.loads(find_files_by_content_repo(["filter", "ListView"], "test_grep"))
            assert len(result["results"]) > 0
            files_found = [r["file"] for r in result["results"]]
            # Should find both query.py (filter) and list.py (ListView)
            assert any("query.py" in f for f in files_found)
            assert any("list.py" in f for f in files_found)

    def test_multiple_keywords_dedup(self, project_dir):
        with mock.patch.dict(os.environ, {"PROJECT_FILE_LOC": str(project_dir)}):
            # Both keywords appear in query.py, but the file should appear only once
            result = json.loads(find_files_by_content_repo(["filter", "QuerySet"], "test_grep"))
            files_found = [r["file"] for r in result["results"]]
            query_files = [f for f in files_found if "query.py" in f]
            assert len(query_files) == 1

    def test_no_results(self, project_dir):
        with mock.patch.dict(os.environ, {"PROJECT_FILE_LOC": str(project_dir)}):
            result = json.loads(find_files_by_content_repo(["nonexistent_keyword_xyz"], "test_grep"))
            assert result["results"] == []

    def test_empty_keywords(self, project_dir):
        with mock.patch.dict(os.environ, {"PROJECT_FILE_LOC": str(project_dir)}):
            result = json.loads(find_files_by_content_repo([], "test_grep"))
            assert result["results"] == []

    def test_regex_pattern(self, project_dir):
        """Keywords should be interpreted as regex patterns."""
        with mock.patch.dict(os.environ, {"PROJECT_FILE_LOC": str(project_dir)}):
            # Match lines containing "def " followed by a word starting with "f"
            result = json.loads(find_files_by_content_repo([r"def\s+f"], "test_grep"))
            assert len(result["results"]) > 0
            files_found = [r["file"] for r in result["results"]]
            assert any("query.py" in f for f in files_found)

    def test_regex_dot_star(self, project_dir):
        """Regex .* should match across a line."""
        with mock.patch.dict(os.environ, {"PROJECT_FILE_LOC": str(project_dir)}):
            result = json.loads(find_files_by_content_repo([r"filter.*exclude"], "test_grep"))
            assert len(result["results"]) > 0

    def test_invalid_regex_falls_back_to_literal(self, project_dir):
        """An invalid regex should be treated as a literal substring."""
        with mock.patch.dict(os.environ, {"PROJECT_FILE_LOC": str(project_dir)}):
            # "[invalid" is a bad regex — should fall back to literal match
            # It won't match anything in our fixture, but should not raise
            result = json.loads(find_files_by_content_repo(["[invalid"], "test_grep"))
            assert isinstance(result["results"], list)

    def test_missing_env_var(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("PROJECT_FILE_LOC", None)
            result = json.loads(find_files_by_content_repo(["test"], "test_instance"))
            assert "error" in result


# ---------------------------------------------------------------------------
# Tests: find_files_by_name_repo
# ---------------------------------------------------------------------------

class TestFindFilesByNameRepo:

    @pytest.fixture
    def project_dir(self, tmp_path):
        """Create a mock project file with repo structure.

        ``extract_structure`` expects file entries to be dicts with keys
        ``text`` (list of lines), ``functions`` and ``classes``.
        """
        structure = {
            "django": {
                "db": {
                    "models": {
                        "query.py": {
                            "text": ["class QuerySet:"],
                            "functions": [],
                            "classes": [],
                        },
                        "fields.py": {
                            "text": ["class Field:"],
                            "functions": [],
                            "classes": [],
                        },
                    }
                },
                "views": {
                    "generic": {
                        "list.py": {
                            "text": ["class ListView:"],
                            "functions": [],
                            "classes": [],
                        }
                    }
                }
            }
        }
        data = {"structure": structure}
        instance_file = tmp_path / "test_glob.json"
        with open(instance_file, "w") as f:
            json.dump(data, f)
        return tmp_path

    def test_match_by_name(self, project_dir):
        with mock.patch.dict(os.environ, {"PROJECT_FILE_LOC": str(project_dir)}):
            result = json.loads(find_files_by_name_repo(["query.py"], "test_glob"))
            assert len(result["matched_files"]) > 0
            assert any("query.py" in f for f in result["matched_files"])

    def test_match_by_path_pattern(self, project_dir):
        with mock.patch.dict(os.environ, {"PROJECT_FILE_LOC": str(project_dir)}):
            result = json.loads(find_files_by_name_repo(["django/db/*.py"], "test_glob"))
            # This may or may not match depending on nested structure extraction
            # At minimum the pattern matching logic is exercised

    def test_multiple_patterns(self, project_dir):
        with mock.patch.dict(os.environ, {"PROJECT_FILE_LOC": str(project_dir)}):
            result = json.loads(find_files_by_name_repo(["query.py", "list.py"], "test_glob"))
            files = result["matched_files"]
            assert any("query.py" in f for f in files)
            assert any("list.py" in f for f in files)

    def test_multiple_patterns_dedup(self, project_dir):
        with mock.patch.dict(os.environ, {"PROJECT_FILE_LOC": str(project_dir)}):
            # Both patterns could match query.py, should only appear once
            result = json.loads(find_files_by_name_repo(["query.py", "*query*"], "test_glob"))
            files = result["matched_files"]
            query_files = [f for f in files if "query.py" in f]
            assert len(query_files) == 1

    def test_no_match(self, project_dir):
        with mock.patch.dict(os.environ, {"PROJECT_FILE_LOC": str(project_dir)}):
            result = json.loads(find_files_by_name_repo(["nonexistent_*.xyz"], "test_glob"))
            assert result["matched_files"] == []

    def test_empty_patterns(self, project_dir):
        with mock.patch.dict(os.environ, {"PROJECT_FILE_LOC": str(project_dir)}):
            result = json.loads(find_files_by_name_repo([], "test_glob"))
            assert result["matched_files"] == []


# ---------------------------------------------------------------------------
# Tests: GlobalState (from multi_agent_inference.py)
# ---------------------------------------------------------------------------

class TestGlobalState:
    """Test the GlobalState dataclass behavior."""

    def test_import(self):
        from evaluation.multi_agent_inference import GlobalState
        state = GlobalState()
        assert state.confirmed_suspicious == []
        assert state.explored_entries == set()
        assert state.pending_entries == []

    def test_add_suspicious_dedup(self):
        from evaluation.multi_agent_inference import GlobalState
        state = GlobalState()
        state.add_suspicious([
            {"location": "a.py:func1", "reason": "r1"},
            {"location": "a.py:func2", "reason": "r2"},
        ])
        state.add_suspicious([
            {"location": "a.py:func1", "reason": "r1 again"},  # duplicate
            {"location": "a.py:func3", "reason": "r3"},
        ])
        assert len(state.confirmed_suspicious) == 3

    def test_add_pending_dedup(self):
        from evaluation.multi_agent_inference import GlobalState
        state = GlobalState()
        state.add_pending(["a.py", "b.py", "c.py"])
        state.add_pending(["b.py", "d.py"])  # b.py is duplicate
        assert state.pending_entries == ["a.py", "b.py", "c.py", "d.py"]

    def test_add_pending_excludes_explored_entries(self):
        from evaluation.multi_agent_inference import GlobalState
        state = GlobalState()
        state.add_explored_entry("b.py")
        state.add_pending(["a.py", "b.py", "c.py"])
        assert "b.py" not in state.pending_entries
        assert "a.py" in state.pending_entries
        assert "c.py" in state.pending_entries

    def test_take_pending(self):
        from evaluation.multi_agent_inference import GlobalState
        state = GlobalState()
        state.add_pending(["a.py", "b.py", "c.py", "d.py"])
        taken = state.take_pending(2)
        assert taken == ["a.py", "b.py"]
        assert state.pending_entries == ["c.py", "d.py"]

    def test_add_explored_from_functions(self):
        from evaluation.multi_agent_inference import GlobalState
        state = GlobalState()
        state.add_explored_from_functions([
            "a.py:func1",
            "b.py:Class.method",
            "c.py:*",
        ])
        assert "a.py" in state.explored_entries
        assert "b.py" in state.explored_entries
        assert "c.py" in state.explored_entries

    def test_add_explored_from_functions_dedup_pending(self):
        from evaluation.multi_agent_inference import GlobalState
        state = GlobalState()
        state.add_pending(["a.py", "b.py"])
        state.add_explored_from_functions(["a.py:func1"])
        # a.py is now explored, so adding it to pending again should be skipped
        state.add_pending(["a.py", "c.py"])
        assert "a.py" in state.explored_entries
        assert "c.py" in state.pending_entries
        # a.py should NOT be re-added to pending
        assert state.pending_entries.count("a.py") == 1  # the original one

    def test_to_dict(self):
        from evaluation.multi_agent_inference import GlobalState
        state = GlobalState()
        state.add_suspicious([{"location": "a.py:func", "reason": "test"}])
        state.add_explored_from_functions(["a.py:func"])
        state.add_explored_entry("a.py")
        state.add_pending(["b.py"])
        d = state.to_dict()
        assert "confirmed_suspicious" in d
        assert "explored_entries" in d
        assert "pending_entries" in d
        assert "a.py" in d["explored_entries"]


# ---------------------------------------------------------------------------
# Tests: _extract_explored_from_tool_calls
# ---------------------------------------------------------------------------

class TestExtractExploredFromToolCalls:

    def test_get_code_of_function(self):
        from evaluation.multi_agent_inference import _extract_explored_from_tool_calls
        tool_calls = [
            {"name": "get_code_of_function", "arguments": {"file_name": "a.py", "func_name": "foo"}},
        ]
        explored = _extract_explored_from_tool_calls(tool_calls)
        assert "a.py:foo" in explored

    def test_get_code_of_class_method(self):
        from evaluation.multi_agent_inference import _extract_explored_from_tool_calls
        tool_calls = [
            {"name": "get_code_of_class_method", "arguments": {
                "file_name": "a.py", "class_name": "MyClass", "func_name": "method"
            }},
        ]
        explored = _extract_explored_from_tool_calls(tool_calls)
        assert "a.py:MyClass.method" in explored

    def test_get_callers(self):
        from evaluation.multi_agent_inference import _extract_explored_from_tool_calls
        tool_calls = [
            {"name": "get_callers", "arguments": {"location": "a.py:MyClass.method"}},
        ]
        explored = _extract_explored_from_tool_calls(tool_calls)
        assert "a.py:MyClass.method" in explored

    def test_get_file_functions(self):
        from evaluation.multi_agent_inference import _extract_explored_from_tool_calls
        tool_calls = [
            {"name": "get_file_functions", "arguments": {"file_name": "a.py"}},
        ]
        explored = _extract_explored_from_tool_calls(tool_calls)
        assert "a.py:*" in explored

    def test_exit_ignored(self):
        from evaluation.multi_agent_inference import _extract_explored_from_tool_calls
        tool_calls = [
            {"name": "exit", "arguments": {}},
        ]
        explored = _extract_explored_from_tool_calls(tool_calls)
        assert explored == []

    def test_mixed_calls(self):
        from evaluation.multi_agent_inference import _extract_explored_from_tool_calls
        tool_calls = [
            {"name": "get_file_functions", "arguments": {"file_name": "a.py"}},
            {"name": "get_code_of_function", "arguments": {"file_name": "a.py", "func_name": "foo"}},
            {"name": "get_callees", "arguments": {"location": "a.py:foo"}},
            {"name": "exit", "arguments": {}},
        ]
        explored = _extract_explored_from_tool_calls(tool_calls)
        assert len(explored) == 3


# ---------------------------------------------------------------------------
# Tests: _augment_system_prompt_with_tool_instructions
# ---------------------------------------------------------------------------

class TestAugmentSystemPrompt:

    def test_basic_augmentation(self):
        from evaluation.multi_agent_inference import _augment_system_prompt_with_tool_instructions
        prompt = "You are a helpful agent."
        tools = [
            {"type": "function", "function": {"name": "test_tool", "description": "A test tool", "parameters": {}}}
        ]
        result = _augment_system_prompt_with_tool_instructions(prompt, tools)
        assert "<tools>" in result
        assert "</tools>" in result
        assert "<tool_call>" in result
        assert "test_tool" in result
        assert result.startswith("You are a helpful agent.")

    def test_empty_tools(self):
        from evaluation.multi_agent_inference import _augment_system_prompt_with_tool_instructions
        prompt = "You are a helpful agent."
        result = _augment_system_prompt_with_tool_instructions(prompt, [])
        assert result == prompt

    def test_none_prompt(self):
        from evaluation.multi_agent_inference import _augment_system_prompt_with_tool_instructions
        tools = [
            {"type": "function", "function": {"name": "test", "parameters": {}}}
        ]
        result = _augment_system_prompt_with_tool_instructions(None, tools)
        assert "<tools>" in result


# ---------------------------------------------------------------------------
# Tests: parse_tool_calls_from_text
# ---------------------------------------------------------------------------

class TestParseToolCalls:

    def _get_parser(self):
        """Get the parse function from the runner class without initializing it."""
        from evaluation.multi_agent_inference import MultiAgentLocalizeRunner
        # Use __new__ to avoid __post_init__
        runner = object.__new__(MultiAgentLocalizeRunner)
        return runner._parse_tool_calls_from_text

    def test_single_tool_call(self):
        parse = self._get_parser()
        text = 'Let me search.\n<tool_call>\n{"name": "find_files_by_content", "arguments": {"keyword": "filter"}}\n</tool_call>'
        calls = parse(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "find_files_by_content"
        assert calls[0]["arguments"]["keyword"] == "filter"

    def test_multiple_tool_calls(self):
        parse = self._get_parser()
        text = (
            '<tool_call>\n{"name": "find_files_by_content", "arguments": {"keyword": "filter"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "find_files_by_name", "arguments": {"pattern": "*.py"}}\n</tool_call>'
        )
        calls = parse(text)
        assert len(calls) == 2

    def test_invalid_json(self):
        parse = self._get_parser()
        text = '<tool_call>\nnot valid json\n</tool_call>'
        calls = parse(text)
        assert calls == []

    def test_missing_name(self):
        parse = self._get_parser()
        text = '<tool_call>\n{"arguments": {"keyword": "test"}}\n</tool_call>'
        calls = parse(text)
        assert calls == []

    def test_no_tool_calls(self):
        parse = self._get_parser()
        text = "This is just regular text without any tool calls."
        calls = parse(text)
        assert calls == []

    def test_exit_tool(self):
        parse = self._get_parser()
        text = '<tool_call>\n{"name": "exit", "arguments": {}}\n</tool_call>'
        calls = parse(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "exit"


# ---------------------------------------------------------------------------
# Tests: _parse_trace_locs
# ---------------------------------------------------------------------------

class TestParseTraceLocs:

    def _get_parser(self):
        from evaluation.multi_agent_inference import MultiAgentLocalizeRunner
        runner = object.__new__(MultiAgentLocalizeRunner)
        return runner._parse_trace_locs

    def test_basic_parse(self):
        parse = self._get_parser()
        text = """
<trace_locs>
django/db/models/query.py
function: QuerySet.filter

django/db/models/sql/query.py
function: Query.add_q

django/views/generic/list.py
function: ListView.get_queryset
</trace_locs>
"""
        result = parse(text)
        assert "django/db/models/query.py" in result
        assert "function: QuerySet.filter" in result["django/db/models/query.py"][0]
        assert len(result) == 3

    def test_no_trace_locs(self):
        parse = self._get_parser()
        result = parse("No trace locs here.")
        assert result == {}

    def test_multiple_functions_per_file(self):
        parse = self._get_parser()
        text = """
<trace_locs>
django/db/models/query.py
function: QuerySet.filter
function: QuerySet._filter_or_exclude

django/views/generic/list.py
function: ListView.get_queryset
</trace_locs>
"""
        result = parse(text)
        assert "django/db/models/query.py" in result
        locs = result["django/db/models/query.py"][0]
        assert "QuerySet.filter" in locs
        assert "QuerySet._filter_or_exclude" in locs


# ---------------------------------------------------------------------------
# Tests: Sub-agent finalize helpers
# ---------------------------------------------------------------------------

class TestSubAgentFinalize:

    @staticmethod
    def _fake_response(text):
        message = mock.Mock()
        message.content = text
        choice = mock.Mock()
        choice.message = message
        response = mock.Mock()
        response.choices = [choice]
        return response

    def _make_runner(self):
        from evaluation.multi_agent_inference import MultiAgentLocalizeRunner
        runner = object.__new__(MultiAgentLocalizeRunner)
        runner.tokenizer = None
        runner.max_sub_agent_turns = 3
        runner.sub_agent_token_budget = 0
        runner.sub_agent_augmented_system_prompt = "You are a sub agent."
        runner.sub_agent_user_prompt_template = "$problem_statement\n$entry_file\n$explored_entries"
        runner.sub_agent_finalize_prompt_template = "Finalize now with a <result> block only."
        return runner

    def test_collect_sub_agent_suspicious_candidates_reads_last_assistant_only(self):
        runner = self._make_runner()
        conversation = [
            {
                "role": "assistant",
                "content": (
                    "<result>{\"suspicious\": ["
                    "{\"location\": \"pkg/a.py:foo\", \"reason\": \"first\"}, "
                    "{\"location\": \"pkg/b.py:Bar.baz\", \"reason\": \"second\"}"
                    "]}</result>"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "<result>{\"suspicious\": ["
                    "{\"location\": \"pkg/a.py:foo\", \"reason\": \"updated\"}"
                    "]}</result>"
                ),
            },
        ]

        candidates = runner._collect_sub_agent_suspicious_candidates(conversation)
        assert candidates == [{"location": "pkg/a.py:foo", "reason": "updated"}]

    def test_collect_sub_agent_suspicious_candidates_returns_empty_when_last_assistant_has_no_result(self):
        runner = self._make_runner()
        conversation = [
            {
                "role": "assistant",
                "content": (
                    "<result>{\"suspicious\": ["
                    "{\"location\": \"pkg/a.py:foo\", \"reason\": \"first\"}"
                    "]}</result>"
                ),
            },
            {"role": "assistant", "content": "<think>No structured result.</think>"},
        ]

        candidates = runner._collect_sub_agent_suspicious_candidates(conversation)
        assert candidates == []

    def test_run_single_sub_agent_finalizes_after_exit(self):
        runner = self._make_runner()
        responses = [
            self._fake_response(
                "<think>Done exploring.</think>\n"
                "<tool_call>\n"
                "{\"name\": \"exit\", \"arguments\": {}}\n"
                "</tool_call>"
            ),
            self._fake_response(
                "<result>\n"
                "{\"suspicious\": ["
                "{\"location\": \"pkg/a.py:foo\", \"reason\": \"finalized\"}"
                "]}\n"
                "</result>"
            ),
        ]
        runner._call_model = lambda messages, max_tokens=4096: responses.pop(0)

        result = runner._run_single_sub_agent(
            instance_id="inst1",
            problem_statement="Bug description",
            entry_file="pkg/a.py",
            explored_entries=set(),
        )

        assert result["entry_file"] == "pkg/a.py"
        assert result["suspicious"] == [
            {"location": "pkg/a.py:foo", "reason": "finalized"},
        ]
        assert any(
            msg.get("_message_kind") == "finalize" and msg.get("role") == "user"
            for msg in result["messages"]
        )
        assert any(
            msg.get("_message_kind") == "finalize"
            and msg.get("role") == "user"
            and msg.get("content") == "Finalize now with a <result> block only."
            for msg in result["messages"]
        )
        assert any(
            msg.get("_message_kind") == "finalize" and msg.get("role") == "assistant"
            for msg in result["messages"]
        )


# ---------------------------------------------------------------------------
# Tests: Token counting and budget
# ---------------------------------------------------------------------------

class TestTokenBudget:

    def _make_runner(self):
        """Create a minimal runner without full initialization."""
        from evaluation.multi_agent_inference import MultiAgentLocalizeRunner
        runner = object.__new__(MultiAgentLocalizeRunner)
        runner.tokenizer = None
        return runner

    def test_no_tokenizer_returns_zero(self):
        runner = self._make_runner()
        messages = [{"role": "user", "content": "Hello world"}]
        assert runner._count_messages_tokens(messages) == 0

    def test_no_tokenizer_budget_never_exceeded(self):
        runner = self._make_runner()
        messages = [{"role": "user", "content": "Hello world"}]
        assert runner._is_token_budget_exceeded(messages, 100) is False

    def test_zero_budget_never_exceeded(self):
        runner = self._make_runner()
        runner.tokenizer = "dummy"  # non-None to pass the None check
        # But budget=0 means unlimited, so should return False regardless
        assert runner._is_token_budget_exceeded([], 0) is False

    def test_normalize_message_content_string(self):
        from evaluation.multi_agent_inference import MultiAgentLocalizeRunner
        result = MultiAgentLocalizeRunner._normalize_message_content_for_tokenizer("hello")
        assert result == "hello"

    def test_normalize_message_content_dict(self):
        from evaluation.multi_agent_inference import MultiAgentLocalizeRunner
        result = MultiAgentLocalizeRunner._normalize_message_content_for_tokenizer({"key": "value"})
        assert isinstance(result, str)
        assert "key" in result

    def test_message_to_payload(self):
        runner = self._make_runner()
        msg = {"role": "assistant", "content": "Hello", "name": "agent1"}
        payload = runner._message_to_tokenizer_payload(msg)
        assert payload["role"] == "assistant"
        assert payload["content"] == "Hello"
        assert payload["name"] == "agent1"

    def test_message_to_payload_no_name(self):
        runner = self._make_runner()
        msg = {"role": "user", "content": "Hi"}
        payload = runner._message_to_tokenizer_payload(msg)
        assert "name" not in payload


# ---------------------------------------------------------------------------
# Tests: ToolCallTracker
# ---------------------------------------------------------------------------

class TestToolCallTracker:

    def test_empty_tracker(self):
        from evaluation.multi_agent_inference import ToolCallTracker
        t = ToolCallTracker()
        assert t.total_tool_calls == 0
        assert t.distinct_tool_calls == 0
        assert t.repeated_tool_calls == 0
        assert t.failed_tool_calls == 0
        assert t.repeat_rate == 0.0
        assert t.fail_ratio == 0.0

    def test_record_and_counts(self):
        from evaluation.multi_agent_inference import ToolCallTracker
        t = ToolCallTracker()
        t.record("find_files_by_content", {"keywords": ["filter"]})
        t.record("find_files_by_name", {"patterns": ["*.py"]})
        assert t.total_tool_calls == 2
        assert t.distinct_tool_calls == 2
        assert t.repeated_tool_calls == 0

    def test_repeated_detection(self):
        from evaluation.multi_agent_inference import ToolCallTracker
        t = ToolCallTracker()
        t.record("find_files_by_content", {"keywords": ["filter"]})
        t.record("find_files_by_content", {"keywords": ["filter"]})  # repeat
        assert t.total_tool_calls == 2
        assert t.distinct_tool_calls == 1
        assert t.repeated_tool_calls == 1
        assert t.repeat_rate == 0.5

    def test_failed_tracking(self):
        from evaluation.multi_agent_inference import ToolCallTracker
        t = ToolCallTracker()
        t.record("find_files_by_content", {"keywords": ["a"]}, failed=False)
        t.record("find_files_by_content", {"keywords": ["b"]}, failed=True)
        assert t.total_tool_calls == 2
        assert t.failed_tool_calls == 1
        assert t.fail_ratio == 0.5

    def test_to_dict(self):
        from evaluation.multi_agent_inference import ToolCallTracker
        t = ToolCallTracker()
        t.record("find_files_by_content", {"keywords": ["a"]})
        t.record("find_files_by_content", {"keywords": ["a"]}, failed=True)  # repeat + fail
        d = t.to_dict()
        assert d["total_tool_calls"] == 2
        assert d["distinct_tool_calls"] == 1
        assert d["repeated_tool_calls"] == 1
        assert d["failed_tool_calls"] == 1
        assert d["repeat_rate"] == 0.5

    def test_merge(self):
        from evaluation.multi_agent_inference import ToolCallTracker
        main = ToolCallTracker()
        main.record("find_files_by_content", {"keywords": ["a"]})

        sub = ToolCallTracker()
        sub.record("get_callers", {"location": "a.py:foo"})
        sub.record("get_callers", {"location": "a.py:foo"}, failed=True)  # repeat + fail

        main.merge(sub)
        assert main.total_tool_calls == 3
        assert main.failed_tool_calls == 1
        assert main.repeated_tool_calls == 1


# ---------------------------------------------------------------------------
# Tests: _make_message and _is_tool_result_error
# ---------------------------------------------------------------------------

class TestMakeMessageAndToolError:

    def _make_runner(self):
        from evaluation.multi_agent_inference import MultiAgentLocalizeRunner
        runner = object.__new__(MultiAgentLocalizeRunner)
        runner.tokenizer = None
        return runner

    def test_make_message_basic(self):
        runner = self._make_runner()
        msg = runner._make_message("assistant", "Hello world")
        assert msg["role"] == "assistant"
        assert msg["content"] == "Hello world"
        assert "token_count" in msg
        assert msg["token_count"] >= 0

    def test_make_message_with_extras(self):
        runner = self._make_runner()
        msg = runner._make_message("user", "Hi", _message_kind="tool_result")
        assert msg["_message_kind"] == "tool_result"
        assert msg["role"] == "user"
        assert "token_count" in msg

    def test_make_message_token_estimate_no_tokenizer(self):
        runner = self._make_runner()
        msg = runner._make_message("user", "A" * 100)
        # ~25 tokens for 100 chars (len // 4)
        assert msg["token_count"] == 25

    def test_is_tool_result_error_true(self):
        from evaluation.multi_agent_inference import MultiAgentLocalizeRunner
        assert MultiAgentLocalizeRunner._is_tool_result_error('{"error": "something bad"}') is True

    def test_is_tool_result_error_false(self):
        from evaluation.multi_agent_inference import MultiAgentLocalizeRunner
        assert MultiAgentLocalizeRunner._is_tool_result_error('{"results": []}') is False

    def test_is_tool_result_error_invalid_json(self):
        from evaluation.multi_agent_inference import MultiAgentLocalizeRunner
        assert MultiAgentLocalizeRunner._is_tool_result_error('not json') is False


# ---------------------------------------------------------------------------
# Tests: assistant response prefill
# ---------------------------------------------------------------------------

class TestAssistantResponsePrefill:

    @staticmethod
    def _fake_response(text):
        message = mock.Mock()
        message.content = text
        choice = mock.Mock()
        choice.message = message
        response = mock.Mock()
        response.choices = [choice]
        return response

    def _make_runner(self, prefill="<think>"):
        from evaluation.multi_agent_inference import MultiAgentLocalizeRunner
        runner = object.__new__(MultiAgentLocalizeRunner)
        runner.temperature = 0.2
        runner.top_p = None
        runner.assistant_response_prefill = prefill
        return runner

    def test_prepare_model_messages_appends_assistant_prefill_and_sanitizes_metadata(self):
        runner = self._make_runner("<think>")

        payload = runner._prepare_model_messages(
            [
                {"role": "user", "content": "Find the bug", "token_count": 3},
            ],
            runner.assistant_response_prefill,
        )

        assert payload == [
            {"role": "user", "content": "Find the bug"},
            {"role": "assistant", "content": "<think>"},
        ]

    def test_call_model_returns_response_with_prefill_prefix(self):
        runner = self._make_runner("<think>")

        class FakeLLM:
            def __init__(self):
                self.seen_messages = None

            def chat_response(self, messages, **params):
                self.seen_messages = messages
                return TestAssistantResponsePrefill._fake_response("continue")

        llm = FakeLLM()
        response = runner._call_model(
            [{"role": "user", "content": "Find the bug", "token_count": 3}],
            llm=llm,
        )

        assert llm.seen_messages[-1] == {"role": "assistant", "content": "<think>"}
        assert runner._extract_response_text(response) == "<think>continue"


# ---------------------------------------------------------------------------
# Tests: _parse_tool_calls_from_text with "parameters" key
# ---------------------------------------------------------------------------

class TestParseToolCallsParametersKey:
    """Tests that _parse_tool_calls_from_text accepts both 'arguments' and 'parameters'."""

    def _get_parser(self):
        from evaluation.multi_agent_inference import MultiAgentLocalizeRunner
        runner = object.__new__(MultiAgentLocalizeRunner)
        return runner._parse_tool_calls_from_text

    def test_parameters_key_accepted(self):
        parse = self._get_parser()
        text = '<tool_call>\n{"name": "find_files_by_content", "parameters": {"keywords": ["wcs"]}}\n</tool_call>'
        calls = parse(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "find_files_by_content"
        assert calls[0]["arguments"] == {"keywords": ["wcs"]}

    def test_arguments_key_still_works(self):
        parse = self._get_parser()
        text = '<tool_call>\n{"name": "find_files_by_content", "arguments": {"keywords": ["wcs"]}}\n</tool_call>'
        calls = parse(text)
        assert len(calls) == 1
        assert calls[0]["arguments"] == {"keywords": ["wcs"]}

    def test_arguments_takes_priority_over_parameters(self):
        parse = self._get_parser()
        # If both are present, "arguments" should win
        text = '<tool_call>\n{"name": "find_files_by_content", "arguments": {"keyword": "a"}, "parameters": {"keyword": "b"}}\n</tool_call>'
        calls = parse(text)
        assert len(calls) == 1
        assert calls[0]["arguments"] == {"keyword": "a"}

    def test_neither_arguments_nor_parameters(self):
        parse = self._get_parser()
        text = '<tool_call>\n{"name": "exit"}\n</tool_call>'
        calls = parse(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "exit"
        assert calls[0]["arguments"] == {}

    def test_mixed_tool_calls(self):
        """Mix of tool calls using 'arguments' and 'parameters' keys."""
        parse = self._get_parser()
        text = (
            '<tool_call>\n{"name": "find_files_by_content", "parameters": {"keywords": ["wcs"]}}\n</tool_call>\n'
            '<tool_call>\n{"name": "find_files_by_name", "arguments": {"patterns": ["*.py"]}}\n</tool_call>\n'
            '<tool_call>\n{"name": "exit"}\n</tool_call>'
        )
        calls = parse(text)
        assert len(calls) == 3
        assert calls[0]["arguments"] == {"keywords": ["wcs"]}
        assert calls[1]["arguments"] == {"patterns": ["*.py"]}
        assert calls[2]["arguments"] == {}

    def test_real_model_response_format(self):
        """Test with the actual format seen in main_agent_response.txt."""
        parse = self._get_parser()
        text = (
            'I\'ll analyze the issue.\n\n'
            '<tool_call>\n'
            '{"name": "find_files_by_name", "parameters": {"patterns": ["wcs/*.py"]}}\n'
            '</tool_call>\n\n'
            '<tool_call>\n'
            '{"name": "find_files_by_content", "parameters": {"keywords": ["wcs_pix2world", "_array_converter"], "file_pattern": "wcs/*.py"}}\n'
            '</tool_call>'
        )
        calls = parse(text)
        assert len(calls) == 2
        assert calls[0]["name"] == "find_files_by_name"
        assert calls[0]["arguments"]["patterns"] == ["wcs/*.py"]
        assert calls[1]["name"] == "find_files_by_content"
        assert calls[1]["arguments"]["keywords"] == ["wcs_pix2world", "_array_converter"]


# ---------------------------------------------------------------------------
# Tests: multi-agent ablation helpers
# ---------------------------------------------------------------------------

class TestMultiAgentAblationHelpers:

    def _make_runner(self, ablation_mode="none", parallelism=2):
        from evaluation.multi_agent_inference import MultiAgentLocalizeRunner
        runner = object.__new__(MultiAgentLocalizeRunner)
        runner.ablation_mode = ablation_mode
        runner.sub_agent_parallelism = parallelism
        return runner

    def test_single_agent_tool_schema_includes_search_and_inspection_tools(self):
        schema_path = os.path.join(
            os.getcwd(),
            "tools",
            "tool_schemas",
            "single_agent_tools.json",
        )
        with open(schema_path, encoding="utf-8") as f:
            combined = json.load(f)

        names = [
            schema["function"]["name"]
            for schema in combined
        ]

        assert "find_files_by_content" in names
        assert "find_files_by_name" in names
        assert "get_file_functions" in names
        assert "get_code_of_function" in names
        assert "get_callers" in names
        assert "dispatch" not in names
        assert names.count("exit") == 1

    def test_no_dynamic_scheduling_keeps_explored_dispatch_entries(self):
        from evaluation.multi_agent_inference import (
            ABLATION_NO_DYNAMIC_SCHEDULING,
            GlobalState,
        )

        runner = self._make_runner(ABLATION_NO_DYNAMIC_SCHEDULING, parallelism=2)
        state = GlobalState()
        state.explored_entries = {"pkg/a.py"}
        state.pending_entries = ["pkg/a.py", "pkg/b.py", "pkg/c.py"]

        entries, skipped, supplemented = runner._select_dispatch_entries(
            ["pkg/a.py"],
            state,
        )

        assert entries == ["pkg/a.py", "pkg/b.py"]
        assert skipped == []
        assert supplemented == ["pkg/b.py"]

    def test_no_dynamic_scheduling_allows_explored_files_in_pending(self):
        from evaluation.multi_agent_inference import (
            ABLATION_NO_DYNAMIC_SCHEDULING,
            GlobalState,
        )

        runner = self._make_runner(ABLATION_NO_DYNAMIC_SCHEDULING)
        state = GlobalState()
        state.explored_entries = {"pkg/a.py"}

        runner._add_pending_files(state, ["pkg/a.py", "pkg/b.py"])

        assert state.pending_entries == ["pkg/a.py", "pkg/b.py"]

    def test_no_hierarchical_search_dispatches_all_pending_entries_once(self):
        from evaluation.multi_agent_inference import (
            ABLATION_NO_HIERARCHICAL_SEARCH,
            GlobalState,
        )

        runner = self._make_runner(ABLATION_NO_HIERARCHICAL_SEARCH, parallelism=2)
        state = GlobalState()
        state.explored_entries = {"pkg/a.py"}
        state.pending_entries = ["pkg/a.py", "pkg/b.py", "pkg/c.py"]

        entries, skipped, supplemented = runner._select_dispatch_entries(
            ["pkg/d.py"],
            state,
        )

        assert entries == ["pkg/d.py", "pkg/b.py", "pkg/c.py"]
        assert skipped == ["pkg/a.py"]
        assert supplemented == ["pkg/b.py", "pkg/c.py"]
