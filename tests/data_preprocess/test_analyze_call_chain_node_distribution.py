from __future__ import annotations

import json
import os
import tempfile
import unittest

from data_preprocess.analyze_call_chain_node_distribution import (
    build_analysis,
    collect_call_chain_records,
)


class AnalyzeCallChainNodeDistributionTest(unittest.TestCase):
    def test_collect_and_summarize_call_chain_records(self) -> None:
        with tempfile.TemporaryDirectory(prefix="call_chain_stats_", dir="/tmp") as tmp_dir:
            call_chain_path = os.path.join(tmp_dir, "repo_a_issue_001_call_chains.json")
            skipped_path = os.path.join(tmp_dir, "repo_b_issue_001_call_chains.json")

            with open(call_chain_path, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "repo": "repo/a",
                        "instance_id": "repo_a_issue_001",
                        "build_config": {"generation_mode": "call_chain"},
                        "components": [
                            {
                                "component_id": "component_0000_entry_0000",
                                "entry_kind": "root",
                                "entry_node_ids": ["pkg/a.py:foo"],
                                "node_ids": ["pkg/a.py:foo", "pkg/a.py:bar", "pkg/a.py:baz"],
                                "graph_edge_list": [["pkg/a.py:foo", "pkg/a.py:bar"]],
                                "source_component_id": "component_0000",
                                "source_component_node_count": 4,
                                "source_component_has_cycle": False,
                                "search_unit_generation_mode": "call_chain",
                            },
                            {
                                "component_id": "component_0001_entry_0000",
                                "entry_kind": "scc_representative",
                                "entry_node_ids": ["pkg/a.py:qux"],
                                "node_ids": ["pkg/a.py:qux"],
                                "graph_edge_list": [],
                                "source_component_id": "component_0001",
                                "source_component_node_count": 1,
                                "source_component_has_cycle": True,
                                "search_unit_generation_mode": "call_chain",
                            },
                        ],
                    },
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

            with open(skipped_path, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "repo": "repo/b",
                        "instance_id": "repo_b_issue_001",
                        "build_config": {"generation_mode": "component_graph"},
                        "components": [
                            {
                                "component_id": "component_0000",
                                "node_ids": ["pkg/b.py:foo", "pkg/b.py:bar"],
                            }
                        ],
                    },
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

            records, file_stats = collect_call_chain_records(tmp_dir)
            summary, repo_summaries = build_analysis(records, file_stats, largest_call_chain_limit=10)

            self.assertEqual(len(records), 2)
            self.assertEqual(file_stats["matched_files"], 2)
            self.assertEqual(file_stats["processed_files"], 1)
            self.assertEqual(file_stats["skipped_non_call_chain_files"], 1)
            self.assertEqual(summary["total_call_chains"], 2)
            self.assertEqual(summary["total_repos"], 1)
            self.assertEqual(summary["global_node_count"]["median"], 2.0)
            self.assertEqual(summary["global_node_count_bucket_histogram"]["1"], 1)
            self.assertEqual(summary["global_node_count_bucket_histogram"]["3"], 1)
            self.assertEqual(summary["global_cycle_counts"]["cyclic"], 1)
            self.assertEqual(repo_summaries[0]["repo"], "repo/a")
            self.assertEqual(repo_summaries[0]["call_chain_count"], 2)
            self.assertEqual(repo_summaries[0]["node_count"]["max"], 3.0)


if __name__ == "__main__":
    unittest.main()
