# Visual Analysis

This directory contains a browser-based viewer for `evaluation/inference.py`
output directories.

## Files

- `index.html`
  Static single-page viewer for one inference output directory.
- `analyse_gt_num.py`
  Standalone utility for plotting the distribution of
  `reward_model.ground_truth` lengths from a parquet file.

## What The Viewer Supports

The viewer is designed for one selected inference output directory such as:

```text
<base_output_dir>/<run_name>/<timestamp>/
```

It automatically detects and visualizes these files when present:

- `args.json`
- `traj/trajs.jsonl`
- `traj/traj_sample.json`
- `summary/<instance_id>.json`
- `summary/components/<instance_id>/<component_id>.json`

## How To Use

You can open the page directly in a browser, or serve it locally:

```bash
python -m http.server -d visual_analysis 8000
```

Then open:

```text
http://127.0.0.1:8000/
```

In the page:

1. Use the folder picker to select one inference output directory.
2. Choose an `instance_id` if the directory contains per-instance artifacts.
3. Switch the view mode to inspect different files.

## Views

- `Run Overview`
  Shows run-level config, discovered files, and one row per detected instance.
- `Trajectory`
  Shows the main-session messages saved in `traj/trajs.jsonl` for the selected
  instance.
- `Component Summary`
  Shows the merged per-instance summary from `summary/<instance_id>.json`.
- `Component Detail`
  Shows one component's full saved raw trace from
  `summary/components/<instance_id>/<component_id>.json`, including the search
  messages, the per-component `<trace_locs>` block, plus any saved
  component-summary generation or merge trace.
- `Raw File`
  Shows the raw contents of any detected file in the selected directory.

## Notes

- The viewer reads files locally in the browser. No server-side parsing is
  required.
- The folder picker relies on `webkitdirectory`, so Chromium-based browsers are
  the most reliable choice.
- In `bfs_component_graph` runs, the `Trajectory` view shows the main session
  saved to `trajs.jsonl`. Per-component raw traces are shown in
  `Component Detail`, and any saved component-summary trajectory is shown there
  as additional message sections.
