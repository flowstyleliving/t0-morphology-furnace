#!/usr/bin/env python3
"""Compatibility shim for the historical PRI MLX pipeline module.

The canonical reusable runtime now lives in `pri_runtime.py`. This module
preserves the long-lived import path across the repo while keeping plotting
and experiment CLI concerns out of the hot-path runtime import.
"""

from __future__ import annotations

import os

import pri_runtime as _runtime
from pri_experiment_figures import make_figures


def __getattr__(name: str):
    return getattr(_runtime, name)


def __dir__():
    return sorted(set(dir(_runtime) + ["make_figures", "main"]))


__all__ = [name for name in dir(_runtime) if not name.startswith("__")] + [
    "make_figures",
    "main",
]


def main() -> int:
    cfg = _runtime.cfg
    _runtime.print_header("START")

    results_df, dataset_df = _runtime.run_experiment(cfg)

    summary_df = _runtime.run_analysis(results_df, cfg)
    summary_path = _runtime.write_frame(summary_df, os.path.join(cfg.save_dir, "summary"))
    failures_df = _runtime.log_failure_cases(results_df, cfg)

    make_figures(results_df, cfg)

    _runtime.print_header("COMPLETE")
    print(f"  Dataset samples: {len(dataset_df)}")
    print(f"  Result rows: {len(results_df)}")
    print(f"  Summary file: {summary_path}")
    print(f"  Failure rows: {len(failures_df)}")
    print(f"  Output dir: {cfg.save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
