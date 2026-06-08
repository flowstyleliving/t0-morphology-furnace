#!/usr/bin/env python3
"""Meta-analysis entrypoint for comprehensive shadow-ambiguity per-pair JSON."""
from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUN_PATH = HERE / "comprehensive_run.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("shadow_comprehensive_run", RUN_PATH)
    if spec is None or spec.loader is None:
        raise SystemExit(f"could not load {RUN_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(["meta", *(__import__("sys").argv[1:])])


if __name__ == "__main__":
    main()
