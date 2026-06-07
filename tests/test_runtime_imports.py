from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _probe(code: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


def test_pri_runtime_import_does_not_load_plotting_modules():
    payload = _probe(
        f"""
import json, sys
sys.path.insert(0, {str(REPO_ROOT)!r})
import pri_runtime
print(json.dumps({{
    "matplotlib.pyplot": "matplotlib.pyplot" in sys.modules,
    "seaborn": "seaborn" in sys.modules,
}}))
"""
    )
    assert payload == {"matplotlib.pyplot": False, "seaborn": False}


def test_pri_v2_shim_import_does_not_load_plotting_modules():
    payload = _probe(
        f"""
import json, sys
sys.path.insert(0, {str(REPO_ROOT)!r})
import pri_v2_mlx_pipeline
print(json.dumps({{
    "matplotlib.pyplot": "matplotlib.pyplot" in sys.modules,
    "seaborn": "seaborn" in sys.modules,
}}))
"""
    )
    assert payload == {"matplotlib.pyplot": False, "seaborn": False}


def test_pri_calibrator_jsonl_helper_does_not_import_runtime(tmp_path):
    data_path = tmp_path / "mini.jsonl"
    data_path.write_text('{"prompt":"hello","label":0}\n{"prompt":"bye","label":1}\n')
    payload = _probe(
        f"""
import json, sys
sys.path.insert(0, {str(REPO_ROOT)!r})
import pri_calibrator
before = "pri_runtime" in sys.modules
pri_calibrator._load_calibration_jsonl({str(data_path)!r})
after = "pri_runtime" in sys.modules
print(json.dumps({{"before": before, "after": after}}))
"""
    )
    assert payload == {"before": False, "after": False}
