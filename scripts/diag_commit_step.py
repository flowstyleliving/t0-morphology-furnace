#!/usr/bin/env python3
"""Panel-wide gen-step-of-commit diagnostic (t0 mechanistic gate).

Step-0 showed Qwen 2.5 emits a CoT preamble ("To determine if...") rather
than committing YES/NO at gen_step=1, while Mistral-Nemo commits immediately.
This measures, per model on the pinned ANLI R1 n=200, the gen_step at which
the model LOCKS IN the answer it ultimately gives — deciding whether the
"commit step = gen_step 1" framing (under which the js_no_bos 0.82 and all
RAUQ/SinkProbe/calibrator numbers were scored) is a scoped caveat or a
t0-headline mis-location for reasoning-tuned models.

commit_step = smallest k such that the STRICT parser
(emphatic_closing | answer_prefix | bare_first_word) on decode(ids[:k])
equals parse_yes_no(full generation). 'weak_only' = answer only recoverable
via trailing/last-anywhere tiers (no explicit commitment token). 'abstain'
= no YES/NO at all within --cap. Cap default 128 (generous for ANLI;
adequacy is validated on 2 models before any panel run).

Also emits a per-sample substrate CSV (<out-stem>.persample.csv): one row per
sample (label/gold/final/status/commit_step/commit_bucket/n_gen/surprise_gen1/
first_tok) so ONE run answers commit-locus + the (A) model-error sets + the
(A)/(B) split + the latency-confound (commit_step vs label vs surprise_gen1)
+ per-model minority-n — joinable by sample_idx to the run-03 inter-head
per-sample CSVs (same pinned data, hash-verified). Instrument once; analyse
many.

Usage: .venv/bin/python scripts/diag_commit_step.py --model <slug> --out <json> [--cap 128] [--limit 0]
"""
from __future__ import annotations
import argparse, csv, json, os, statistics, sys, tempfile
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def strict_parse(t, io_module):
    strict = (
        io_module.tier_emphatic_closing,
        io_module.tier_answer_prefix,
        io_module.tier_bare_first_word,
    )
    for fn in strict:
        r = fn(t)
        if r is not None:
            return r
    return None

def bucket(k):
    if k == 1: return "step1"
    if k <= 4: return "step2_4"
    if k <= 16: return "step5_16"
    if k <= 64: return "step17_64"
    return "step65_cap"

def main() -> int:
    import pri_v2_io_plugins as io
    import pri_runtime as pipeline
    from pri_calibrator import _load_calibration_jsonl

    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", default=str(REPO/"experiments/anli-sweep/2026-05-15/run-02/anli_R1_seed20260513_n100.jsonl"))
    p.add_argument("--out", required=True)
    p.add_argument("--cap", type=int, default=128)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    prompts, labels, dh = _load_calibration_jsonl(a.data)
    if a.limit:
        prompts, labels = prompts[:a.limit], labels[:a.limit]
    cfg = pipeline.Config(); cfg.layers_to_probe = ["final"]; cfg.v3_capture = False
    model, tok, proj, li = pipeline.load_model(a.model, cfg)
    strat = io.get_prompt_strategy(a.model)
    cell = {(s, l): 0 for s in ("correct","wrong") for l in (0,1)}
    abstain = {0: 0, 1: 0}
    buckets = {b: 0 for b in ("step1","step2_4","step5_16","step17_64","step65_cap","weak_only")}
    commit_ks, samples, rows = [], [], []
    for i,(pr,lab) in enumerate(zip(prompts, labels)):
        lab = int(lab)
        try:
            tr = pipeline.trace_sample(model=model, tokenizer=tok, prompt=strat(pr,tok),
                layer_indices=li, output_projection=proj, max_new_tokens=a.cap, v3_capture=False)
        except Exception as e:
            print(f"[diag] {i} FAIL {e}"); continue
        ids = tr.get("gen_token_ids") or []
        full = tr.get("generated_text") or ""
        final = io.parse_yes_no(full)
        gold = "NO" if lab == 1 else "YES"
        gs = tr.get("gen_surprises") or []
        surprise_gen1 = float(gs[0]) if gs else float("nan")
        first_tok = pipeline.decode_ids(tok, ids[:1]).replace(chr(10), " ")[:24] if ids else ""
        if final is None:
            abstain[lab] += 1; cstep = None; cb = "abstain"; status = "abstain"
        else:
            status = "correct" if final == gold else "wrong"
            cell[(status, lab)] += 1
            cstep = None
            for k in range(1, len(ids)+1):
                if strict_parse(pipeline.decode_ids(tok, ids[:k]), io) == final:
                    cstep = k; break
            cb = bucket(cstep) if cstep else "weak_only"
            buckets[cb] += 1
            if cstep: commit_ks.append(cstep)
        rows.append({"sample_idx": i, "label": lab, "gold": gold,
                     "final": final or "", "status": status,
                     "commit_step": cstep if cstep else "", "commit_bucket": cb,
                     "n_gen": len(ids), "surprise_gen1": surprise_gen1,
                     "first_tok": first_tok})
        if i < 10:
            samples.append({"idx": i, "label": lab, "gold": gold, "final": final,
                            "commit_step": cstep, "bucket": cb, "n_gen": len(ids),
                            "raw": full[:140].replace(chr(10)," ")})
        if (i+1) % 50 == 0: print(f"[diag] {a.model} {i+1}/{len(prompts)}")
    n = len(prompts); committed = sum(buckets[b] for b in buckets)
    out = {
        "model": a.model, "data_hash": dh, "n_total": n, "cap": a.cap,
        "joint_2x2": {f"{s}|label{l}": cell[(s,l)] for (s,l) in cell},
        "abstain": {"label0": abstain[0], "label1": abstain[1], "total": abstain[0]+abstain[1]},
        "commit_step_buckets": buckets,
        "frac_immediate_step1": round(buckets["step1"]/n, 4),
        "frac_cot_step_gt1": round((committed - buckets["step1"])/n, 4),
        "frac_abstain": round((abstain[0]+abstain[1])/n, 4),
        "median_commit_step": (statistics.median(commit_ks) if commit_ks else None),
        "max_commit_step": (max(commit_ks) if commit_ks else None),
        "n_model_error": cell[("wrong",0)] + cell[("wrong",1)],
        "samples": samples,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    # Per-sample substrate CSV (atomic; sibling to the JSON, schema-stable so
    # it joins by sample_idx to run-03 inter-head per-sample CSVs).
    op = Path(a.out)
    csv_path = op.parent / (op.stem + ".persample.csv")
    cols = ["sample_idx","label","gold","final","status","commit_step",
            "commit_bucket","n_gen","surprise_gen1","first_tok"]
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", newline="", dir=csv_path.parent,
                prefix="."+csv_path.stem+".", suffix=".csv.tmp", delete=False) as f:
            tmp = Path(f.name)
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
            for r in rows: w.writerow(r)
        os.replace(tmp, csv_path)
    except OSError as e:
        if tmp is not None and tmp.exists(): tmp.unlink()
        raise SystemExit(f"failed to write {csv_path}: {e}")
    print(f"[diag] per-sample substrate -> {csv_path} ({len(rows)} rows)")
    print(json.dumps({k: out[k] for k in ("model","commit_step_buckets","frac_immediate_step1","frac_cot_step_gt1","frac_abstain","median_commit_step","max_commit_step","n_model_error")}, indent=2))
    return 0
if __name__ == "__main__": sys.exit(main())
