"""Evaluate every available model on a dataset and emit a Markdown table plus a
JSON dump. Used to populate docs/REPORT.md (requirement #8).

    python scripts/run_full_eval.py --dataset argsme --num-queries 49
    python scripts/run_full_eval.py --dataset argsme --num-queries 49 --refine-compare
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows consoles default to cp1252, which can't encode characters like the
# delta sign; force UTF-8 so printing tables never crashes.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from ir_core.engine import ENGINE
from ir_core.eval.evaluate import compare_refinement, evaluate_model

MODEL_ORDER = ["tfidf", "bm25", "word2vec", "bert", "hybrid_parallel", "hybrid_serial"]
METRICS = ["MAP", "Recall", "P@10", "nDCG@10"]


def md_table(rows: list[dict], cols: list[str], head: str) -> str:
    out = [f"| {head} | " + " | ".join(cols) + " |",
           "|" + "---|" * (len(cols) + 1)]
    for r in rows:
        out.append(f"| {r['name']} | " +
                   " | ".join(f"{r[c]:.4f}" if isinstance(r.get(c), float) else str(r.get(c, '')) for c in cols) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--num-queries", type=int, default=49)
    ap.add_argument("--eval-depth", type=int, default=100)
    ap.add_argument("--refine-compare", action="store_true",
                    help="also produce the before/after-refinement table (BM25)")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    models = [m for m in MODEL_ORDER if m in ENGINE.available_models(args.dataset)]
    print(f"Evaluating models: {models}\n", flush=True)

    rows, raw = [], {}
    for m in models:
        print(f"--- {m} ---", flush=True)
        r = evaluate_model(args.dataset, m, num_queries=args.num_queries,
                           eval_depth=args.eval_depth,
                           refine_opts={"enabled": False},
                           progress=lambda i, n, q: print(f"  [{i}/{n}]", flush=True) if i % 10 == 0 else None)
        rows.append({"name": m, **r["metrics"]})
        raw[m] = r["metrics"]
        print(f"  {r['metrics']}\n", flush=True)

    print("\n### Per-model (basic pipeline)\n")
    table = md_table(rows, METRICS + ["num_queries"], "Model")
    print(table)

    compare_table = None
    if args.refine_compare:
        print("\n### Before/after query refinement\n", flush=True)
        crows = []
        for m in [x for x in ("bm25", "tfidf") if x in models]:
            c = compare_refinement(args.dataset, m, num_queries=args.num_queries,
                                   eval_depth=args.eval_depth)
            for phase in ("before", "after"):
                crows.append({"name": f"{m} ({phase})", **c[phase]})
            crows.append({"name": f"{m} (delta)", **c["delta"]})
            raw[f"compare_{m}"] = c
        compare_table = md_table(crows, METRICS, "Model / phase")

    # write the JSON before printing, so a console-encoding hiccup can't lose it
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"per_model": raw, "table": table,
                       "compare_table": compare_table}, f, indent=2)
        print(f"\nWrote {args.out}")
    if compare_table:
        print("\n### Before/after query refinement\n")
        print(compare_table)


if __name__ == "__main__":
    main()
