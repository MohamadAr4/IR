"""CLI evaluation (requirement #8): MAP / Recall / P@10 / nDCG@10 for a model,
optionally comparing the basic pipeline against the refined one.

Examples
--------
    python scripts/evaluate.py --dataset argsme --model bm25 --num-queries 50
    python scripts/evaluate.py --dataset argsme --model bm25 --compare
    python scripts/evaluate.py --dataset argsme --model hybrid_serial --num-queries 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ir_core.eval.evaluate import compare_refinement, evaluate_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="bm25")
    ap.add_argument("--num-queries", type=int, default=50)
    ap.add_argument("--eval-depth", type=int, default=100)
    ap.add_argument("--k1", type=float, default=None)
    ap.add_argument("--b", type=float, default=None)
    ap.add_argument("--compare", action="store_true",
                    help="run baseline vs refined and show the delta")
    ap.add_argument("--refine", action="store_true",
                    help="evaluate with refinement on (single run)")
    args = ap.parse_args()

    def prog(i, n, qid):
        if i % 10 == 0 or i == n:
            print(f"  [{i}/{n}] {qid}", flush=True)

    if args.compare:
        out = compare_refinement(args.dataset, args.model, num_queries=args.num_queries,
                                 eval_depth=args.eval_depth, bm25_k1=args.k1, bm25_b=args.b,
                                 progress=prog)
    else:
        refine_opts = {"enabled": True, "spell": True, "expand": True, "history": False} \
            if args.refine else {"enabled": False}
        out = evaluate_model(args.dataset, args.model, num_queries=args.num_queries,
                             eval_depth=args.eval_depth, bm25_k1=args.k1, bm25_b=args.b,
                             refine_opts=refine_opts, progress=prog)
        out = {"dataset": out["dataset"], "model": out["model"],
               "refinement_enabled": out["refinement_enabled"], "metrics": out["metrics"]}

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
