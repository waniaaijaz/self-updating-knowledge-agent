#!/usr/bin/env python3
"""Measure contradiction-detection quality on the labeled benchmark.

This is the script that turns the resume bullet from a claim into a number.
Run it, put the number in the bullet, and be ready to explain the threshold
trade-off in the interview.

    python scripts/evaluate.py
    python scripts/evaluate.py --sweep        # precision/recall across thresholds
    python scripts/evaluate.py --offline      # heuristic backend, no downloads
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.nli import get_detector  # noqa: E402

POSITIVE = "CONTRADICTION"


def confusion(scores, labels, threshold):
    tp = fp = fn = tn = 0
    for score, label in zip(scores, labels):
        predicted = score >= threshold
        actual = label == POSITIVE
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def metrics(tp, fp, fn, tn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    return precision, recall, f1, accuracy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--threshold", type=float, default=config.AUTO_SUPERSEDE_THRESHOLD)
    ap.add_argument("--benchmark", default=str(config.DATA_DIR / "benchmark.json"))
    args = ap.parse_args()

    detector = get_detector(force_offline=True if args.offline else None)
    pairs = json.loads(Path(args.benchmark).read_text())["pairs"]
    print(f"Detector: {detector.backend}")
    print(f"Benchmark: {len(pairs)} pairs "
          f"({sum(1 for p in pairs if p['label'] == POSITIVE)} positive)\n")

    scores, labels, rows = [], [], []
    for pair in pairs:
        result = detector.evaluate_bidirectional(pair["old"], pair["new"])
        scores.append(result.contradiction_score)
        labels.append(pair["label"])
        rows.append((pair["id"], pair["label"], result.contradiction_score, result.label))

    tp, fp, fn, tn = confusion(scores, labels, args.threshold)
    precision, recall, f1, accuracy = metrics(tp, fp, fn, tn)

    print(f"{'pair':<22} {'gold':<18} {'score':>7}  {'nli_label':<14} flag")
    print("-" * 72)
    for pid, gold, score, nli_label in rows:
        flag = "FLAG" if score >= args.threshold else ""
        wrong = "  <-- MISS" if (score >= args.threshold) != (gold == POSITIVE) else ""
        print(f"{pid:<22} {gold:<18} {score:>7.3f}  {nli_label:<14} {flag}{wrong}")

    print(f"\nThreshold {args.threshold:.2f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Precision {precision:.3f}   Recall {recall:.3f}   "
          f"F1 {f1:.3f}   Accuracy {accuracy:.3f}")

    if args.sweep:
        print("\nThreshold sweep")
        print(f"{'thr':>5} {'prec':>7} {'recall':>7} {'F1':>7} {'FP':>4} {'FN':>4}")
        best = None
        for i in range(1, 20):
            thr = i / 20
            p, r, f, _ = metrics(*confusion(scores, labels, thr))
            c = confusion(scores, labels, thr)
            print(f"{thr:>5.2f} {p:>7.3f} {r:>7.3f} {f:>7.3f} {c[1]:>4} {c[2]:>4}")
            if best is None or f > best[1]:
                best = (thr, f)
        print(f"\nBest F1 {best[1]:.3f} at threshold {best[0]:.2f}")
        print("Note: for this system, false positives are worse than false negatives —\n"
              "a false positive silently deletes a rule that is still in force, while a\n"
              "false negative just leaves a stale chunk retrievable. That is why the\n"
              "production gate sits above the F1-optimal point, with the band underneath\n"
              "routed to human review instead of discarded.")


if __name__ == "__main__":
    main()
