#!/usr/bin/env python3
"""Score archived baseline prediction files against an explicit label mapping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def load_labels(path: Path, split: str | None) -> dict[str, int]:
    labels = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if split and row.get("split") and row["split"] != split:
                continue
            identifier = str(row["example_id"])
            if identifier in labels:
                raise ValueError(f"{path}:{line_number}: duplicate example_id")
            labels[identifier] = int(row["label"])
    return labels


def normalize_predictions(payload: Any) -> dict[str, dict[str, Any]]:
    if isinstance(payload, dict) and "per_example" in payload:
        rows = payload["per_example"]
        normalized = {}
        for row in rows:
            identifier = str(row["example_id"])
            normalized[identifier] = {
                "score": row.get("score", row.get("p_fab")),
                "pred": row.get("pred", int(row.get("verdict") == "FABRICATED")),
            }
        return normalized
    if isinstance(payload, dict) and "predictions" in payload:
        payload = payload["predictions"]
    if not isinstance(payload, dict):
        raise ValueError("prediction file must contain an id-keyed object or per_example list")
    normalized = {}
    for identifier, row in payload.items():
        if not isinstance(row, dict):
            raise ValueError(f"prediction for {identifier} is not an object")
        normalized[str(identifier)] = {
            "score": row.get("score", row.get("p_fab")),
            "pred": row.get("pred", int(row.get("verdict") == "FABRICATED")),
        }
    return normalized


def score(labels: dict[str, int], predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    matched = [identifier for identifier in labels if identifier in predictions]
    missing = [identifier for identifier in labels if identifier not in predictions]
    extra = [identifier for identifier in predictions if identifier not in labels]
    if not matched:
        raise ValueError("labels and predictions have no matching example IDs")
    valid = [
        identifier
        for identifier in matched
        if predictions[identifier]["score"] is not None
        and predictions[identifier]["pred"] in (0, 1)
    ]
    if not valid:
        raise ValueError("no matched predictions contain both score and binary prediction")
    y_true = np.asarray([labels[identifier] for identifier in valid], dtype=int)
    y_score = np.asarray(
        [predictions[identifier]["score"] for identifier in valid], dtype=float
    )
    y_pred = np.asarray(
        [predictions[identifier]["pred"] for identifier in valid], dtype=int
    )
    true_positive = int(((y_pred == 1) & (y_true == 1)).sum())
    false_positive = int(((y_pred == 1) & (y_true == 0)).sum())
    false_negative = int(((y_pred == 0) & (y_true == 1)).sum())
    true_negative = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": len(valid),
        "n_labels": len(labels),
        "n_predictions": len(predictions),
        "n_missing": len(missing),
        "n_extra": len(extra),
        "n_invalid": len(matched) - len(valid),
        "AUROC": float(roc_auc_score(y_true, y_score)),
        "AUPRC": float(average_precision_score(y_true, y_score)),
        "accuracy": float((y_true == y_pred).mean()),
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "TP": true_positive,
        "FP": false_positive,
        "FN": false_negative,
        "TN": true_negative,
        "missing_example_ids": missing,
        "extra_prediction_ids": extra,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--split", default="test", help="mapping split; use '' for all rows")
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="repeat for each prediction JSON",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    labels = load_labels(args.labels, args.split or None)
    results = []
    for specification in args.prediction:
        if "=" not in specification:
            raise ValueError("--prediction must use NAME=PATH")
        name, raw_path = specification.split("=", 1)
        path = Path(raw_path)
        predictions = normalize_predictions(json.loads(path.read_text(encoding="utf-8")))
        row = {"baseline": name, "prediction_file": path.name}
        row.update(score(labels, predictions))
        results.append(row)
        print(
            f"{name}: n={row['n']} AUROC={row['AUROC']:.6f} "
            f"AUPRC={row['AUPRC']:.6f}"
        )

    report = {
        "label_file": args.labels.name,
        "split": args.split or None,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

