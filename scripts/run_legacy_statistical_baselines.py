#!/usr/bin/env python3
"""Run the three archived statistical baselines without legacy DB dependencies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scinumbench_v2.legacy_baselines import (
    LEGACY_STATISTICAL_BASELINES,
    parse_serialized_table,
)
from scinumbench_v2.legacy_zero_shot import example_id, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="JSONL with example_id (or bench_id) and serialized table text",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    records = load_jsonl(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parsed = {}
    parse_failures = 0
    for record in records:
        identifier = example_id(record)
        if identifier in parsed:
            raise ValueError(f"duplicate example id: {identifier}")
        table = parse_serialized_table(record["text"])
        parse_failures += table is None
        parsed[identifier] = table

    for name, detector in LEGACY_STATISTICAL_BASELINES.items():
        predictions = {}
        for identifier, table in parsed.items():
            score, verdict = detector(table)
            predictions[identifier] = {
                "score": float(score),
                "verdict": verdict,
                "pred": int(verdict == "FABRICATED"),
            }
        output = args.output_dir / f"baseline_preds_{name}.json"
        output.write_text(
            json.dumps(predictions, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        positives = sum(row["pred"] for row in predictions.values())
        print(f"wrote {output} ({len(predictions)} rows; positive={positives})")
    print(f"parse failures: {parse_failures}")


if __name__ == "__main__":
    main()

