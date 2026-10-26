"""Shared prompt and parsing helpers for the archived zero-shot baselines."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ZERO_SHOT_INSTRUCTION = (
    "Below is a table from a scientific paper. Decide whether it contains a "
    "numerical/statistical inconsistency (e.g. a mean impossible for the reported N, "
    "a confidence interval that does not contain its point estimate, percentages "
    "inconsistent with the counts, or a p-value contradicting the confidence interval). "
    "Answer with a single word: Yes or No."
)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "text" not in record:
                raise ValueError(f"{path}:{line_number}: missing text field")
            records.append(record)
    return records


def example_id(record: dict[str, Any]) -> str:
    for field in ("example_id", "bench_id"):
        if field in record:
            return str(record[field])
    raise ValueError("input row must contain example_id or bench_id")


def parse_yes_no(text: str) -> str:
    """Apply the verdict parser used by the archived Gemma/Qwen runs."""

    match = re.search(r"\b(yes|no)\b", text.strip().lower())
    if match:
        return "FABRICATED" if match.group(1) == "yes" else "PLAUSIBLE"
    return (
        "FABRICATED"
        if re.search(r"fabricat|inconsist", text.lower())
        else "PLAUSIBLE"
    )


def build_chat_prompts(records: list[dict[str, Any]], tokenizer: Any) -> list[str]:
    prompts = []
    for record in records:
        message = [{"role": "user", "content": ZERO_SHOT_INSTRUCTION + "\n\n" + record["text"]}]
        prompts.append(
            tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
        )
    return prompts

