#!/usr/bin/env python3
"""Run the archived first-token Qwen zero-shot self-judge."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from scinumbench_v2.legacy_zero_shot import (
    build_chat_prompts,
    example_id,
    load_jsonl,
    parse_yes_no,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--hf-home", type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    if args.hf_home:
        os.environ["HF_HOME"] = str(args.hf_home)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    records = load_jsonl(args.input)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
    ).to(args.device).eval()

    def token_ids(words: list[str]) -> set[int]:
        identifiers = set()
        for word in words:
            for variant in (word, " " + word):
                encoded = tokenizer.encode(variant, add_special_tokens=False)
                if len(encoded) == 1:
                    identifiers.add(encoded[0])
        return identifiers

    yes_ids = token_ids(["Yes", "yes", "YES", "Y"])
    no_ids = token_ids(["No", "no", "NO", "N"])
    if not yes_ids or not no_ids:
        raise RuntimeError("tokenizer did not expose both Yes and No single-token variants")

    prompts = build_chat_prompts(records, tokenizer)
    predictions = []
    parse_failures = 0
    for start in range(0, len(records), args.batch_size):
        batch_records = records[start : start + args.batch_size]
        batch_prompts = prompts[start : start + args.batch_size]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        ).to(args.device)
        with torch.no_grad():
            logits = model(**encoded).logits[:, -1, :].float()

        for row, record in enumerate(batch_records):
            generated_token = tokenizer.decode([int(logits[row].argmax().item())])
            verdict = parse_yes_no(generated_token)
            yes_logit = max(float(logits[row, token].item()) for token in yes_ids)
            no_logit = max(float(logits[row, token].item()) for token in no_ids)
            maximum = max(yes_logit, no_logit)
            yes_exp = math.exp(yes_logit - maximum)
            no_exp = math.exp(no_logit - maximum)
            probability = yes_exp / (yes_exp + no_exp)
            if not math.isfinite(probability):
                parse_failures += 1
                probability = float(verdict == "FABRICATED")
            predictions.append(
                {
                    "example_id": example_id(record),
                    "verdict": verdict,
                    "pred": int(verdict == "FABRICATED"),
                    "p_fab": probability,
                }
            )
        print(f"processed {min(start + args.batch_size, len(records))}/{len(records)}")

    result = {
        "model": args.model,
        "n": len(predictions),
        "parse_fail": parse_failures,
        "per_example": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

