#!/usr/bin/env python3
"""Run the archived Gemma zero-shot judge and save first-token probabilities."""

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
    parser.add_argument("--model", default="google/gemma-3-12b-it")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--hf-home", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    if args.hf_home:
        os.environ["HF_HOME"] = str(args.hf_home)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    records = load_jsonl(args.input)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = build_chat_prompts(records, tokenizer)
    sampling = SamplingParams(temperature=0.0, max_tokens=8, logprobs=20)
    model = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
    )
    outputs = model.generate(prompts, sampling)

    predictions = {}
    probability_fallbacks = 0
    for record, output in zip(records, outputs):
        generated = output.outputs[0].text
        verdict = parse_yes_no(generated)
        probability = None
        try:
            first_token = output.outputs[0].logprobs[0]
            yes_logit = -math.inf
            no_logit = -math.inf
            for logprob in first_token.values():
                token = logprob.decoded_token.strip().lower()
                if token in {"yes", "y"}:
                    yes_logit = max(yes_logit, float(logprob.logprob))
                elif token in {"no", "n"}:
                    no_logit = max(no_logit, float(logprob.logprob))
            if math.isfinite(yes_logit) or math.isfinite(no_logit):
                maximum = max(yes_logit, no_logit)
                yes_exp = math.exp(yes_logit - maximum)
                no_exp = math.exp(no_logit - maximum)
                probability = yes_exp / (yes_exp + no_exp)
        except (IndexError, KeyError, TypeError, ValueError):
            probability = None
        if probability is None:
            probability_fallbacks += 1
            probability = float(verdict == "FABRICATED")
        predictions[example_id(record)] = {
            "score": probability,
            "verdict": verdict,
            "pred": int(verdict == "FABRICATED"),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {args.output} ({len(predictions)} rows; "
        f"probability fallbacks={probability_fallbacks})"
    )


if __name__ == "__main__":
    main()

