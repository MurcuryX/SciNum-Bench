#!/usr/bin/env python3
"""Extract frozen-LLM table representations from a frozen V2 split bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

from scinumbench_v2.freeze_manifest import load_public_extraction_rows, manifest_json


SPLITS = ("train", "validation", "test")
REVISION_RE = re.compile(r"[0-9a-f]{40}")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _weight_manifest(snapshot: Path) -> tuple[str, list[dict[str, object]]]:
    weight_files = sorted(
        path
        for path in snapshot.rglob("*")
        if path.is_file() and path.suffix.lower() in {".safetensors", ".bin", ".pt"}
    )
    if not weight_files:
        raise SystemExit(f"no model weight files found in resolved snapshot {snapshot}")
    entries = [
        {
            "path": str(path.relative_to(snapshot)),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in weight_files
    ]
    return _sha256_text(_canonical_json(entries)), entries


def _tokenizer_hash(tokenizer: object, *, tokenizer_id: str, revision: str) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None and hasattr(backend, "to_str"):
        representation = backend.to_str()
    else:
        vocabulary = getattr(tokenizer, "get_vocab")()
        representation = _canonical_json(sorted(vocabulary.items()))
    payload = {
        "tokenizer_id": tokenizer_id,
        "revision": revision,
        "class": type(tokenizer).__name__,
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
        "backend_sha256": _sha256_text(representation),
    }
    return _sha256_text(_canonical_json(payload))


def _load_rows(
    split_dir: Path,
    manifest: dict[str, object],
) -> list[dict[str, object]]:
    """Load labeled train/validation plus the frozen label-blind test view."""

    partitions = load_public_extraction_rows(manifest, split_dir)
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for split in SPLITS:
        for record in partitions[split]:
            example_id = str(record["example_id"])
            if example_id in seen:
                raise SystemExit(f"duplicate example_id: {example_id}")
            seen.add(example_id)
            rows.append(record)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True, help="exact 40-hex HF commit")
    parser.add_argument("--tokenizer", help="defaults to --model")
    parser.add_argument("--tokenizer-revision", help="defaults to --model-revision")
    parser.add_argument("--layers", required=True, help="comma-separated hidden-state indices")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hf-home", default="/data/shared_models/huggingface")
    parser.add_argument("--allow-truncation", action="store_true")
    args = parser.parse_args()

    tokenizer_id = args.tokenizer or args.model
    tokenizer_revision = args.tokenizer_revision or args.model_revision
    if REVISION_RE.fullmatch(args.model_revision) is None:
        raise SystemExit("--model-revision must be an exact lowercase 40-hex commit")
    if REVISION_RE.fullmatch(tokenizer_revision) is None:
        raise SystemExit("--tokenizer-revision must be an exact lowercase 40-hex commit")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.model):
        raise SystemExit("--model must be a stable Hugging Face repository ID, not a local path")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", tokenizer_id):
        raise SystemExit("--tokenizer must be a stable Hugging Face repository ID")

    if args.batch_size <= 0 or args.max_length <= 0:
        raise SystemExit("batch size and max length must be positive")
    layers = tuple(int(item) for item in args.layers.split(",") if item.strip())
    if not layers or len(set(layers)) != len(layers) or any(layer <= 0 for layer in layers):
        raise SystemExit("--layers must contain unique positive integers")

    os.environ["HF_HOME"] = args.hf_home
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import numpy as np
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = _load_rows(args.split_dir, manifest)
    dataset_manifest_sha = _sha256_text(manifest_json(manifest))
    test_inputs = manifest.get("test_inputs")
    if not isinstance(test_inputs, dict):
        raise SystemExit("manifest lacks frozen label-blind test_inputs")

    texts = [str(row["text"]) for row in rows]
    model_snapshot = Path(
        snapshot_download(
            repo_id=args.model,
            revision=args.model_revision,
            local_files_only=True,
        )
    )
    weight_manifest_sha, weight_files = _weight_manifest(model_snapshot)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        revision=tokenizer_revision,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    token_lengths = np.asarray(
        [len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]) for text in texts],
        dtype=np.int32,
    )
    too_long = np.flatnonzero(token_lengths > args.max_length)
    if len(too_long) and not args.allow_truncation:
        maximum = int(token_lengths.max())
        raise SystemExit(
            f"{len(too_long)} examples exceed max_length={args.max_length}; "
            f"longest={maximum}. Increase --max-length or explicitly allow truncation."
        )

    model = AutoModel.from_pretrained(
        args.model,
        revision=args.model_revision,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(args.device).eval()
    resolved_commit = getattr(model.config, "_commit_hash", None)
    if resolved_commit is not None and str(resolved_commit) != args.model_revision:
        raise SystemExit(
            f"resolved model commit {resolved_commit} differs from requested {args.model_revision}"
        )
    hidden_size = int(model.config.hidden_size)
    layer_count = int(model.config.num_hidden_layers)
    if any(layer > layer_count for layer in layers):
        raise SystemExit(f"requested layer exceeds model depth {layer_count}: {layers}")

    count = len(rows)
    mean_features = {
        layer: np.empty((count, hidden_size), dtype=np.float16) for layer in layers
    }
    last_features = {
        layer: np.empty((count, hidden_size), dtype=np.float16) for layer in layers
    }

    start = time.monotonic()
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for offset in range(0, count, args.batch_size):
        batch_texts = texts[offset : offset + args.batch_size]
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        )
        encoded = {key: value.to(args.device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model(
                **encoded,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        hidden_states = output.hidden_states
        mask = encoded["attention_mask"].unsqueeze(-1).to(torch.bfloat16)
        denominator = mask.sum(dim=1).clamp(min=1)
        last_position = encoded["attention_mask"].sum(dim=1) - 1
        batch_index = torch.arange(len(batch_texts), device=args.device)
        for layer in layers:
            hidden = hidden_states[layer]
            pooled = (hidden * mask).sum(dim=1) / denominator
            final = hidden[batch_index, last_position]
            end = offset + len(batch_texts)
            mean_features[layer][offset:end] = pooled.to(torch.float16).cpu().numpy()
            last_features[layer][offset:end] = final.to(torch.float16).cpu().numpy()
        del output, hidden_states, encoded

    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.synchronize()
        peak_memory = int(torch.cuda.max_memory_allocated())
        gpu_name = torch.cuda.get_device_name(torch.device(args.device))
    else:
        peak_memory = 0
        gpu_name = "CPU"
    elapsed = time.monotonic() - start

    model_config_sha = _sha256_text(_canonical_json(model.config.to_dict()))
    tokenizer_sha = _tokenizer_hash(
        tokenizer,
        tokenizer_id=tokenizer_id,
        revision=tokenizer_revision,
    )
    extraction_metadata = {
        "schema": "scinumbench-v2-extraction-metadata/3",
        "dataset_manifest_sha256": dataset_manifest_sha,
        "test_inputs_sha256": str(test_inputs["sha256"]),
        "test_inputs_schema_version": str(test_inputs["schema_version"]),
        "model_id": args.model,
        "model_revision": args.model_revision,
        "model_weight_manifest_sha256": weight_manifest_sha,
        "model_weight_files": weight_files,
        "model_config_sha256": model_config_sha,
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_sha256": tokenizer_sha,
        "extractor_code_sha256": _sha256_file(Path(__file__).resolve()),
        "max_length": args.max_length,
        "allow_truncation": bool(args.allow_truncation),
        "truncated_examples": int(len(too_long)),
        "layers": list(layers),
        "pooling": ["mean", "last"],
        "layer_semantics": (
            "hidden_states[k]: output after transformer block k; embedding state is index 0"
        ),
        "model_dtype": "bfloat16",
        "feature_dtype": "float16",
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
    }
    extraction_metadata_json = _canonical_json(extraction_metadata)
    extraction_metadata_sha = _sha256_text(extraction_metadata_json)

    arrays: dict[str, object] = {}
    for layer in layers:
        arrays[f"mean_layer_{layer}"] = mean_features[layer]
        arrays[f"last_layer_{layer}"] = last_features[layer]
    arrays.update(
        {
            "example_id": np.asarray([str(row["example_id"]) for row in rows]),
            "source_id": np.asarray([str(row["source_id"]) for row in rows]),
            "arxiv_id": np.asarray(
                [str(row.get("paper_id") or row.get("arxiv_id") or "") for row in rows]
            ),
            # Test labels are deliberately sealed. The runner opens test.jsonl
            # only after consuming the dataset/study-scoped global ledger.
            "label": np.asarray(
                [-1 if str(row["split"]) == "test" else int(row["label"]) for row in rows],
                dtype=np.int8,
            ),
            "split": np.asarray([str(row["split"]) for row in rows]),
            "strategy": np.asarray([str(row.get("actual_strategy") or "") for row in rows]),
            "text_sha256": np.asarray([_sha256_text(text) for text in texts]),
            "token_length": token_lengths,
            "dataset_manifest_sha256": np.asarray([dataset_manifest_sha]),
            "extraction_metadata_json": np.asarray([extraction_metadata_json]),
            "extraction_metadata_sha256": np.asarray([extraction_metadata_sha]),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, args.output)
    output_sha = _sha256_file(args.output)

    report = {
        "schema_version": "scinumbench-v2-hidden-features/3",
        "model": args.model,
        "model_commit": args.model_revision,
        "tokenizer": tokenizer_id,
        "tokenizer_commit": tokenizer_revision,
        "model_weight_manifest_sha256": weight_manifest_sha,
        "model_config_sha256": model_config_sha,
        "tokenizer_sha256": tokenizer_sha,
        "extraction_metadata_sha256": extraction_metadata_sha,
        "extractor_code_sha256": extraction_metadata["extractor_code_sha256"],
        "layers": list(layers),
        "pooling": ["attention_mask_mean", "last_nonpadding_token"],
        "examples": count,
        "split_counts": {
            split: sum(str(row["split"]) == split for row in rows) for split in SPLITS
        },
        "max_length": args.max_length,
        "longest_untruncated_tokens": int(token_lengths.max(initial=0)),
        "truncated_examples": int(len(too_long)),
        "batch_size": args.batch_size,
        "dtype": "bfloat16_model/float16_features",
        "dataset_manifest_sha256": dataset_manifest_sha,
        "test_inputs_sha256": str(test_inputs["sha256"]),
        "output_sha256": output_sha,
        "elapsed_seconds": elapsed,
        "examples_per_second": count / elapsed if elapsed else None,
        "peak_device_memory_bytes": peak_memory,
        "device_name": gpu_name,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "selection_labels_used": False,
        "test_labels_embedded": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

