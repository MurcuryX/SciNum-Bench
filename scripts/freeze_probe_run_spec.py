#!/usr/bin/env python3
"""Freeze the external anchors required before a final probe test run."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path

from scinumbench_v2.experiment_protocol import (
    PRIMARY_PROBE_SEEDS,
    assert_expected_model_identity,
    assert_feature_archives_aligned,
    load_verified_feature_archive,
    prepare_test_ledger_binding,
)


def _load_runner(project_root: Path):
    path = project_root / "scripts" / "run_probe_experiment_v2.py"
    spec = importlib.util.spec_from_file_location("scinumbench_v2_probe_runner", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import probe runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument("--feature-report", action="append", default=[])
    parser.add_argument("--expected-model", action="append", default=[])
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--test-ledger-root", type=Path, required=True)
    parser.add_argument("--seeds", default=",".join(str(x) for x in PRIMARY_PROBE_SEEDS))
    parser.add_argument("--c-grid", default="0.001,0.01,0.1,1,10")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261026)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        timestamp = datetime.fromisoformat(args.created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SystemExit("--created-at must be ISO-8601") from exc
    if timestamp.tzinfo is None:
        raise SystemExit("--created-at must include a timezone")
    if args.output.exists():
        raise SystemExit("--output already exists; a frozen run spec is immutable")

    runner = _load_runner(args.project_root)
    feature_paths = runner._parse_feature_specs(args.feature)
    report_paths = runner._parse_feature_specs(args.feature_report)
    expected_models = runner._parse_expected_models(args.expected_model)
    if set(feature_paths) != set(report_paths) or set(feature_paths) != set(expected_models):
        raise SystemExit("feature, report, and expected-model aliases must match exactly")
    seeds = runner._parse_unique_ints(args.seeds)
    if seeds != PRIMARY_PROBE_SEEDS:
        raise SystemExit("run spec must use the preregistered primary seed list")
    c_values = runner._parse_unique_floats(args.c_grid)
    if args.max_iter <= 0 or args.bootstrap_replicates <= 0:
        raise SystemExit("--max-iter and --bootstrap-replicates must be positive")

    source_revision, source_hashes = runner._source_revision(args.project_root)
    archives = {
        alias: load_verified_feature_archive(
            path,
            split_dir=args.split_dir,
            manifest_path=args.manifest,
        )
        for alias, path in sorted(feature_paths.items())
    }
    assert_feature_archives_aligned(archives)
    reference = archives[sorted(archives)[0]]
    bindings: dict[str, dict[str, str]] = {}
    for alias, archive in sorted(archives.items()):
        model_id, revision = expected_models[alias]
        assert_expected_model_identity(
            alias,
            archive,
            expected_model_id=model_id,
            expected_revision=revision,
        )
        if archive.extraction_metadata["extractor_code_sha256"] != source_hashes[
            "scripts/extract_hidden_features.py"
        ]:
            raise SystemExit(f"feature archive {alias!r} used another extractor revision")
        report_sha256 = runner._sha256_file(report_paths[alias])
        runner._validate_feature_report(
            alias,
            report_paths[alias],
            archive,
            expected_report_sha256=report_sha256,
            expected_extractor_sha256=source_hashes[
                "scripts/extract_hidden_features.py"
            ],
        )
        bindings[alias] = {
            "archive_sha256": archive.archive_sha256,
            "report_sha256": report_sha256,
            "model_id": model_id,
            "model_revision": revision,
        }

    ledger_binding = prepare_test_ledger_binding(
        args.test_ledger_root,
        dataset_manifest_sha256=reference.dataset_manifest_sha256,
    )
    payload = {
        "schema": "scinumbench-v2-probe-run-spec/2",
        "created_at": args.created_at,
        "study_id": args.study_id,
        "dataset_manifest_sha256": reference.dataset_manifest_sha256,
        "canonical_test_ledger_root": ledger_binding["canonical_root"],
        "test_ledger_binding": ledger_binding,
        "source_revision": source_revision,
        "features": bindings,
        "seeds": list(seeds),
        "c_grid_hex": [value.hex() for value in c_values],
        "max_iter": args.max_iter,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
    }
    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"
    _atomic_text(args.output, rendered)
    print(
        json.dumps(
            {
                "run_spec": str(args.output),
                "sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
                "source_revision": source_revision,
                "dataset_manifest_sha256": reference.dataset_manifest_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
