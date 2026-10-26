#!/usr/bin/env python3
"""Run the frozen SciNumBench V2 probe study under a sealed protocol.

Train is the only fitting partition. Layer/pooling/C, probability fusion, and
all thresholds are selected on validation. Test labels remain absent from the
feature archives and are opened only after a dataset/study-scoped global ledger
record has been atomically consumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
from typing import Iterable, Mapping

from scinumbench_v2.experiment_protocol import (
    PRIMARY_PROBE_SEEDS,
    SeededCandidateScoreSet,
    acquire_test_access_guard,
    assert_expected_model_identity,
    assert_feature_archives_aligned,
    average_precision,
    binary_f1,
    build_seeded_probability_fusions,
    evaluate_f1_with_frozen_threshold,
    evaluate_seed_mean_choice_on_test,
    fit_probe_candidates,
    function_implementation_hash,
    grouped_bootstrap_seed_mean_auroc_difference,
    grouped_bootstrap_seed_mean_metrics,
    load_verified_feature_archive,
    predict_probe_on_test,
    seed_mean_auroc_implementation_hash,
    seed_mean_choice_payload,
    select_f1_threshold_on_validation,
    select_seed_mean_auroc_on_validation,
    summarize_seed_metrics,
    threshold_choice_payload,
    unlock_test_partition,
    validate_test_ledger_binding,
)
from scinumbench_v2.split_protocol import (
    CandidateScoreSet,
    Partition,
    binary_auroc,
    candidate_config_hash,
    compute_dataset_manifest_hash,
)


DEFAULT_SEEDS = ",".join(str(seed) for seed in PRIMARY_PROBE_SEEDS)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _write_json(path: Path, value: object) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    _atomic_text(
        path,
        "".join(_canonical_json(row) + "\n" for row in rows),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_revision(project_root: Path) -> tuple[str, dict[str, str]]:
    paths = (
        project_root / "scripts" / "extract_hidden_features.py",
        project_root / "scripts" / "freeze_probe_run_spec.py",
        project_root / "scripts" / "run_probe_experiment_v2.py",
        project_root / "src" / "scinumbench_v2" / "experiment_protocol.py",
        project_root / "src" / "scinumbench_v2" / "split_protocol.py",
        project_root / "src" / "scinumbench_v2" / "freeze_manifest.py",
    )
    hashes = {str(path.relative_to(project_root)): _sha256_file(path) for path in paths}
    revision = hashlib.sha256(_canonical_json(hashes).encode("utf-8")).hexdigest()
    return revision, hashes


def _parse_feature_specs(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit("--feature must use ALIAS=PATH")
        name, raw_path = value.split("=", 1)
        if re.fullmatch(r"[A-Za-z0-9_.-]+", name) is None:
            raise SystemExit(f"invalid feature alias: {name!r}")
        if name in result:
            raise SystemExit(f"duplicate feature alias: {name}")
        result[name] = Path(raw_path)
    if not result:
        raise SystemExit("at least one --feature ALIAS=PATH is required")
    return result


def _parse_expected_models(values: list[str]) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    for value in values:
        if "=" not in value or "@" not in value:
            raise SystemExit("--expected-model must use ALIAS=REPO_ID@40_HEX_COMMIT")
        alias, identity = value.split("=", 1)
        model_id, revision = identity.rsplit("@", 1)
        if alias in result:
            raise SystemExit(f"duplicate expected model alias: {alias}")
        if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model_id) is None:
            raise SystemExit(f"invalid expected model repository ID: {model_id!r}")
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise SystemExit("expected model revision must be exact lowercase 40-hex")
        result[alias] = (model_id, revision)
    return result


def _load_canonical_run_spec(path: Path, expected_sha256: str) -> dict[str, object]:
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise SystemExit("--run-spec-sha256 must be a lowercase SHA-256 digest")
    if not path.is_file():
        raise SystemExit(f"run spec not found: {path}")
    payload = path.read_bytes()
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != expected_sha256:
        raise SystemExit(
            "run spec hash mismatch: "
            f"expected {expected_sha256}, observed {observed_sha256}"
        )
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit("run spec must be valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise SystemExit("run spec must be a JSON object")
    if payload != (_canonical_json(value) + "\n").encode("utf-8"):
        raise SystemExit("run spec must use canonical JSON plus one trailing newline")
    if value.get("schema") != "scinumbench-v2-probe-run-spec/2":
        raise SystemExit("unsupported run spec schema")
    return value


def _validate_feature_report(
    alias: str,
    path: Path,
    archive: object,
    *,
    expected_report_sha256: str,
    expected_extractor_sha256: str,
) -> dict[str, object]:
    if re.fullmatch(r"[0-9a-f]{64}", expected_report_sha256) is None:
        raise SystemExit(f"run spec has invalid report hash for {alias!r}")
    if not path.is_file():
        raise SystemExit(f"feature extraction report not found for {alias!r}: {path}")
    observed_report_sha256 = _sha256_file(path)
    if observed_report_sha256 != expected_report_sha256:
        raise SystemExit(f"feature extraction report hash mismatch for {alias!r}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"feature extraction report for {alias!r} is invalid JSON") from exc
    if not isinstance(report, dict):
        raise SystemExit(f"feature extraction report for {alias!r} must be an object")
    if report.get("schema_version") != "scinumbench-v2-hidden-features/3":
        raise SystemExit(f"feature extraction report for {alias!r} has wrong schema")

    metadata = getattr(archive, "extraction_metadata")
    expected = {
        "output_sha256": getattr(archive, "archive_sha256"),
        "dataset_manifest_sha256": getattr(archive, "dataset_manifest_sha256"),
        "test_inputs_sha256": metadata["test_inputs_sha256"],
        "model": getattr(archive, "model_id"),
        "model_commit": getattr(archive, "model_revision"),
        "tokenizer": metadata["tokenizer_id"],
        "tokenizer_commit": metadata["tokenizer_revision"],
        "model_weight_manifest_sha256": metadata[
            "model_weight_manifest_sha256"
        ],
        "model_config_sha256": metadata["model_config_sha256"],
        "tokenizer_sha256": metadata["tokenizer_sha256"],
        "extraction_metadata_sha256": getattr(
            archive, "extraction_metadata_sha256"
        ),
        "extractor_code_sha256": expected_extractor_sha256,
        "layers": metadata["layers"],
        "max_length": metadata["max_length"],
        "truncated_examples": metadata["truncated_examples"],
        "examples": len(getattr(archive, "example_ids")),
    }
    mismatches = [
        field_name
        for field_name, expected_value in expected.items()
        if report.get(field_name) != expected_value
    ]
    if mismatches:
        raise SystemExit(
            f"feature extraction report mismatch for {alias!r}: "
            + ", ".join(mismatches)
        )
    if report.get("pooling") != ["attention_mask_mean", "last_nonpadding_token"]:
        raise SystemExit(f"feature extraction report pooling mismatch for {alias!r}")
    splits = getattr(archive, "splits")
    expected_split_counts = {
        split: sum(value == split for value in splits)
        for split in ("train", "validation", "test")
    }
    if report.get("split_counts") != expected_split_counts:
        raise SystemExit(f"feature extraction report split counts mismatch for {alias!r}")
    if report.get("selection_labels_used") is not False:
        raise SystemExit(f"feature extraction report used selection labels for {alias!r}")
    if report.get("test_labels_embedded") is not False:
        raise SystemExit(f"feature extraction report embedded test labels for {alias!r}")
    return report


def _parse_unique_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise SystemExit("--seeds must be comma-separated integers") from exc
    if not parsed or len(set(parsed)) != len(parsed):
        raise SystemExit("--seeds must be non-empty and unique")
    return parsed


def _parse_unique_floats(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise SystemExit("--c-grid must be comma-separated numbers") from exc
    if (
        not parsed
        or len(set(parsed)) != len(parsed)
        or any(not math.isfinite(item) or item <= 0 for item in parsed)
    ):
        raise SystemExit("--c-grid values must be finite, unique, and positive")
    return parsed


def _validate_probe_run_spec(
    spec: Mapping[str, object],
    *,
    study_id: str,
    dataset_manifest_sha256: str,
    ledger_root: Path,
    source_revision: str,
    archives: Mapping[str, object],
    feature_report_paths: Mapping[str, Path],
    expected_models: Mapping[str, tuple[str, str]],
    seeds: tuple[int, ...],
    c_values: tuple[float, ...],
    max_iter: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[dict[str, str], dict[str, object]]:
    expected_fields = {
        "schema",
        "created_at",
        "study_id",
        "dataset_manifest_sha256",
        "canonical_test_ledger_root",
        "test_ledger_binding",
        "source_revision",
        "features",
        "seeds",
        "c_grid_hex",
        "max_iter",
        "bootstrap_replicates",
        "bootstrap_seed",
    }
    if set(spec) != expected_fields:
        raise SystemExit(
            "run spec fields differ from the frozen schema: "
            f"missing={sorted(expected_fields - set(spec))}, "
            f"extra={sorted(set(spec) - expected_fields)}"
        )
    raw_ledger_binding = spec.get("test_ledger_binding")
    if not isinstance(raw_ledger_binding, Mapping):
        raise SystemExit("run spec test_ledger_binding must be an object")
    try:
        ledger_binding = validate_test_ledger_binding(
            ledger_root,
            raw_ledger_binding,
            dataset_manifest_sha256=dataset_manifest_sha256,
        )
    except Exception as exc:
        raise SystemExit(f"run spec test ledger binding failed: {exc}") from exc
    scalar_expected = {
        "study_id": study_id,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "canonical_test_ledger_root": ledger_binding["canonical_root"],
        "source_revision": source_revision,
        "seeds": list(seeds),
        "c_grid_hex": [value.hex() for value in c_values],
        "max_iter": max_iter,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
    }
    mismatches = [
        field_name
        for field_name, expected_value in scalar_expected.items()
        if spec.get(field_name) != expected_value
    ]
    if mismatches:
        raise SystemExit("run spec mismatch: " + ", ".join(mismatches))
    created_at = spec.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        raise SystemExit("run spec created_at must be a non-empty string")
    feature_specs = spec.get("features")
    if not isinstance(feature_specs, Mapping) or set(feature_specs) != set(archives):
        raise SystemExit("run spec feature aliases differ from CLI feature aliases")
    report_hashes: dict[str, str] = {}
    for alias, archive in sorted(archives.items()):
        binding = feature_specs.get(alias)
        if not isinstance(binding, Mapping) or set(binding) != {
            "archive_sha256",
            "report_sha256",
            "model_id",
            "model_revision",
        }:
            raise SystemExit(f"run spec feature binding is invalid for {alias!r}")
        expected_model_id, expected_model_revision = expected_models[alias]
        report_sha256 = str(binding.get("report_sha256", ""))
        expected_binding = {
            "archive_sha256": getattr(archive, "archive_sha256"),
            "report_sha256": _sha256_file(feature_report_paths[alias]),
            "model_id": expected_model_id,
            "model_revision": expected_model_revision,
        }
        if any(binding.get(key) != value for key, value in expected_binding.items()):
            raise SystemExit(f"run spec feature binding mismatch for {alias!r}")
        if re.fullmatch(r"[0-9a-f]{64}", report_sha256) is None:
            raise SystemExit(f"run spec report hash is invalid for {alias!r}")
        report_hashes[alias] = report_sha256
    return report_hashes, ledger_binding


def _mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def _bootstrap_payload(result: object) -> dict[str, object]:
    return {
        "grouping": result.grouping,
        "group_count": result.group_count,
        "replicates": result.replicates,
        "confidence": result.confidence,
        "seed": result.seed,
        "point": dict(result.point),
        "lower": dict(result.lower),
        "upper": dict(result.upper),
    }


def _difference_payload(result: object) -> dict[str, object]:
    return {
        "grouping": result.grouping,
        "metric": result.metric,
        "group_count": result.group_count,
        "replicates": result.replicates,
        "confidence": result.confidence,
        "seed": result.seed,
        "point_difference": result.point_difference,
        "lower": result.lower,
        "upper": result.upper,
    }


def _seed_report_payload(result: object) -> dict[str, object]:
    return {
        "estimand": "mean and sample SD of independently seeded probe metrics",
        "per_seed": {str(seed): dict(values) for seed, values in result.per_seed.items()},
        "mean": dict(result.seed_mean),
        "sample_std": dict(result.seed_sample_std),
        "probability_ensemble_secondary": {
            "selection": "not independently optimized; uses primary-selected config",
            **dict(result.probability_ensemble),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument("--feature-report", action="append", default=[])
    parser.add_argument("--expected-model", action="append", default=[])
    parser.add_argument("--run-spec", type=Path, required=True)
    parser.add_argument("--run-spec-sha256", required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--test-ledger-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--c-grid", default="0.001,0.01,0.1,1,10")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261026)
    args = parser.parse_args()

    feature_specs = _parse_feature_specs(args.feature)
    feature_report_specs = _parse_feature_specs(args.feature_report)
    expected_models = _parse_expected_models(args.expected_model)
    if set(expected_models) != set(feature_specs):
        raise SystemExit("--expected-model aliases must exactly match --feature aliases")
    if set(feature_report_specs) != set(feature_specs):
        raise SystemExit("--feature-report aliases must exactly match --feature aliases")
    run_spec = _load_canonical_run_spec(args.run_spec, args.run_spec_sha256)
    seeds = _parse_unique_ints(args.seeds)
    if seeds != PRIMARY_PROBE_SEEDS:
        raise SystemExit(
            "--seeds must exactly match the preregistered ordered primary seed list: "
            + DEFAULT_SEEDS
        )
    c_values = _parse_unique_floats(args.c_grid)
    if args.max_iter <= 0 or args.bootstrap_replicates <= 0:
        raise SystemExit("--max-iter and --bootstrap-replicates must be positive")
    output_resolved = args.output_dir.resolve()
    ledger_resolved = args.test_ledger_root.resolve()
    if (
        ledger_resolved == output_resolved
        or output_resolved in ledger_resolved.parents
        or ledger_resolved in output_resolved.parents
    ):
        raise SystemExit("--test-ledger-root must be independent of --output-dir")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit("--output-dir must be absent or empty for a final run")

    project_root = Path(__file__).resolve().parents[1]
    source_revision, source_hashes = _source_revision(project_root)
    archives = {
        name: load_verified_feature_archive(
            path,
            split_dir=args.split_dir,
            manifest_path=args.manifest,
        )
        for name, path in sorted(feature_specs.items())
    }
    for alias, archive in archives.items():
        expected_id, expected_revision = expected_models[alias]
        assert_expected_model_identity(
            alias,
            archive,
            expected_model_id=expected_id,
            expected_revision=expected_revision,
        )
        extractor_hash = source_hashes["scripts/extract_hidden_features.py"]
        if archive.extraction_metadata["extractor_code_sha256"] != extractor_hash:
            raise SystemExit(
                f"feature archive {alias!r} was produced by a different extractor revision"
            )
    assert_feature_archives_aligned(archives)
    reference = archives[sorted(archives)[0]]
    feature_report_hashes, ledger_binding = _validate_probe_run_spec(
        run_spec,
        study_id=args.study_id,
        dataset_manifest_sha256=reference.dataset_manifest_sha256,
        ledger_root=args.test_ledger_root,
        source_revision=source_revision,
        archives=archives,
        feature_report_paths=feature_report_specs,
        expected_models=expected_models,
        seeds=seeds,
        c_values=c_values,
        max_iter=args.max_iter,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    feature_reports = {
        alias: _validate_feature_report(
            alias,
            feature_report_specs[alias],
            archives[alias],
            expected_report_sha256=feature_report_hashes[alias],
            expected_extractor_sha256=source_hashes[
                "scripts/extract_hidden_features.py"
            ],
        )
        for alias in sorted(archives)
    }
    dataset_id = "scinumbench-v2-hard:" + reference.dataset_manifest_sha256
    validation_indices = reference.indices(Partition.VALIDATION)
    test_indices = reference.indices(Partition.TEST)
    validation_ids = tuple(reference.example_ids[index] for index in validation_indices)
    validation_labels = reference.selection_labels(Partition.VALIDATION)
    test_ids = tuple(reference.example_ids[index] for index in test_indices)
    expected_test_manifest = compute_dataset_manifest_hash(
        Partition.TEST,
        dataset_id,
        test_ids,
    )

    searches = {
        name: fit_probe_candidates(
            archive,
            archive_name=name,
            dataset_id=dataset_id,
            c_values=c_values,
            seeds=seeds,
            max_iter=args.max_iter,
        )
        for name, archive in archives.items()
    }
    base_choices = {
        name: select_seed_mean_auroc_on_validation(
            search.seeded_candidate_score_set(seeds),
            expected_test_manifest_hash=expected_test_manifest,
        )
        for name, search in searches.items()
    }

    base_per_seed_validation_scores = {
        name: searches[name].candidates[choice.candidate_name].per_seed_validation_scores
        for name, choice in base_choices.items()
    }
    base_config_hashes = {
        name: searches[name].candidates[choice.candidate_name].config_hash
        for name, choice in base_choices.items()
    }
    validation_fusions = build_seeded_probability_fusions(
        base_per_seed_validation_scores,
        base_config_hashes,
        seeds=seeds,
    )
    fusion_validation_set = SeededCandidateScoreSet(
        partition=Partition.VALIDATION,
        dataset_id=dataset_id,
        sample_ids=validation_ids,
        labels=validation_labels,
        seeds=seeds,
        candidates=validation_fusions.per_seed_scores,
        candidate_config_hashes=validation_fusions.config_hashes,
    )
    fusion_choice = select_seed_mean_auroc_on_validation(
        fusion_validation_set,
        expected_test_manifest_hash=expected_test_manifest,
    )
    final_members = validation_fusions.members[fusion_choice.candidate_name]
    per_seed_final_validation = dict(
        validation_fusions.per_seed_scores[fusion_choice.candidate_name]
    )
    final_validation_scores = validation_fusions.probability_ensemble_scores[
        fusion_choice.candidate_name
    ]
    ensemble_validation_set = CandidateScoreSet(
        partition=Partition.VALIDATION,
        dataset_id=dataset_id,
        sample_ids=validation_ids,
        labels=validation_labels,
        candidates={fusion_choice.candidate_name: final_validation_scores},
        candidate_config_hashes={
            fusion_choice.candidate_name: validation_fusions.config_hashes[
                fusion_choice.candidate_name
            ]
        },
    )
    threshold_choice = select_f1_threshold_on_validation(
        ensemble_validation_set,
        candidate_name=fusion_choice.candidate_name,
        expected_test_manifest_hash=expected_test_manifest,
    )
    per_seed_threshold_choices = {}
    for seed in seeds:
        candidate_name = f"{fusion_choice.candidate_name}:seed={seed}"
        config_hash = candidate_config_hash(
            {
                "schema": "scinumbench-v2-seed-specific-fusion/1",
                "fusion_config_hash": validation_fusions.config_hashes[
                    fusion_choice.candidate_name
                ],
                "seed": seed,
            }
        )
        score_set = CandidateScoreSet(
            partition=Partition.VALIDATION,
            dataset_id=dataset_id,
            sample_ids=validation_ids,
            labels=validation_labels,
            candidates={candidate_name: per_seed_final_validation[seed]},
            candidate_config_hashes={candidate_name: config_hash},
        )
        per_seed_threshold_choices[seed] = select_f1_threshold_on_validation(
            score_set,
            candidate_name=candidate_name,
            expected_test_manifest_hash=expected_test_manifest,
        )

    probe_leaderboard: list[dict[str, object]] = []
    for alias, search in sorted(searches.items()):
        for candidate_name, candidate in sorted(search.candidates.items()):
            per_seed_auroc = {
                seed: binary_auroc(validation_labels, candidate.per_seed_validation_scores[seed])
                for seed in seeds
            }
            mean, std = _mean_std(list(per_seed_auroc.values()))
            probe_leaderboard.append(
                {
                    "archive_alias": alias,
                    "model_id": archives[alias].model_id,
                    "model_revision": archives[alias].model_revision,
                    "candidate_name": candidate_name,
                    "config_hash": candidate.config_hash,
                    "config": dict(candidate.config),
                    "n_iter_by_seed": {
                        str(seed): candidate.n_iter_by_seed[seed] for seed in seeds
                    },
                    "validation_seed_auroc": {
                        str(seed): per_seed_auroc[seed] for seed in seeds
                    },
                    "validation_seed_auroc_mean": mean,
                    "validation_seed_auroc_sample_std": std,
                    "validation_probability_ensemble_auroc": binary_auroc(
                        validation_labels, candidate.validation_scores
                    ),
                    "selected": candidate_name == base_choices[alias].candidate_name,
                }
            )

    fusion_leaderboard: list[dict[str, object]] = []
    for fusion_name, members in sorted(validation_fusions.members.items()):
        per_seed_auroc = {
            seed: binary_auroc(
                validation_labels,
                validation_fusions.per_seed_scores[fusion_name][seed],
            )
            for seed in seeds
        }
        mean, std = _mean_std(list(per_seed_auroc.values()))
        fusion_leaderboard.append(
            {
                "candidate_name": fusion_name,
                "members": list(members),
                "config_hash": validation_fusions.config_hashes[fusion_name],
                "validation_seed_auroc": {
                    str(seed): per_seed_auroc[seed] for seed in seeds
                },
                "validation_seed_auroc_mean": mean,
                "validation_seed_auroc_sample_std": std,
                "validation_probability_ensemble_auroc": binary_auroc(
                    validation_labels,
                    validation_fusions.probability_ensemble_scores[fusion_name],
                ),
                "validation_probability_ensemble_ap": average_precision(
                    validation_labels,
                    validation_fusions.probability_ensemble_scores[fusion_name],
                ),
                "selected": fusion_name == fusion_choice.candidate_name,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_path = args.output_dir / "validation_leaderboard.json"
    _write_json(
        leaderboard_path,
        {
            "schema": "scinumbench-v2-validation-leaderboard/3",
            "selection_estimand": "arithmetic mean of 10 per-seed validation AUROCs",
            "seeds": list(seeds),
            "probability_ensemble_role": "secondary, using the primary-selected config",
            "probe_candidates": probe_leaderboard,
            "fusion_candidates": fusion_leaderboard,
        },
    )
    receipts = {
        "schema": "scinumbench-v2-complete-selection-receipts/3",
        "probe_choices": {
            name: seed_mean_choice_payload(choice)
            for name, choice in sorted(base_choices.items())
        },
        "fusion_choice": seed_mean_choice_payload(fusion_choice),
        "secondary_probability_ensemble_threshold": threshold_choice_payload(
            threshold_choice
        ),
        "per_seed_thresholds": {
            str(seed): threshold_choice_payload(per_seed_threshold_choices[seed])
            for seed in seeds
        },
    }
    receipts_path = args.output_dir / "validation_selection_receipts.json"
    _write_json(receipts_path, receipts)

    validation_prediction_rows: list[dict[str, object]] = []
    for index, example_id in enumerate(validation_ids):
        validation_prediction_rows.append(
            {
                "example_id": example_id,
                "source_id": reference.source_ids[validation_indices[index]],
                "paper_id": reference.paper_ids[validation_indices[index]],
                "label": validation_labels[index],
                "selected_fusion": fusion_choice.candidate_name,
                "probability_ensemble_score": final_validation_scores[index],
                "per_seed_scores": {
                    str(seed): per_seed_final_validation[seed][index] for seed in seeds
                },
                "selected_base_scores": {
                    name: {
                        "probability_ensemble": searches[name]
                        .candidates[base_choices[name].candidate_name]
                        .validation_scores[index],
                        "per_seed": {
                            str(seed): searches[name]
                            .candidates[base_choices[name].candidate_name]
                            .per_seed_validation_scores[seed][index]
                            for seed in seeds
                        },
                    }
                    for name in sorted(searches)
                },
            }
        )
    validation_predictions_path = args.output_dir / "validation_predictions.jsonl"
    _write_jsonl(validation_predictions_path, validation_prediction_rows)

    selection_receipts_sha = _sha256_file(receipts_path)
    test_protocol_digest = hashlib.sha256(
        _canonical_json(
            {
                "schema": "scinumbench-v2-complete-test-precommitment/3",
                "study_id": args.study_id,
                "run_spec_sha256": args.run_spec_sha256,
                "feature_reports_sha256": feature_report_hashes,
                "probe_receipts": {
                    name: choice.protocol_digest
                    for name, choice in sorted(base_choices.items())
                },
                "fusion_receipt": fusion_choice.protocol_digest,
                "secondary_ensemble_threshold_receipt": threshold_choice.protocol_digest,
                "seed_threshold_receipts": {
                    str(seed): per_seed_threshold_choices[seed].protocol_digest
                    for seed in seeds
                },
                "validation_leaderboard_sha256": _sha256_file(leaderboard_path),
            }
        ).encode("utf-8")
    ).hexdigest()
    grant = acquire_test_access_guard(
        args.test_ledger_root,
        ledger_binding=ledger_binding,
        study_id=args.study_id,
        dataset_manifest_sha256=reference.dataset_manifest_sha256,
        protocol_digest=test_protocol_digest,
        selection_receipts_sha256=selection_receipts_sha,
        feature_archive_hashes={
            name: archive.archive_sha256 for name, archive in archives.items()
        },
        source_revision=source_revision,
    )
    test_partitions = {
        name: unlock_test_partition(
            archive,
            grant,
            split_dir=args.split_dir,
            manifest_path=args.manifest,
        )
        for name, archive in archives.items()
    }
    reference_test = test_partitions[sorted(test_partitions)[0]]
    for name, partition in test_partitions.items():
        if (
            partition.example_ids != reference_test.example_ids
            or partition.labels != reference_test.labels
            or partition.source_ids != reference_test.source_ids
            or partition.paper_ids != reference_test.paper_ids
        ):
            raise RuntimeError(f"unlocked test partition drift for archive {name}")

    base_test_predictions = {
        name: predict_probe_on_test(
            searches[name],
            archives[name],
            test_partitions[name],
            base_choices[name].candidate_name,
        )
        for name in sorted(searches)
    }
    test_fusions = build_seeded_probability_fusions(
        {name: prediction.per_seed_scores for name, prediction in base_test_predictions.items()},
        base_config_hashes,
        seeds=seeds,
    )
    if test_fusions.config_hashes != validation_fusions.config_hashes:
        raise RuntimeError("validation/test fusion configuration drift")

    test_labels = reference_test.labels
    fusion_test_set = SeededCandidateScoreSet(
        partition=Partition.TEST,
        dataset_id=dataset_id,
        sample_ids=reference_test.example_ids,
        labels=test_labels,
        seeds=seeds,
        candidates=test_fusions.per_seed_scores,
        candidate_config_hashes=test_fusions.config_hashes,
    )
    primary_evaluation = evaluate_seed_mean_choice_on_test(fusion_choice, fusion_test_set)
    per_seed_final_test = dict(
        test_fusions.per_seed_scores[fusion_choice.candidate_name]
    )
    final_test_scores = test_fusions.probability_ensemble_scores[
        fusion_choice.candidate_name
    ]
    ensemble_test_set = CandidateScoreSet(
        partition=Partition.TEST,
        dataset_id=dataset_id,
        sample_ids=reference_test.example_ids,
        labels=test_labels,
        candidates={fusion_choice.candidate_name: final_test_scores},
        candidate_config_hashes={
            fusion_choice.candidate_name: test_fusions.config_hashes[
                fusion_choice.candidate_name
            ]
        },
    )
    evaluate_f1_with_frozen_threshold(threshold_choice, ensemble_test_set)
    for seed in seeds:
        choice = per_seed_threshold_choices[seed]
        seed_test_set = CandidateScoreSet(
            partition=Partition.TEST,
            dataset_id=dataset_id,
            sample_ids=reference_test.example_ids,
            labels=test_labels,
            candidates={choice.candidate_name: per_seed_final_test[seed]},
            candidate_config_hashes={
                choice.candidate_name: choice.candidate_config_hash
            },
        )
        evaluate_f1_with_frozen_threshold(choice, seed_test_set)
    seed_metric_report = summarize_seed_metrics(
        test_labels,
        per_seed_final_test,
        {
            seed: per_seed_threshold_choices[seed].threshold
            for seed in seeds
        },
        probability_ensemble_scores=final_test_scores,
        probability_ensemble_threshold=threshold_choice.threshold,
    )

    test_prediction_rows: list[dict[str, object]] = []
    for index, example_id in enumerate(reference_test.example_ids):
        test_prediction_rows.append(
            {
                "example_id": example_id,
                "source_id": reference_test.source_ids[index],
                "paper_id": reference_test.paper_ids[index],
                "label": test_labels[index],
                "selected_fusion": fusion_choice.candidate_name,
                "probability_ensemble_score": final_test_scores[index],
                "probability_ensemble_prediction": int(
                    final_test_scores[index] >= threshold_choice.threshold
                ),
                "per_seed_scores": {
                    str(seed): per_seed_final_test[seed][index] for seed in seeds
                },
                "per_seed_predictions": {
                    str(seed): int(
                        per_seed_final_test[seed][index]
                        >= per_seed_threshold_choices[seed].threshold
                    )
                    for seed in seeds
                },
                "selected_base_scores": {
                    name: {
                        "probability_ensemble": base_test_predictions[name].mean_scores[index],
                        "per_seed": {
                            str(seed): base_test_predictions[name].per_seed_scores[seed][index]
                            for seed in seeds
                        },
                    }
                    for name in sorted(base_test_predictions)
                },
            }
        )
    test_predictions_path = args.output_dir / "test_predictions.jsonl"
    _write_jsonl(test_predictions_path, test_prediction_rows)

    frozen_seed_thresholds = {
        seed: per_seed_threshold_choices[seed].threshold for seed in seeds
    }
    source_bootstrap = grouped_bootstrap_seed_mean_metrics(
        test_labels,
        per_seed_final_test,
        frozen_seed_thresholds,
        reference_test.source_ids,
        seeds=seeds,
        grouping="source_id",
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    paper_bootstrap = grouped_bootstrap_seed_mean_metrics(
        test_labels,
        per_seed_final_test,
        frozen_seed_thresholds,
        reference_test.paper_ids,
        seeds=seeds,
        grouping="paper_id",
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    comparison_intervals: dict[str, object] = {}
    for name, prediction in sorted(base_test_predictions.items()):
        comparison_intervals[name] = {
            "interpretation": "mean_seed_auroc_fusion_minus_mean_seed_auroc_base",
            "source_grouped": _difference_payload(
                grouped_bootstrap_seed_mean_auroc_difference(
                    test_labels,
                    per_seed_final_test,
                    prediction.per_seed_scores,
                    reference_test.source_ids,
                    seeds=seeds,
                    grouping="source_id",
                    replicates=args.bootstrap_replicates,
                    seed=args.bootstrap_seed,
                )
            ),
            "paper_grouped": _difference_payload(
                grouped_bootstrap_seed_mean_auroc_difference(
                    test_labels,
                    per_seed_final_test,
                    prediction.per_seed_scores,
                    reference_test.paper_ids,
                    seeds=seeds,
                    grouping="paper_id",
                    replicates=args.bootstrap_replicates,
                    seed=args.bootstrap_seed,
                )
            ),
        }

    import numpy as np
    import joblib
    import scipy
    import sklearn
    import threadpoolctl

    parameter_arrays: dict[str, object] = {}
    for archive_name, choice in sorted(base_choices.items()):
        candidate = searches[archive_name].candidates[choice.candidate_name]
        safe_name = re.sub(r"[^A-Za-z0-9_]", "_", archive_name)
        for seed, (scaler, classifier) in candidate.fitted_estimators.items():
            prefix = f"{safe_name}_seed_{seed}"
            parameter_arrays[f"{prefix}_scaler_mean"] = scaler.mean_
            parameter_arrays[f"{prefix}_scaler_scale"] = scaler.scale_
            parameter_arrays[f"{prefix}_coef"] = classifier.coef_
            parameter_arrays[f"{prefix}_intercept"] = classifier.intercept_
            parameter_arrays[f"{prefix}_n_iter"] = classifier.n_iter_
    parameter_path = args.output_dir / "selected_probe_parameters.npz"
    temporary_parameter_path = parameter_path.with_name(parameter_path.name + ".tmp.npz")
    np.savez(temporary_parameter_path, **parameter_arrays)
    os.replace(temporary_parameter_path, parameter_path)

    dependency_versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "threadpoolctl": threadpoolctl.__version__,
    }
    artifact_paths = {
        "selection_receipts": receipts_path,
        "validation_leaderboard": leaderboard_path,
        "validation_predictions": validation_predictions_path,
        "test_predictions": test_predictions_path,
        "selected_parameters": parameter_path,
    }
    artifact_hashes = {name: _sha256_file(path) for name, path in artifact_paths.items()}
    report = {
        "schema": "scinumbench-v2-probe-experiment/3",
        "status": "TEST_EVALUATED_ONCE_GLOBAL_LEDGER",
        "study_id": args.study_id,
        "dataset_id": dataset_id,
        "dataset_manifest_sha256": reference.dataset_manifest_sha256,
        "run_spec_sha256": args.run_spec_sha256,
        "test_ledger": {
            "logical_scope": "dataset_manifest_sha256 + study_id",
            "record_name": grant.ledger_path.name,
            "record_sha256": grant.ledger_payload_sha256,
        },
        "feature_archives": {
            name: {
                "file_name": archive.path.name,
                "sha256": archive.archive_sha256,
                "model_id": archive.model_id,
                "model_revision": archive.model_revision,
                "extraction_metadata_sha256": archive.extraction_metadata_sha256,
                "extraction_metadata": dict(archive.extraction_metadata),
                "extraction_report_file_name": feature_report_specs[name].name,
                "extraction_report_sha256": feature_report_hashes[name],
                "feature_keys": list(archive.feature_keys),
            }
            for name, archive in sorted(archives.items())
        },
        "seeds": list(seeds),
        "seed_count": len(seeds),
        "primary_estimand": "mean of 10 independently seeded probe metrics",
        "probability_ensemble_role": "secondary diagnostic using primary-selected config",
        "c_grid": list(c_values),
        "selection_partition": "validation",
        "selection_estimand": "arithmetic mean of 10 per-seed validation AUROCs",
        "test_selection_used": False,
        "test_precommitment_digest": test_protocol_digest,
        "selected_probes": {
            name: {
                "candidate_name": choice.candidate_name,
                "validation_seed_auroc": {
                    str(seed): value for seed, value in choice.validation_seed_auroc
                },
                "validation_seed_auroc_mean": choice.validation_score,
                "config": dict(searches[name].candidates[choice.candidate_name].config),
                "config_hash": searches[name].candidates[choice.candidate_name].config_hash,
                "n_iter_by_seed": {
                    str(seed): searches[name]
                    .candidates[choice.candidate_name]
                    .n_iter_by_seed[seed]
                    for seed in seeds
                },
                "receipt_protocol_digest": choice.protocol_digest,
            }
            for name, choice in sorted(base_choices.items())
        },
        "selected_fusion": {
            "candidate_name": fusion_choice.candidate_name,
            "members": list(final_members),
            "validation_seed_auroc": {
                str(seed): value for seed, value in fusion_choice.validation_seed_auroc
            },
            "validation_seed_auroc_mean": fusion_choice.validation_score,
            "config_hash": validation_fusions.config_hashes[fusion_choice.candidate_name],
            "receipt_protocol_digest": fusion_choice.protocol_digest,
        },
        "test": {
            "rows": len(reference_test.example_ids),
            "sources": len(set(reference_test.source_ids)),
            "papers": len(set(reference_test.paper_ids)),
            "seed_metrics": _seed_report_payload(seed_metric_report),
            "primary_seed_mean_auroc": primary_evaluation.mean_auroc,
            "primary_seed_auroc_sample_std": primary_evaluation.sample_std_auroc,
            "test_manifest_hash": primary_evaluation.test_manifest_hash,
            "test_labeled_manifest_hash": primary_evaluation.test_labeled_manifest_hash,
        },
        "metric_implementations": {
            "binary_auroc": function_implementation_hash(binary_auroc),
            "mean_of_seed_binary_auroc": seed_mean_auroc_implementation_hash(),
            "average_precision": function_implementation_hash(average_precision),
            "binary_f1_at_threshold": function_implementation_hash(binary_f1),
            "grouped_bootstrap_seed_mean_metrics": function_implementation_hash(
                grouped_bootstrap_seed_mean_metrics
            ),
            "grouped_bootstrap_seed_mean_auroc_difference": function_implementation_hash(
                grouped_bootstrap_seed_mean_auroc_difference
            ),
        },
        "confidence_intervals": {
            "scope_note": (
                "sample uncertainty for the primary mean-of-seed estimand; every cluster "
                "resample evaluates all frozen seeds and then averages their metrics"
            ),
            "source_grouped": _bootstrap_payload(source_bootstrap),
            "paper_grouped": _bootstrap_payload(paper_bootstrap),
            "paired_auroc_differences": comparison_intervals,
        },
        "artifacts_sha256": artifact_hashes,
        "source_revision": source_revision,
        "source_files_sha256": source_hashes,
        "dependency_versions": dependency_versions,
    }
    report_path = args.output_dir / "experiment_report.json"
    _write_json(report_path, report)

    results_manifest = {
        "schema": "scinumbench-v2-results-manifest/1",
        "study_id": args.study_id,
        "dataset_manifest_sha256": reference.dataset_manifest_sha256,
        "test_ledger_payload_sha256": grant.ledger_payload_sha256,
        "run_spec_sha256": args.run_spec_sha256,
        "source_revision": source_revision,
        "source_files_sha256": source_hashes,
        "dependency_versions": dependency_versions,
        "feature_archives_sha256": {
            name: archive.archive_sha256 for name, archive in sorted(archives.items())
        },
        "feature_reports_sha256": feature_report_hashes,
        "result_files_sha256": {
            **artifact_hashes,
            "experiment_report": _sha256_file(report_path),
        },
    }
    results_manifest_path = args.output_dir / "results_manifest.json"
    _write_json(results_manifest_path, results_manifest)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
