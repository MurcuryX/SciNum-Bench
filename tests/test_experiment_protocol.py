from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import MappingProxyType

import pytest

import scinumbench_v2.experiment_protocol as protocol_module
from scinumbench_v2.experiment_protocol import (
    PRIMARY_PROBE_SEEDS,
    ExperimentProtocolError,
    FeatureArchive,
    FrozenThresholdChoice,
    SeededCandidateScoreSet,
    acquire_test_access_guard,
    assert_expected_model_identity,
    average_precision,
    binary_f1,
    build_probability_fusions,
    build_seeded_probability_fusions,
    evaluate_f1_with_frozen_threshold,
    evaluate_seed_mean_choice_on_test,
    fit_probe_candidates,
    grouped_bootstrap_auroc_difference,
    grouped_bootstrap_metrics,
    grouped_bootstrap_seed_mean_auroc_difference,
    grouped_bootstrap_seed_mean_metrics,
    load_verified_feature_archive,
    prepare_test_ledger_binding,
    select_f1_threshold_on_validation,
    select_seed_mean_auroc_on_validation,
    summarize_seed_metrics,
    unlock_test_partition,
)
from scinumbench_v2.freeze_manifest import (
    BLIND_TEST_SCHEMA_VERSION,
    freeze_manifest,
    manifest_json,
    write_label_blind_test_inputs,
)
from scinumbench_v2.split_protocol import (
    CandidateScoreSet,
    Partition,
    candidate_config_hash,
    compute_dataset_manifest_hash,
)


REVISION = "a" * 40


def score_set(partition: Partition, scores: tuple[float, ...]) -> CandidateScoreSet:
    ids = tuple(f"{partition.value}-{index}" for index in range(4))
    return CandidateScoreSet(
        partition=partition,
        dataset_id="release",
        sample_ids=ids,
        labels=(0, 1, 0, 1),
        candidates={"candidate": scores},
        candidate_config_hashes={"candidate": candidate_config_hash({"name": "fixed"})},
    )


def extraction_metadata(
    manifest_sha: str,
    *,
    test_inputs_sha: str = "1" * 64,
    model_id: str = "Org/Model-7B",
) -> dict:
    weight_files = [
        {"path": "model.safetensors", "bytes": 1, "sha256": "c" * 64}
    ]
    weight_manifest_sha = hashlib.sha256(
        json.dumps(
            weight_files, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    return {
        "schema": "scinumbench-v2-extraction-metadata/3",
        "dataset_manifest_sha256": manifest_sha,
        "test_inputs_sha256": test_inputs_sha,
        "test_inputs_schema_version": BLIND_TEST_SCHEMA_VERSION,
        "model_id": model_id,
        "model_revision": REVISION,
        "model_weight_manifest_sha256": weight_manifest_sha,
        "model_weight_files": weight_files,
        "model_config_sha256": "d" * 64,
        "tokenizer_id": model_id,
        "tokenizer_revision": REVISION,
        "tokenizer_sha256": "e" * 64,
        "extractor_code_sha256": "f" * 64,
        "max_length": 2048,
        "allow_truncation": False,
        "truncated_examples": 0,
        "layers": [1],
        "pooling": ["mean", "last"],
        "layer_semantics": "hidden_states[k] after block k",
        "model_dtype": "bfloat16",
        "feature_dtype": "float16",
        "transformers_version": "test",
        "torch_version": "test",
    }


def frozen_fixture(tmp_path: Path, *, model_id: str = "Org/Model-7B") -> dict:
    np = pytest.importorskip("numpy")
    split_dir = tmp_path / "splits"
    split_dir.mkdir(parents=True)
    split_paths = {}
    rows = []
    for split_index, split in enumerate(("train", "validation", "test")):
        split_rows = []
        for source_index in (0, 1):
            for label in (0, 1):
                value = split_index * 100 + source_index * 10 + label
                record = {
                    "example_id": f"{split}-{source_index}-{label}",
                    "source_id": f"source-{split}-{source_index}",
                    "src_table_id": f"source-{split}-{source_index}",
                    "paper_id": f"paper-{split}-{source_index}",
                    "arxiv_id": f"paper-{split}-{source_index}",
                    "label": label,
                    "split": split,
                    "text": (
                        f"| Method | Metric |\n| --- | --- |\n"
                        f"| Model-{split}-{source_index} | {value} |"
                    ),
                }
                split_rows.append(record)
                rows.append(record)
        path = split_dir / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in split_rows),
            encoding="utf-8",
        )
        split_paths[split] = path
    test_inputs_path = split_dir / "test_inputs.jsonl"
    write_label_blind_test_inputs(split_paths["test"], test_inputs_path)
    manifest = freeze_manifest(
        split_paths,
        seed=7,
        ratios={"train": 0.7, "validation": 0.15, "test": 0.15},
        code_revision="test-revision",
        created_at="2026-10-26T00:00:00+08:00",
        test_inputs_path=test_inputs_path,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest_json(manifest), encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest_json(manifest).encode()).hexdigest()
    metadata = extraction_metadata(
        manifest_sha,
        test_inputs_sha=str(manifest["test_inputs"]["sha256"]),
        model_id=model_id,
    )
    extractor_path = Path(__file__).resolve().parents[1] / "scripts" / "extract_hidden_features.py"
    metadata["extractor_code_sha256"] = hashlib.sha256(extractor_path.read_bytes()).hexdigest()
    metadata_json = json.dumps(
        metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    archive_path = tmp_path / "features.npz"
    labels = [(-1 if row["split"] == "test" else row["label"]) for row in rows]
    common = {
        "example_id": np.asarray([row["example_id"] for row in rows]),
        "source_id": np.asarray([row["source_id"] for row in rows]),
        "arxiv_id": np.asarray([row["arxiv_id"] for row in rows]),
        "label": np.asarray(labels),
        "split": np.asarray([row["split"] for row in rows]),
        "text_sha256": np.asarray(
            [hashlib.sha256(row["text"].encode()).hexdigest() for row in rows]
        ),
        "token_length": np.asarray([16] * len(rows)),
        "dataset_manifest_sha256": np.asarray([manifest_sha]),
        "extraction_metadata_json": np.asarray([metadata_json]),
        "extraction_metadata_sha256": np.asarray(
            [hashlib.sha256(metadata_json.encode()).hexdigest()]
        ),
        "mean_layer_1": np.arange(len(rows) * 2, dtype=np.float16).reshape(len(rows), 2),
        "last_layer_1": np.arange(
            len(rows) * 2, len(rows) * 4, dtype=np.float16
        ).reshape(
            len(rows), 2
        ),
    }
    np.savez(archive_path, **common)
    archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    report = {
        "schema_version": "scinumbench-v2-hidden-features/3",
        "model": model_id,
        "model_commit": REVISION,
        "tokenizer": model_id,
        "tokenizer_commit": REVISION,
        "model_weight_manifest_sha256": metadata["model_weight_manifest_sha256"],
        "model_config_sha256": metadata["model_config_sha256"],
        "tokenizer_sha256": metadata["tokenizer_sha256"],
        "extraction_metadata_sha256": hashlib.sha256(metadata_json.encode()).hexdigest(),
        "extractor_code_sha256": metadata["extractor_code_sha256"],
        "layers": [1],
        "pooling": ["attention_mask_mean", "last_nonpadding_token"],
        "examples": len(rows),
        "split_counts": {split: 4 for split in ("train", "validation", "test")},
        "max_length": 2048,
        "longest_untruncated_tokens": 16,
        "truncated_examples": 0,
        "batch_size": 2,
        "dtype": "bfloat16_model/float16_features",
        "dataset_manifest_sha256": manifest_sha,
        "test_inputs_sha256": str(manifest["test_inputs"]["sha256"]),
        "output_sha256": archive_sha256,
        "elapsed_seconds": 1.0,
        "examples_per_second": float(len(rows)),
        "peak_device_memory_bytes": 1,
        "device_name": "test",
        "transformers_version": "test",
        "torch_version": "test",
        "selection_labels_used": False,
        "test_labels_embedded": False,
    }
    report_path = tmp_path / "features.report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "split_dir": split_dir,
        "split_paths": split_paths,
        "manifest_path": manifest_path,
        "manifest_sha": manifest_sha,
        "test_inputs_path": test_inputs_path,
        "archive_path": archive_path,
        "feature_report_path": report_path,
        "feature_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "model_id": model_id,
        "rows": rows,
        "arrays": common,
    }


def frozen_run_spec(
    fixture: dict,
    *,
    ledger_root: Path,
    study_id: str,
    c_values: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0),
    max_iter: int = 2000,
    bootstrap_replicates: int = 10000,
    bootstrap_seed: int = 20261026,
) -> tuple[Path, str]:
    project_root = Path(__file__).resolve().parents[1]
    namespace = runpy.run_path(
        str(project_root / "scripts" / "run_probe_experiment_v2.py")
    )
    source_revision, _ = namespace["_source_revision"](project_root)
    archive_sha256 = hashlib.sha256(fixture["archive_path"].read_bytes()).hexdigest()
    report_sha256 = hashlib.sha256(
        fixture["feature_report_path"].read_bytes()
    ).hexdigest()
    ledger_binding = prepare_test_ledger_binding(
        ledger_root,
        dataset_manifest_sha256=fixture["manifest_sha"],
    )
    spec = {
        "schema": "scinumbench-v2-probe-run-spec/2",
        "created_at": "2026-10-26T00:00:00+08:00",
        "study_id": study_id,
        "dataset_manifest_sha256": fixture["manifest_sha"],
        "canonical_test_ledger_root": ledger_binding["canonical_root"],
        "test_ledger_binding": ledger_binding,
        "source_revision": source_revision,
        "features": {
            "model": {
                "archive_sha256": archive_sha256,
                "report_sha256": report_sha256,
                "model_id": fixture["model_id"],
                "model_revision": REVISION,
            }
        },
        "seeds": list(PRIMARY_PROBE_SEEDS),
        "c_grid_hex": [value.hex() for value in c_values],
        "max_iter": max_iter,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
    }
    payload = json.dumps(
        spec,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"
    name = hashlib.sha256(
        f"{study_id}\0{ledger_root.resolve()}".encode()
    ).hexdigest()[:12]
    path = fixture["manifest_path"].parent / f"run-spec-{name}.json"
    path.write_text(payload, encoding="utf-8")
    return path, hashlib.sha256(payload.encode()).hexdigest()


def direct_archive(np) -> FeatureArchive:
    features = np.asarray([[0.0], [1.0], [1000.0], [2000.0], [-900.0], [-800.0]])
    return FeatureArchive(
        path=Path("fake.npz"),
        archive_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        example_ids=("tr0", "tr1", "va0", "va1", "te0", "te1"),
        source_ids=("s0", "s0", "s1", "s1", "s2", "s2"),
        paper_ids=("p0", "p0", "p1", "p1", "p2", "p2"),
        labels=(0, 1, 0, 1, -1, -1),
        splits=("train", "train", "validation", "validation", "test", "test"),
        text_sha256=("0" * 64,) * 6,
        extraction_metadata=MappingProxyType(
            extraction_metadata("b" * 64, model_id="Org/Model-7B")
        ),
        extraction_metadata_sha256="c" * 64,
        model_id="Org/Model-7B",
        model_revision=REVISION,
        features=MappingProxyType({"mean_layer_1": features}),
    )


def guard(ledger_root: Path, manifest_sha: str, archive_sha: str, study: str = "study"):
    ledger_binding = prepare_test_ledger_binding(
        ledger_root,
        dataset_manifest_sha256=manifest_sha,
    )
    return acquire_test_access_guard(
        ledger_root,
        ledger_binding=ledger_binding,
        study_id=study,
        dataset_manifest_sha256=manifest_sha,
        protocol_digest="1" * 64,
        selection_receipts_sha256="2" * 64,
        feature_archive_hashes={"model": archive_sha},
        source_revision="3" * 64,
    )


def ledger_record_paths(ledger_root: Path, manifest_sha: str) -> list[Path]:
    directory = ledger_root / manifest_sha
    return sorted(
        path
        for path in directory.glob("*.json")
        if not path.name.startswith(".")
    )


def test_threshold_is_selected_on_validation_and_metric_bound() -> None:
    validation = score_set(Partition.VALIDATION, (0.1, 0.8, 0.6, 0.7))
    test = score_set(Partition.TEST, (0.2, 0.9, 0.4, 0.8))
    expected_test = compute_dataset_manifest_hash(Partition.TEST, "release", test.sample_ids)
    choice = select_f1_threshold_on_validation(
        validation,
        candidate_name="candidate",
        expected_test_manifest_hash=expected_test,
    )
    assert choice.selected_on is Partition.VALIDATION
    assert choice.threshold == 0.7
    assert choice.validation_f1 == 1.0
    assert choice.metric_id == "binary_f1_at_threshold"
    assert len(choice.metric_implementation_hash) == 64
    assert evaluate_f1_with_frozen_threshold(choice, test) == 1.0


def test_threshold_cannot_be_selected_on_test_or_forged() -> None:
    test = score_set(Partition.TEST, (0.2, 0.9, 0.4, 0.8))
    with pytest.raises(ExperimentProtocolError, match="validation"):
        select_f1_threshold_on_validation(
            test,
            candidate_name="candidate",
            expected_test_manifest_hash=test.dataset_manifest_hash,
        )
    with pytest.raises(ExperimentProtocolError, match="only be created"):
        FrozenThresholdChoice()


def test_probability_fusions_are_deterministic_and_config_bound() -> None:
    config_a = candidate_config_hash({"model": "a"})
    config_b = candidate_config_hash({"model": "b"})
    first = build_probability_fusions(
        {"b": (0.3, 0.9), "a": (0.1, 0.5)},
        {"a": config_a, "b": config_b},
    )
    second = build_probability_fusions(
        {"a": (0.1, 0.5), "b": (0.3, 0.9)},
        {"b": config_b, "a": config_a},
    )
    assert dict(first.scores) == dict(second.scores)
    assert first.scores["mean:a+b"] == pytest.approx((0.2, 0.7))
    assert first.members["mean:a+b"] == ("a", "b")
    assert dict(first.config_hashes) == dict(second.config_hashes)


def test_metrics_handle_ties_and_fixed_threshold() -> None:
    assert average_precision((0, 1), (0.5, 0.5)) == pytest.approx(0.5)
    assert binary_f1((0, 1, 1, 0), (0.1, 0.8, 0.7, 0.6), 0.7) == 1.0
    with pytest.raises(ValueError, match="equal"):
        average_precision((0,), (0.2, 0.3))


def test_seed_mean_and_probability_ensemble_are_distinct_estimands() -> None:
    labels = (0, 1, 0, 1)
    per_seed = {
        1: (0.0, 0.8, 0.9, 1.0),
        2: (0.9, 1.0, 0.0, 0.8),
    }
    ensemble = tuple((left + right) / 2 for left, right in zip(per_seed[1], per_seed[2]))
    report = summarize_seed_metrics(
        labels,
        per_seed,
        {1: 0.5, 2: 0.5},
        probability_ensemble_scores=ensemble,
        probability_ensemble_threshold=0.5,
    )
    assert report.seed_mean["auroc"] == pytest.approx(0.75)
    assert report.probability_ensemble["auroc"] == pytest.approx(1.0)
    assert report.seed_mean["auroc"] != report.probability_ensemble["auroc"]


def test_primary_selector_uses_mean_seed_auroc_not_mean_probability_auroc() -> None:
    labels = (0, 1, 0, 1)
    sample_ids = tuple(f"validation-{index}" for index in range(4))
    configs = {
        "A": candidate_config_hash({"candidate": "A"}),
        "B": candidate_config_hash({"candidate": "B"}),
    }
    candidates = {
        "A": {
            1: (0.0, 1.0, 2.0, 3.0),
            2: (0.0, 1.0, 2.0, 3.0),
        },
        "B": {
            1: (0.0, 1.0, 2.0, 3.0),
            2: (2.0, 3.0, 1.0, 0.0),
        },
    }
    validation = SeededCandidateScoreSet(
        partition=Partition.VALIDATION,
        dataset_id="release",
        sample_ids=sample_ids,
        labels=labels,
        seeds=(1, 2),
        candidates=candidates,
        candidate_config_hashes=configs,
    )
    test_ids = tuple(f"test-{index}" for index in range(4))
    expected_test = compute_dataset_manifest_hash(Partition.TEST, "release", test_ids)
    choice = select_seed_mean_auroc_on_validation(
        validation,
        expected_test_manifest_hash=expected_test,
    )
    assert choice.candidate_name == "A"
    assert choice.validation_score == pytest.approx(0.75)
    assert dict(choice.validation_seed_auroc) == {1: 0.75, 2: 0.75}

    # The old selector would choose B: AUROC(mean probabilities)=0.875 > 0.75.
    old_ensemble_a = tuple(
        (left + right) / 2 for left, right in zip(candidates["A"][1], candidates["A"][2])
    )
    old_ensemble_b = tuple(
        (left + right) / 2 for left, right in zip(candidates["B"][1], candidates["B"][2])
    )
    assert protocol_module.binary_auroc(labels, old_ensemble_b) == pytest.approx(0.875)
    assert protocol_module.binary_auroc(labels, old_ensemble_a) == pytest.approx(0.75)

    test = SeededCandidateScoreSet(
        partition=Partition.TEST,
        dataset_id="release",
        sample_ids=test_ids,
        labels=labels,
        seeds=(1, 2),
        candidates=candidates,
        candidate_config_hashes=configs,
    )
    evaluated = evaluate_seed_mean_choice_on_test(choice, test)
    assert evaluated.candidate_name == "A"
    assert evaluated.mean_auroc == pytest.approx(0.75)
    with pytest.raises(ExperimentProtocolError, match="validation"):
        select_seed_mean_auroc_on_validation(
            test,
            expected_test_manifest_hash=expected_test,
        )


def test_seeded_fusion_and_primary_grouped_bootstrap_follow_seed_then_mean() -> None:
    labels = (0, 1, 0, 1, 0, 1)
    groups = ("a", "a", "b", "b", "c", "c")
    seeds = (1, 2)
    base = {
        "left": {
            1: (0.1, 0.9, 0.2, 0.8, 0.3, 0.7),
            2: (0.2, 0.8, 0.3, 0.7, 0.4, 0.6),
        },
        "right": {
            1: (0.4, 0.6, 0.3, 0.7, 0.2, 0.8),
            2: (0.3, 0.7, 0.4, 0.6, 0.1, 0.9),
        },
    }
    configs = {
        name: candidate_config_hash({"name": name}) for name in base
    }
    fused = build_seeded_probability_fusions(base, configs, seeds=seeds)
    scores = fused.per_seed_scores["mean:left+right"]
    assert scores[1][0] == pytest.approx(0.25)
    first = grouped_bootstrap_seed_mean_metrics(
        labels,
        scores,
        {1: 0.5, 2: 0.5},
        groups,
        seeds=seeds,
        grouping="source_id",
        replicates=100,
        seed=17,
    )
    second = grouped_bootstrap_seed_mean_metrics(
        labels,
        scores,
        {1: 0.5, 2: 0.5},
        groups,
        seeds=seeds,
        grouping="source_id",
        replicates=100,
        seed=17,
    )
    assert first == second
    expected_point = sum(
        protocol_module.binary_auroc(labels, scores[value]) for value in seeds
    ) / len(seeds)
    assert first.point["auroc"] == pytest.approx(expected_point)
    difference = grouped_bootstrap_seed_mean_auroc_difference(
        labels,
        scores,
        base["left"],
        groups,
        seeds=seeds,
        grouping="source_id",
        replicates=100,
        seed=23,
    )
    assert difference.metric == "mean_seed_auroc_difference_left_minus_right"
    assert difference.lower <= difference.point_difference <= difference.upper


def test_grouped_bootstraps_are_paired_and_reproducible() -> None:
    labels = (0, 1, 0, 1, 0, 1)
    scores = (0.1, 0.9, 0.4, 0.6, 0.8, 0.2)
    groups = ("a", "a", "b", "b", "c", "c")
    first = grouped_bootstrap_metrics(
        labels, scores, groups, threshold=0.5, grouping="source_id", replicates=100, seed=7
    )
    second = grouped_bootstrap_metrics(
        labels, scores, groups, threshold=0.5, grouping="source_id", replicates=100, seed=7
    )
    assert first == second
    stronger = (0.1, 0.9, 0.2, 0.8, 0.3, 0.7)
    difference = grouped_bootstrap_auroc_difference(
        labels,
        stronger,
        scores,
        groups,
        grouping="paper_id",
        replicates=100,
        seed=19,
    )
    assert difference.lower <= difference.point_difference <= difference.upper


def test_global_test_guard_cannot_be_bypassed_by_output_directory(tmp_path: Path) -> None:
    ledger = tmp_path / "global-ledger"
    (tmp_path / "output-one").mkdir()
    (tmp_path / "output-two").mkdir()
    first = guard(ledger, "a" * 64, "b" * 64)
    assert first.ledger_path.is_file()
    with pytest.raises(ExperimentProtocolError, match="dataset/study"):
        guard(ledger, "a" * 64, "b" * 64)
    # Another study is a distinct preregistered evaluation, not an output-dir bypass.
    assert guard(ledger, "a" * 64, "b" * 64, study="ablation").ledger_path.is_file()


def test_ledger_freeze_rejects_root_and_dataset_symlinks(tmp_path: Path) -> None:
    manifest_a = "a" * 64
    manifest_b = "b" * 64

    target = tmp_path / "target"
    target.mkdir()
    root_link = tmp_path / "root-link"
    root_link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ExperimentProtocolError, match="symbolic-link"):
        prepare_test_ledger_binding(
            root_link,
            dataset_manifest_sha256=manifest_a,
        )

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    parent_link = tmp_path / "parent-link"
    parent_link.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ExperimentProtocolError, match="symbolic-link"):
        prepare_test_ledger_binding(
            parent_link / "ledger",
            dataset_manifest_sha256=manifest_a,
        )

    root = tmp_path / "ledger"
    prepare_test_ledger_binding(root, dataset_manifest_sha256=manifest_a)
    foreign = tmp_path / "foreign-dataset"
    foreign.mkdir()
    (root / manifest_b).symlink_to(foreign, target_is_directory=True)
    with pytest.raises(ExperimentProtocolError, match="secure ordinary directory"):
        prepare_test_ledger_binding(root, dataset_manifest_sha256=manifest_b)


def test_ledger_binding_rejects_dataset_symlink_and_inode_rotation(tmp_path: Path) -> None:
    manifest_sha = "a" * 64
    archive_sha = "b" * 64

    symlink_root = tmp_path / "symlink-ledger"
    symlink_binding = prepare_test_ledger_binding(
        symlink_root,
        dataset_manifest_sha256=manifest_sha,
    )
    first = acquire_test_access_guard(
        symlink_root,
        ledger_binding=symlink_binding,
        study_id="first",
        dataset_manifest_sha256=manifest_sha,
        protocol_digest="1" * 64,
        selection_receipts_sha256="2" * 64,
        feature_archive_hashes={"model": archive_sha},
        source_revision="3" * 64,
    )
    assert first.ledger_path.is_file()
    dataset = symlink_root / manifest_sha
    saved_dataset = tmp_path / "saved-dataset"
    dataset.rename(saved_dataset)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    dataset.symlink_to(foreign, target_is_directory=True)
    with pytest.raises(ExperimentProtocolError, match="securely reopen"):
        acquire_test_access_guard(
            symlink_root,
            ledger_binding=symlink_binding,
            study_id="second",
            dataset_manifest_sha256=manifest_sha,
            protocol_digest="1" * 64,
            selection_receipts_sha256="2" * 64,
            feature_archive_hashes={"model": archive_sha},
            source_revision="3" * 64,
        )

    replacement_root = tmp_path / "replacement-ledger"
    replacement_binding = prepare_test_ledger_binding(
        replacement_root,
        dataset_manifest_sha256=manifest_sha,
    )
    original_dataset = replacement_root / manifest_sha
    archived_dataset = tmp_path / "archived-dataset"
    original_dataset.rename(archived_dataset)
    original_dataset.mkdir()
    (original_dataset / ".scinumbench-ledger-dataset.json").write_bytes(
        (archived_dataset / ".scinumbench-ledger-dataset.json").read_bytes()
    )
    with pytest.raises(ExperimentProtocolError, match="identity changed"):
        acquire_test_access_guard(
            replacement_root,
            ledger_binding=replacement_binding,
            study_id="replacement",
            dataset_manifest_sha256=manifest_sha,
            protocol_digest="1" * 64,
            selection_receipts_sha256="2" * 64,
            feature_archive_hashes={"model": archive_sha},
            source_revision="3" * 64,
        )


def test_ledger_binding_rejects_root_replacement_even_with_copied_markers(
    tmp_path: Path,
) -> None:
    manifest_sha = "a" * 64
    root = tmp_path / "ledger"
    binding = prepare_test_ledger_binding(root, dataset_manifest_sha256=manifest_sha)
    archived_root = tmp_path / "archived-root"
    root.rename(archived_root)
    root.mkdir()
    (root / ".scinumbench-ledger-namespace.json").write_bytes(
        (archived_root / ".scinumbench-ledger-namespace.json").read_bytes()
    )
    (root / manifest_sha).mkdir()
    (root / manifest_sha / ".scinumbench-ledger-dataset.json").write_bytes(
        (
            archived_root
            / manifest_sha
            / ".scinumbench-ledger-dataset.json"
        ).read_bytes()
    )
    with pytest.raises(ExperimentProtocolError, match="root identity changed"):
        acquire_test_access_guard(
            root,
            ledger_binding=binding,
            study_id="replacement",
            dataset_manifest_sha256=manifest_sha,
            protocol_digest="1" * 64,
            selection_receipts_sha256="2" * 64,
            feature_archive_hashes={"model": "b" * 64},
            source_revision="3" * 64,
        )


def test_concurrent_same_study_consumes_exactly_once(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    manifest_sha = "a" * 64
    binding = prepare_test_ledger_binding(root, dataset_manifest_sha256=manifest_sha)

    def attempt() -> bool:
        try:
            acquire_test_access_guard(
                root,
                ledger_binding=binding,
                study_id="concurrent",
                dataset_manifest_sha256=manifest_sha,
                protocol_digest="1" * 64,
                selection_receipts_sha256="2" * 64,
                feature_archive_hashes={"model": "b" * 64},
                source_revision="3" * 64,
            )
            return True
        except ExperimentProtocolError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: attempt(), range(2)))
    assert outcomes.count(True) == 1
    assert outcomes.count(False) == 1


def test_loader_never_opens_test_jsonl_and_test_labels_are_sealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = frozen_fixture(tmp_path)
    original = protocol_module._read_partition
    original_open = Path.open
    original_stat = Path.stat
    labeled_test = fixture["split_paths"]["test"]

    def guarded_read(path: Path, split: str):
        if split == "test":
            raise AssertionError("pre-guard test access")
        return original(path, split)

    def guarded_open(self: Path, *args, **kwargs):
        if self == labeled_test:
            raise AssertionError("pre-guard labeled test open")
        return original_open(self, *args, **kwargs)

    def guarded_stat(self: Path, *args, **kwargs):
        if self == labeled_test:
            raise AssertionError("pre-guard labeled test stat/hash")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(protocol_module, "_read_partition", guarded_read)
    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "stat", guarded_stat)
    loaded = load_verified_feature_archive(
        fixture["archive_path"],
        split_dir=fixture["split_dir"],
        manifest_path=fixture["manifest_path"],
    )
    assert loaded.labels[-4:] == (-1, -1, -1, -1)
    with pytest.raises(ExperimentProtocolError, match="sealed"):
        loaded.selection_labels(Partition.TEST)


def test_extractor_loader_never_opens_stats_or_hashes_labeled_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = frozen_fixture(tmp_path)
    project_root = Path(__file__).resolve().parents[1]
    namespace = runpy.run_path(str(project_root / "scripts" / "extract_hidden_features.py"))
    load_rows = namespace["_load_rows"]
    manifest = json.loads(fixture["manifest_path"].read_text(encoding="utf-8"))
    labeled_test = fixture["split_paths"]["test"]
    original_open = Path.open
    original_stat = Path.stat

    def guarded_open(self: Path, *args, **kwargs):
        if self == labeled_test:
            raise AssertionError("extractor opened labeled test before ledger")
        return original_open(self, *args, **kwargs)

    def guarded_stat(self: Path, *args, **kwargs):
        if self == labeled_test:
            raise AssertionError("extractor statted labeled test before ledger")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "stat", guarded_stat)
    rows = load_rows(fixture["split_dir"], manifest)
    test_rows = [row for row in rows if row["split"] == "test"]
    assert len(test_rows) == 4
    assert all("label" not in row for row in test_rows)


def test_guard_then_unlock_validates_test_text_and_labels(tmp_path: Path) -> None:
    fixture = frozen_fixture(tmp_path)
    loaded = load_verified_feature_archive(
        fixture["archive_path"],
        split_dir=fixture["split_dir"],
        manifest_path=fixture["manifest_path"],
    )
    grant = guard(tmp_path / "ledger", fixture["manifest_sha"], loaded.archive_sha256)
    test = unlock_test_partition(
        loaded,
        grant,
        split_dir=fixture["split_dir"],
        manifest_path=fixture["manifest_path"],
    )
    assert test.labels == (0, 1, 0, 1)
    assert test.example_ids == ("test-0-0", "test-0-1", "test-1-0", "test-1-1")


def test_unlock_rechecks_binding_before_opening_labeled_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = frozen_fixture(tmp_path / "fixture")
    loaded = load_verified_feature_archive(
        fixture["archive_path"],
        split_dir=fixture["split_dir"],
        manifest_path=fixture["manifest_path"],
    )
    root = tmp_path / "ledger"
    grant = guard(root, fixture["manifest_sha"], loaded.archive_sha256)
    dataset = root / fixture["manifest_sha"]
    dataset.rename(tmp_path / "original-dataset")
    foreign = tmp_path / "foreign-dataset"
    foreign.mkdir()
    dataset.symlink_to(foreign, target_is_directory=True)

    original_read = protocol_module._read_partition
    labeled_test_opened = False

    def track_read(path: Path, split: str):
        nonlocal labeled_test_opened
        if split == "test":
            labeled_test_opened = True
        return original_read(path, split)

    monkeypatch.setattr(protocol_module, "_read_partition", track_read)
    with pytest.raises(ExperimentProtocolError, match="binding or record changed"):
        unlock_test_partition(
            loaded,
            grant,
            split_dir=fixture["split_dir"],
            manifest_path=fixture["manifest_path"],
        )
    assert labeled_test_opened is False


def test_loader_rejects_tampered_text_hash_and_model_metadata(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    fixture = frozen_fixture(tmp_path)
    arrays = dict(fixture["arrays"])
    text_hashes = arrays["text_sha256"].copy()
    text_hashes[2] = "0" * 64
    arrays["text_sha256"] = text_hashes
    np.savez(fixture["archive_path"], **arrays)
    with pytest.raises(ExperimentProtocolError, match="text_sha256"):
        load_verified_feature_archive(
            fixture["archive_path"],
            split_dir=fixture["split_dir"],
            manifest_path=fixture["manifest_path"],
        )

    tampered = frozen_fixture(tmp_path / "metadata-tamper")
    tampered_arrays = dict(tampered["arrays"])
    tampered_arrays["extraction_metadata_json"] = np.asarray(
        [str(tampered_arrays["extraction_metadata_json"][0]).replace("Model-7B", "Model-14B")]
    )
    np.savez(tampered["archive_path"], **tampered_arrays)
    with pytest.raises(ExperimentProtocolError, match="metadata hash"):
        load_verified_feature_archive(
            tampered["archive_path"],
            split_dir=tampered["split_dir"],
            manifest_path=tampered["manifest_path"],
        )

    fixture = frozen_fixture(tmp_path / "valid-14b", model_id="Org/Model-14B")
    loaded = load_verified_feature_archive(
        fixture["archive_path"],
        split_dir=fixture["split_dir"],
        manifest_path=fixture["manifest_path"],
    )
    with pytest.raises(ExperimentProtocolError, match="archive proves"):
        assert_expected_model_identity(
            "qwen7b",
            loaded,
            expected_model_id="Org/Model-7B",
            expected_revision=REVISION,
        )


def test_probe_scaler_is_train_only_and_nonconvergence_fails(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    archive = direct_archive(np)
    search = fit_probe_candidates(
        archive,
        archive_name="model",
        dataset_id="release",
        c_values=(1.0,),
        seeds=(42,),
        max_iter=2000,
    )
    candidate = next(iter(search.candidates.values()))
    scaler, _ = candidate.fitted_estimators[42]
    assert scaler.mean_.tolist() == pytest.approx([0.5])
    assert candidate.n_iter_by_seed[42] < 2000

    with pytest.raises(ExperimentProtocolError, match="max_iter|ConvergenceWarning"):
        fit_probe_candidates(
            archive,
            archive_name="model",
            dataset_id="release",
            c_values=(1.0,),
            seeds=(42,),
            max_iter=1,
        )


def test_tiny_end_to_end_runner_freezes_results_manifest(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    fixture = frozen_fixture(tmp_path)
    project_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "results"
    ledger_root = tmp_path / "ledger"
    run_spec, run_spec_sha256 = frozen_run_spec(
        fixture,
        ledger_root=ledger_root,
        study_id="tiny-integration",
        c_values=(1.0,),
        bootstrap_replicates=20,
    )
    command = [
        sys.executable,
        str(project_root / "scripts" / "run_probe_experiment_v2.py"),
        "--split-dir",
        str(fixture["split_dir"]),
        "--manifest",
        str(fixture["manifest_path"]),
        "--feature",
        f"model={fixture['archive_path']}",
        "--feature-report",
        f"model={fixture['feature_report_path']}",
        "--expected-model",
        f"model=Org/Model-7B@{REVISION}",
        "--run-spec",
        str(run_spec),
        "--run-spec-sha256",
        run_spec_sha256,
        "--study-id",
        "tiny-integration",
        "--test-ledger-root",
        str(ledger_root),
        "--output-dir",
        str(output_dir),
        "--c-grid",
        "1",
        "--bootstrap-replicates",
        "20",
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project_root / "src")
    completed = subprocess.run(
        command,
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads((output_dir / "experiment_report.json").read_text())
    assert report["seed_count"] == 10
    assert report["seeds"] == list(PRIMARY_PROBE_SEEDS)
    assert report["primary_estimand"].startswith("mean of 10")
    assert report["selection_estimand"].startswith("arithmetic mean of 10")
    assert "probability_ensemble_secondary" in report["test"]["seed_metrics"]
    assert report["test"]["primary_seed_mean_auroc"] == pytest.approx(
        report["test"]["seed_metrics"]["mean"]["auroc"]
    )
    for metric in ("auroc", "average_precision", "f1"):
        assert report["confidence_intervals"]["source_grouped"]["point"][
            metric
        ] == pytest.approx(report["test"]["seed_metrics"]["mean"][metric])
    leaderboard = json.loads((output_dir / "validation_leaderboard.json").read_text())
    assert leaderboard["selection_estimand"].startswith("arithmetic mean of 10")
    receipts = json.loads((output_dir / "validation_selection_receipts.json").read_text())
    assert receipts["fusion_choice"]["metric_id"] == "mean_of_seed_binary_auroc"
    assert receipts["fusion_choice"]["seeds"] == list(PRIMARY_PROBE_SEEDS)
    results_manifest = json.loads((output_dir / "results_manifest.json").read_text())
    expected_paths = {
        "selection_receipts": "validation_selection_receipts.json",
        "validation_leaderboard": "validation_leaderboard.json",
        "validation_predictions": "validation_predictions.jsonl",
        "test_predictions": "test_predictions.jsonl",
        "selected_parameters": "selected_probe_parameters.npz",
        "experiment_report": "experiment_report.json",
    }
    for key, name in expected_paths.items():
        digest = hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
        assert results_manifest["result_files_sha256"][key] == digest


def test_runner_rejects_seed_drift_and_either_path_nesting(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    fixture = frozen_fixture(tmp_path)
    project_root = Path(__file__).resolve().parents[1]
    base = [
        sys.executable,
        str(project_root / "scripts" / "run_probe_experiment_v2.py"),
        "--split-dir",
        str(fixture["split_dir"]),
        "--manifest",
        str(fixture["manifest_path"]),
        "--feature",
        f"model={fixture['archive_path']}",
        "--feature-report",
        f"model={fixture['feature_report_path']}",
        "--expected-model",
        f"model=Org/Model-7B@{REVISION}",
        "--study-id",
        "preflight-rejection",
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project_root / "src")

    ledger_a = tmp_path / "ledger-a"
    spec_a, spec_a_sha = frozen_run_spec(
        fixture,
        ledger_root=ledger_a,
        study_id="preflight-rejection",
    )
    wrong_seed = subprocess.run(
        [
            *base,
            "--run-spec",
            str(spec_a),
            "--run-spec-sha256",
            spec_a_sha,
            "--test-ledger-root",
            str(ledger_a),
            "--output-dir",
            str(tmp_path / "output-a"),
            "--seeds",
            "1,2,3,4,5,6,7,8,9,10",
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert wrong_seed.returncode != 0
    assert "exactly match" in wrong_seed.stderr
    assert ledger_record_paths(ledger_a, fixture["manifest_sha"]) == []

    ledger = tmp_path / "ledger-b"
    spec_b, spec_b_sha = frozen_run_spec(
        fixture,
        ledger_root=ledger,
        study_id="preflight-rejection",
    )
    output_inside_ledger = ledger / "results"
    nested_output = subprocess.run(
        [
            *base,
            "--run-spec",
            str(spec_b),
            "--run-spec-sha256",
            spec_b_sha,
            "--test-ledger-root",
            str(ledger),
            "--output-dir",
            str(output_inside_ledger),
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert nested_output.returncode != 0
    assert "independent" in nested_output.stderr
    assert ledger_record_paths(ledger, fixture["manifest_sha"]) == []

    output = tmp_path / "output-c"
    ledger_inside_output = output / "ledger"
    spec_c, spec_c_sha = frozen_run_spec(
        fixture,
        ledger_root=ledger_inside_output,
        study_id="preflight-rejection",
    )
    nested_ledger = subprocess.run(
        [
            *base,
            "--run-spec",
            str(spec_c),
            "--run-spec-sha256",
            spec_c_sha,
            "--test-ledger-root",
            str(ledger_inside_output),
            "--output-dir",
            str(output),
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert nested_ledger.returncode != 0
    assert "independent" in nested_ledger.stderr
    assert ledger_record_paths(ledger_inside_output, fixture["manifest_sha"]) == []


def test_runner_rejects_npz_tamper_and_ledger_root_drift_before_grant(
    tmp_path: Path,
) -> None:
    np = pytest.importorskip("numpy")
    fixture = frozen_fixture(tmp_path / "npz")
    project_root = Path(__file__).resolve().parents[1]
    ledger_root = tmp_path / "canonical-ledger"
    run_spec, run_spec_sha256 = frozen_run_spec(
        fixture,
        ledger_root=ledger_root,
        study_id="tamper-preflight",
        c_values=(1.0,),
        bootstrap_replicates=20,
    )
    base = [
        sys.executable,
        str(project_root / "scripts" / "run_probe_experiment_v2.py"),
        "--split-dir",
        str(fixture["split_dir"]),
        "--manifest",
        str(fixture["manifest_path"]),
        "--feature",
        f"model={fixture['archive_path']}",
        "--feature-report",
        f"model={fixture['feature_report_path']}",
        "--expected-model",
        f"model=Org/Model-7B@{REVISION}",
        "--run-spec",
        str(run_spec),
        "--run-spec-sha256",
        run_spec_sha256,
        "--study-id",
        "tamper-preflight",
        "--c-grid",
        "1",
        "--bootstrap-replicates",
        "20",
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project_root / "src")

    changed = dict(fixture["arrays"])
    changed_feature = np.array(changed["mean_layer_1"], copy=True)
    changed_feature[0, 0] += np.float16(1.0)
    changed["mean_layer_1"] = changed_feature
    np.savez(fixture["archive_path"], **changed)
    tampered = subprocess.run(
        [
            *base,
            "--test-ledger-root",
            str(ledger_root),
            "--output-dir",
            str(tmp_path / "tampered-output"),
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert tampered.returncode != 0
    assert "run spec feature binding mismatch" in tampered.stderr
    assert ledger_record_paths(ledger_root, fixture["manifest_sha"]) == []

    clean_fixture = frozen_fixture(tmp_path / "ledger-drift")
    run_spec, run_spec_sha256 = frozen_run_spec(
        clean_fixture,
        ledger_root=ledger_root,
        study_id="ledger-root-preflight",
        c_values=(1.0,),
        bootstrap_replicates=20,
    )
    alternate_root = tmp_path / "alternate-ledger"
    drifted = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "run_probe_experiment_v2.py"),
            "--split-dir",
            str(clean_fixture["split_dir"]),
            "--manifest",
            str(clean_fixture["manifest_path"]),
            "--feature",
            f"model={clean_fixture['archive_path']}",
            "--feature-report",
            f"model={clean_fixture['feature_report_path']}",
            "--expected-model",
            f"model=Org/Model-7B@{REVISION}",
            "--run-spec",
            str(run_spec),
            "--run-spec-sha256",
            run_spec_sha256,
            "--study-id",
            "ledger-root-preflight",
            "--test-ledger-root",
            str(alternate_root),
            "--output-dir",
            str(tmp_path / "drift-output"),
            "--c-grid",
            "1",
            "--bootstrap-replicates",
            "20",
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert drifted.returncode != 0
    assert "test ledger root differs from the frozen binding" in drifted.stderr
    assert not alternate_root.exists()


def test_runner_rejects_frozen_dataset_ledger_rotation_before_output(
    tmp_path: Path,
) -> None:
    pytest.importorskip("numpy")
    fixture = frozen_fixture(tmp_path / "fixture")
    project_root = Path(__file__).resolve().parents[1]
    ledger_root = tmp_path / "ledger"
    run_spec, run_spec_sha256 = frozen_run_spec(
        fixture,
        ledger_root=ledger_root,
        study_id="ledger-rotation-preflight",
        c_values=(1.0,),
        bootstrap_replicates=20,
    )
    dataset = ledger_root / fixture["manifest_sha"]
    dataset.rename(tmp_path / "bound-dataset")
    foreign = tmp_path / "foreign-dataset"
    foreign.mkdir()
    dataset.symlink_to(foreign, target_is_directory=True)
    output = tmp_path / "output"
    command = [
        sys.executable,
        str(project_root / "scripts" / "run_probe_experiment_v2.py"),
        "--split-dir",
        str(fixture["split_dir"]),
        "--manifest",
        str(fixture["manifest_path"]),
        "--feature",
        f"model={fixture['archive_path']}",
        "--feature-report",
        f"model={fixture['feature_report_path']}",
        "--expected-model",
        f"model=Org/Model-7B@{REVISION}",
        "--run-spec",
        str(run_spec),
        "--run-spec-sha256",
        run_spec_sha256,
        "--study-id",
        "ledger-rotation-preflight",
        "--test-ledger-root",
        str(ledger_root),
        "--output-dir",
        str(output),
        "--c-grid",
        "1",
        "--bootstrap-replicates",
        "20",
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project_root / "src")
    completed = subprocess.run(
        command,
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode != 0
    assert "test ledger binding failed" in completed.stderr
    assert not output.exists()
    assert list(foreign.iterdir()) == []

