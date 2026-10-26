"""Fail-closed experiment protocol for SciNumBench V2.

The module keeps the three information boundaries explicit:

* probe parameters are fitted on ``train`` only;
* architecture, regularisation, fusion and thresholds are chosen on
  ``validation`` only;
* ``test`` is evaluated once, after all choices have been sealed.

The implementation intentionally does not import NumPy or scikit-learn at
module import time.  Dataset-contract tests therefore remain runnable in a
minimal installation; experiment functions import their numerical dependencies
only when called.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import hmac
import itertools
import json
import marshal
import math
import os
from pathlib import Path
import random
import re
import secrets
import stat as stat_module
from types import MappingProxyType
from typing import Any, Hashable, Mapping, Sequence

from .freeze_manifest import (
    BLIND_TEST_SCHEMA_VERSION,
    ManifestError,
    load_public_extraction_rows,
    manifest_json,
    verify_manifest,
)
from .split_protocol import (
    CandidateScoreSet,
    FrozenCandidateChoice,
    Partition,
    ProtocolViolation,
    binary_auroc,
    candidate_config_hash,
    compute_dataset_manifest_hash,
)


EXPERIMENT_PROTOCOL_VERSION = "scinumbench-v2-experiment/2"
PRIMARY_PROBE_SEEDS = (13, 29, 42, 67, 101, 137, 173, 211, 251, 307)
_FEATURE_KEY = re.compile(r"^(mean|last)_layer_([1-9][0-9]*)$")


class ExperimentProtocolError(ProtocolViolation):
    """An experiment attempted to cross a frozen information boundary."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("value must be finite JSON-serializable data") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str, field_name: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ExperimentProtocolError(f"{field_name} must be a lowercase SHA-256 digest")


def _require_revision(value: str, field_name: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ExperimentProtocolError(f"{field_name} must be an exact lowercase 40-hex commit")


def function_implementation_hash(function: Any) -> str:
    """Hash a metric's Python implementation for result/receipt binding."""

    code = getattr(function, "__code__", None)
    if code is None:
        raise ValueError("function must expose Python bytecode")
    payload = marshal.dumps(code)
    payload += repr(getattr(function, "__defaults__", None)).encode()
    payload += repr(getattr(function, "__kwdefaults__", None)).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class FeatureArchive:
    """Verified feature archive aligned exactly to a frozen split bundle."""

    path: Path
    archive_sha256: str
    dataset_manifest_sha256: str
    example_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    paper_ids: tuple[str, ...]
    labels: tuple[int, ...]
    splits: tuple[str, ...]
    text_sha256: tuple[str, ...]
    extraction_metadata: Mapping[str, object]
    extraction_metadata_sha256: str
    model_id: str
    model_revision: str
    features: Mapping[str, Any] = field(repr=False, compare=False)

    @property
    def feature_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self.features))

    def indices(self, partition: Partition | str) -> tuple[int, ...]:
        name = Partition(partition).value
        return tuple(index for index, split in enumerate(self.splits) if split == name)

    def selection_labels(self, partition: Partition | str) -> tuple[int, ...]:
        selected = Partition(partition)
        if selected is Partition.TEST:
            raise ExperimentProtocolError(
                "test labels are sealed until a dataset/study test grant is consumed"
            )
        indices = self.indices(selected)
        labels = tuple(self.labels[index] for index in indices)
        if set(labels) != {0, 1}:
            raise ExperimentProtocolError(f"{selected.value} lacks one binary class")
        return labels


_TEST_ACCESS_SEAL = object()


@dataclass(frozen=True, init=False)
class TestAccessGrant:
    ledger_path: Path
    study_id: str
    dataset_manifest_sha256: str
    protocol_digest: str
    ledger_payload_sha256: str
    feature_archive_sha256: tuple[str, ...]
    ledger_binding: Mapping[str, object]

    def __init__(
        self,
        *,
        _seal: object | None = None,
        ledger_path: Path = Path("."),
        study_id: str = "",
        dataset_manifest_sha256: str = "",
        protocol_digest: str = "",
        ledger_payload_sha256: str = "",
        feature_archive_sha256: tuple[str, ...] = (),
        ledger_binding: Mapping[str, object] | None = None,
    ) -> None:
        if _seal is not _TEST_ACCESS_SEAL:
            raise ExperimentProtocolError("TestAccessGrant can only come from the global ledger")
        object.__setattr__(self, "ledger_path", ledger_path)
        object.__setattr__(self, "study_id", study_id)
        object.__setattr__(self, "dataset_manifest_sha256", dataset_manifest_sha256)
        object.__setattr__(self, "protocol_digest", protocol_digest)
        object.__setattr__(self, "ledger_payload_sha256", ledger_payload_sha256)
        object.__setattr__(self, "feature_archive_sha256", feature_archive_sha256)
        object.__setattr__(
            self,
            "ledger_binding",
            MappingProxyType(dict(ledger_binding or {})),
        )


@dataclass(frozen=True)
class UnlockedTestPartition:
    archive_sha256: str
    dataset_manifest_sha256: str
    ledger_payload_sha256: str
    indices: tuple[int, ...]
    example_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    paper_ids: tuple[str, ...]
    labels: tuple[int, ...]


def _read_partition(path: Path, split: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ExperimentProtocolError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise ExperimentProtocolError(f"{path}:{line_number}: record must be an object")
            if record.get("split") != split:
                raise ExperimentProtocolError(f"{path}:{line_number}: embedded split mismatch")
            rows.append(record)
    return rows


def _manifest_file_entry(manifest: Mapping[str, object], split: str) -> Mapping[str, object]:
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not isinstance(files.get(split), Mapping):
        raise ExperimentProtocolError(f"manifest lacks {split} file entry")
    return files[split]  # type: ignore[return-value]


def _verify_public_partition(
    manifest: Mapping[str, object], split: str, path: Path
) -> list[dict[str, object]]:
    """Verify train/validation without opening the sealed test JSONL."""

    entry = _manifest_file_entry(manifest, split)
    expected_sha = str(entry.get("sha256", ""))
    _require_sha256(expected_sha, f"manifest {split} sha256")
    if _sha256_file(path) != expected_sha:
        raise ExperimentProtocolError(f"{split} JSONL differs from frozen manifest")
    rows = _read_partition(path, split)
    if int(entry.get("rows", -1)) != len(rows):
        raise ExperimentProtocolError(f"{split} row count differs from frozen manifest")
    return rows


_EXTRACTION_METADATA_FIELDS = {
    "schema",
    "dataset_manifest_sha256",
    "test_inputs_sha256",
    "test_inputs_schema_version",
    "model_id",
    "model_revision",
    "model_weight_manifest_sha256",
    "model_weight_files",
    "model_config_sha256",
    "tokenizer_id",
    "tokenizer_revision",
    "tokenizer_sha256",
    "extractor_code_sha256",
    "max_length",
    "allow_truncation",
    "truncated_examples",
    "layers",
    "pooling",
    "layer_semantics",
    "model_dtype",
    "feature_dtype",
    "transformers_version",
    "torch_version",
}


def _validate_extraction_metadata(
    metadata: object,
    *,
    manifest_sha: str,
    test_inputs_sha: str,
) -> dict[str, object]:
    if not isinstance(metadata, dict):
        raise ExperimentProtocolError("extraction metadata must be a JSON object")
    missing = sorted(_EXTRACTION_METADATA_FIELDS - set(metadata))
    if missing:
        raise ExperimentProtocolError(f"extraction metadata lacks fields: {missing}")
    if metadata["schema"] != "scinumbench-v2-extraction-metadata/3":
        raise ExperimentProtocolError("unsupported extraction metadata schema")
    if metadata["dataset_manifest_sha256"] != manifest_sha:
        raise ExperimentProtocolError("extraction metadata belongs to another dataset")
    if metadata["test_inputs_sha256"] != test_inputs_sha:
        raise ExperimentProtocolError("extraction metadata belongs to another blind test view")
    if metadata["test_inputs_schema_version"] != BLIND_TEST_SCHEMA_VERSION:
        raise ExperimentProtocolError("unsupported extraction blind-test schema")
    for field_name in (
        "test_inputs_sha256",
        "model_weight_manifest_sha256",
        "model_config_sha256",
        "tokenizer_sha256",
        "extractor_code_sha256",
    ):
        _require_sha256(str(metadata[field_name]), field_name)
    _require_revision(str(metadata["model_revision"]), "model_revision")
    _require_revision(str(metadata["tokenizer_revision"]), "tokenizer_revision")
    for field_name in ("model_id", "tokenizer_id", "layer_semantics", "model_dtype", "feature_dtype"):
        if not isinstance(metadata[field_name], str) or not str(metadata[field_name]).strip():
            raise ExperimentProtocolError(f"extraction metadata {field_name} must be non-empty")
    layers = metadata["layers"]
    pooling = metadata["pooling"]
    if (
        not isinstance(layers, list)
        or not layers
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in layers)
        or len(set(layers)) != len(layers)
    ):
        raise ExperimentProtocolError("extraction metadata layers are invalid")
    if pooling != ["mean", "last"]:
        raise ExperimentProtocolError("extraction metadata pooling must be ['mean','last']")
    weight_files = metadata["model_weight_files"]
    if not isinstance(weight_files, list) or not weight_files:
        raise ExperimentProtocolError("model weight manifest must be a non-empty list")
    for entry in weight_files:
        if not isinstance(entry, dict):
            raise ExperimentProtocolError("model weight manifest entry must be an object")
        relative = entry.get("path")
        size = entry.get("bytes")
        digest = entry.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ExperimentProtocolError("model weight manifest path is unsafe")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ExperimentProtocolError("model weight manifest size is invalid")
        _require_sha256(str(digest), "model weight file sha256")
    computed_weight_manifest = hashlib.sha256(_canonical_json(weight_files)).hexdigest()
    if computed_weight_manifest != metadata["model_weight_manifest_sha256"]:
        raise ExperimentProtocolError("model weight manifest digest mismatch")
    if not isinstance(metadata["max_length"], int) or int(metadata["max_length"]) <= 0:
        raise ExperimentProtocolError("extraction metadata max_length is invalid")
    if not isinstance(metadata["truncated_examples"], int) or int(metadata["truncated_examples"]) < 0:
        raise ExperimentProtocolError("extraction metadata truncated_examples is invalid")
    if metadata["allow_truncation"] is not True and int(metadata["truncated_examples"]) != 0:
        raise ExperimentProtocolError("archive claims truncation without explicit permission")
    return metadata


def load_verified_feature_archive(
    path: str | os.PathLike[str],
    *,
    split_dir: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
) -> FeatureArchive:
    """Load selection data without opening the frozen test JSONL.

    Test labels in the archive must be the sentinel ``-1``.  The actual test
    JSONL is opened only by :func:`unlock_test_partition` after a global ledger
    grant has been atomically consumed.
    """

    import numpy as np

    feature_path = Path(path)
    manifest_file = Path(manifest_path)
    split_root = Path(split_dir)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest_sha = hashlib.sha256(manifest_json(manifest).encode("utf-8")).hexdigest()
    try:
        public_partitions = load_public_extraction_rows(manifest, split_root)
    except ManifestError as exc:
        raise ExperimentProtocolError(str(exc)) from exc
    selection_rows = [
        *public_partitions["train"],
        *public_partitions["validation"],
    ]
    blind_test_rows = public_partitions["test"]
    public_rows = [*selection_rows, *blind_test_rows]
    blind_entry = manifest.get("test_inputs")
    if not isinstance(blind_entry, Mapping):
        raise ExperimentProtocolError("manifest lacks frozen label-blind test inputs")
    test_inputs_sha = str(blind_entry.get("sha256", ""))
    _require_sha256(test_inputs_sha, "test_inputs sha256")

    with np.load(feature_path, allow_pickle=False) as archive:
        required = {
            "example_id",
            "source_id",
            "arxiv_id",
            "label",
            "split",
            "text_sha256",
            "token_length",
            "dataset_manifest_sha256",
            "extraction_metadata_json",
            "extraction_metadata_sha256",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ExperimentProtocolError(f"feature archive lacks keys: {missing}")
        stored_manifest = tuple(str(value) for value in archive["dataset_manifest_sha256"])
        if stored_manifest != (manifest_sha,):
            raise ExperimentProtocolError("feature archive belongs to a different frozen dataset")
        metadata_values = tuple(str(value) for value in archive["extraction_metadata_json"])
        metadata_hash_values = tuple(
            str(value) for value in archive["extraction_metadata_sha256"]
        )
        if len(metadata_values) != 1 or len(metadata_hash_values) != 1:
            raise ExperimentProtocolError("extraction metadata must be stored exactly once")
        metadata_hash = hashlib.sha256(metadata_values[0].encode("utf-8")).hexdigest()
        if metadata_hash_values != (metadata_hash,):
            raise ExperimentProtocolError("extraction metadata hash mismatch")
        try:
            metadata = _validate_extraction_metadata(
                json.loads(metadata_values[0]),
                manifest_sha=manifest_sha,
                test_inputs_sha=test_inputs_sha,
            )
        except json.JSONDecodeError as exc:
            raise ExperimentProtocolError("extraction metadata is invalid JSON") from exc

        row_count = len(public_rows)
        observed = {
            "example_id": tuple(str(value) for value in archive["example_id"]),
            "source_id": tuple(str(value) for value in archive["source_id"]),
            "arxiv_id": tuple(str(value) for value in archive["arxiv_id"]),
            "label": tuple(int(value) for value in archive["label"]),
            "split": tuple(str(value) for value in archive["split"]),
            "text_sha256": tuple(str(value) for value in archive["text_sha256"]),
            "token_length": tuple(int(value) for value in archive["token_length"]),
        }
        if any(len(values) != row_count for values in observed.values()):
            raise ExperimentProtocolError("feature archive metadata arrays have wrong length")
        expected_public = {
            "example_id": tuple(str(row["example_id"]) for row in public_rows),
            "source_id": tuple(str(row["source_id"]) for row in public_rows),
            "arxiv_id": tuple(
                str(row.get("paper_id") or row.get("arxiv_id") or "") for row in public_rows
            ),
            "label": tuple(
                -1 if str(row["split"]) == "test" else int(row["label"])
                for row in public_rows
            ),
            "split": tuple(str(row["split"]) for row in public_rows),
            "text_sha256": tuple(
                hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest()
                for row in public_rows
            ),
        }
        for key, expected_values in expected_public.items():
            if observed[key] != expected_values:
                raise ExperimentProtocolError(
                    f"feature archive {key} differs from frozen public extraction inputs"
                )
        expected_split_layout = (
            ("train",) * len(public_partitions["train"])
            + ("validation",) * len(public_partitions["validation"])
            + ("test",) * len(blind_test_rows)
        )
        if observed["split"] != expected_split_layout:
            raise ExperimentProtocolError("feature archive split layout is not train/validation/test")
        selection_count = len(selection_rows)
        if observed["label"][selection_count:] != (-1,) * len(blind_test_rows):
            raise ExperimentProtocolError("feature archive must seal every test label as -1")
        if len(set(observed["example_id"])) != row_count:
            raise ExperimentProtocolError("feature archive example IDs are not unique")
        if not all(observed["source_id"]) or not all(observed["arxiv_id"]):
            raise ExperimentProtocolError("every example must have a paper identity")
        if any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in observed["text_sha256"]):
            raise ExperimentProtocolError("feature archive contains invalid text hashes")
        if any(value <= 0 for value in observed["token_length"]):
            raise ExperimentProtocolError("feature archive contains invalid token lengths")
        observed_truncated = sum(
            value > int(metadata["max_length"]) for value in observed["token_length"]
        )
        if observed_truncated != int(metadata["truncated_examples"]):
            raise ExperimentProtocolError("token lengths disagree with truncation metadata")

        feature_arrays: dict[str, Any] = {}
        for key in archive.files:
            if _FEATURE_KEY.fullmatch(key) is None:
                continue
            values = np.asarray(archive[key])
            if values.ndim != 2 or values.shape[0] != row_count or values.shape[1] == 0:
                raise ExperimentProtocolError(f"feature matrix {key!r} has invalid shape")
            if not np.isfinite(values).all():
                raise ExperimentProtocolError(f"feature matrix {key!r} contains non-finite values")
            if str(values.dtype) != str(metadata["feature_dtype"]):
                raise ExperimentProtocolError(
                    f"feature matrix {key!r} dtype disagrees with extraction metadata"
                )
            copied = np.array(values, dtype=np.float64, copy=True)
            copied.setflags(write=False)
            feature_arrays[key] = copied
        if not feature_arrays:
            raise ExperimentProtocolError("feature archive contains no supported feature matrix")
        expected_feature_keys = {
            f"{pooling}_layer_{layer}"
            for pooling in metadata["pooling"]  # type: ignore[union-attr]
            for layer in metadata["layers"]  # type: ignore[union-attr]
        }
        if set(feature_arrays) != expected_feature_keys:
            raise ExperimentProtocolError("feature matrices disagree with extraction metadata")

    for split in ("train", "validation"):
        split_labels = {
            observed["label"][index]
            for index, value in enumerate(observed["split"])
            if value == split
        }
        if split_labels != {0, 1}:
            raise ExperimentProtocolError(f"{split} lacks one binary class")

    return FeatureArchive(
        path=feature_path,
        archive_sha256=_sha256_file(feature_path),
        dataset_manifest_sha256=manifest_sha,
        example_ids=observed["example_id"],
        source_ids=observed["source_id"],
        paper_ids=observed["arxiv_id"],
        labels=observed["label"],
        splits=observed["split"],
        text_sha256=observed["text_sha256"],
        extraction_metadata=MappingProxyType(metadata),
        extraction_metadata_sha256=metadata_hash,
        model_id=str(metadata["model_id"]),
        model_revision=str(metadata["model_revision"]),
        features=MappingProxyType(feature_arrays),
    )


def assert_expected_model_identity(
    alias: str,
    archive: FeatureArchive,
    *,
    expected_model_id: str,
    expected_revision: str,
) -> None:
    """Prevent a display alias from substituting for model provenance."""

    _require_revision(expected_revision, f"expected revision for {alias}")
    if archive.model_id != expected_model_id or archive.model_revision != expected_revision:
        raise ExperimentProtocolError(
            f"feature alias {alias!r} expected {expected_model_id}@{expected_revision}, "
            f"archive proves {archive.model_id}@{archive.model_revision}"
        )


def assert_feature_archives_aligned(archives: Mapping[str, FeatureArchive]) -> None:
    """Require all model archives to describe the exact same ordered examples."""

    if not archives:
        raise ValueError("at least one feature archive is required")
    items = sorted(archives.items())
    reference_name, reference = items[0]
    fields = (
        "dataset_manifest_sha256",
        "example_ids",
        "source_ids",
        "paper_ids",
        "labels",
        "splits",
        "text_sha256",
    )
    for name, archive in items[1:]:
        for field_name in fields:
            if getattr(archive, field_name) != getattr(reference, field_name):
                raise ExperimentProtocolError(
                    f"feature archives {reference_name!r} and {name!r} differ in {field_name}"
                )


@dataclass(frozen=True)
class ProbeCandidate:
    name: str
    config: Mapping[str, object]
    config_hash: str
    validation_scores: tuple[float, ...]
    per_seed_validation_scores: Mapping[int, tuple[float, ...]]
    n_iter_by_seed: Mapping[int, int]
    fitted_estimators: Mapping[int, tuple[Any, Any]] = field(repr=False, compare=False)


@dataclass(frozen=True)
class ProbeSearch:
    archive_name: str
    dataset_id: str
    validation_ids: tuple[str, ...]
    validation_labels: tuple[int, ...]
    candidates: Mapping[str, ProbeCandidate]

    def probability_ensemble_candidate_score_set(self) -> CandidateScoreSet:
        """Secondary diagnostic scores; never use for primary config selection."""

        return CandidateScoreSet(
            partition=Partition.VALIDATION,
            dataset_id=self.dataset_id,
            sample_ids=self.validation_ids,
            labels=self.validation_labels,
            candidates={
                name: candidate.validation_scores
                for name, candidate in self.candidates.items()
            },
            candidate_config_hashes={
                name: candidate.config_hash for name, candidate in self.candidates.items()
            },
        )

    def seeded_candidate_score_set(self, seeds: Sequence[int]) -> "SeededCandidateScoreSet":
        return SeededCandidateScoreSet(
            partition=Partition.VALIDATION,
            dataset_id=self.dataset_id,
            sample_ids=self.validation_ids,
            labels=self.validation_labels,
            seeds=tuple(int(seed) for seed in seeds),
            candidates={
                name: candidate.per_seed_validation_scores
                for name, candidate in self.candidates.items()
            },
            candidate_config_hashes={
                name: candidate.config_hash for name, candidate in self.candidates.items()
            },
        )


@dataclass(frozen=True)
class SeededCandidateScoreSet:
    """Candidate predictions indexed by independent training seed."""

    partition: Partition
    dataset_id: str
    sample_ids: tuple[str, ...]
    labels: tuple[int, ...]
    seeds: tuple[int, ...]
    candidates: Mapping[str, Mapping[int, tuple[float, ...]]]
    candidate_config_hashes: Mapping[str, str]
    dataset_manifest_hash: str = field(init=False)
    labeled_manifest_hash: str = field(init=False)
    candidate_set_digest: str = field(init=False)

    def __post_init__(self) -> None:
        partition = Partition(self.partition)
        sample_ids = tuple(str(value) for value in self.sample_ids)
        labels = tuple(int(value) for value in self.labels)
        seeds = tuple(int(value) for value in self.seeds)
        if not sample_ids or len(sample_ids) != len(labels):
            raise ValueError("sample_ids and labels must have equal non-zero length")
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("sample_ids must be unique")
        if set(labels) != {0, 1}:
            raise ValueError("seeded score sets must contain both binary labels")
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be non-empty and unique")
        configs = {
            str(name): str(digest)
            for name, digest in self.candidate_config_hashes.items()
        }
        normalized: dict[str, Mapping[int, tuple[float, ...]]] = {}
        if not self.candidates or set(self.candidates) != set(configs):
            raise ValueError("seeded candidates/config hashes must share non-empty names")
        for name, by_seed in self.candidates.items():
            if set(by_seed) != set(seeds):
                raise ValueError(f"candidate {name!r} does not contain the frozen seed set")
            normalized_seed_scores: dict[int, tuple[float, ...]] = {}
            for seed in seeds:
                values = tuple(float(value) for value in by_seed[seed])
                if len(values) != len(labels):
                    raise ValueError(f"candidate {name!r}, seed {seed} has wrong length")
                if any(not math.isfinite(value) for value in values):
                    raise ValueError(f"candidate {name!r}, seed {seed} is non-finite")
                normalized_seed_scores[seed] = values
            _require_sha256(configs[str(name)], f"config hash for {name!r}")
            normalized[str(name)] = MappingProxyType(normalized_seed_scores)

        anchor = CandidateScoreSet(
            partition=partition,
            dataset_id=self.dataset_id,
            sample_ids=sample_ids,
            labels=labels,
            candidates={name: values[seeds[0]] for name, values in normalized.items()},
            candidate_config_hashes=configs,
        )
        object.__setattr__(self, "partition", partition)
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "candidates", MappingProxyType(normalized))
        object.__setattr__(self, "candidate_config_hashes", MappingProxyType(configs))
        object.__setattr__(self, "dataset_manifest_hash", anchor.dataset_manifest_hash)
        object.__setattr__(self, "labeled_manifest_hash", anchor.labeled_manifest_hash)
        object.__setattr__(self, "candidate_set_digest", anchor.candidate_set_digest)


def seed_mean_auroc(
    labels: Sequence[int],
    by_seed: Mapping[int, Sequence[float]],
    seeds: Sequence[int],
) -> tuple[dict[int, float], float, float]:
    """Evaluate each independent seed, then report mean and sample SD."""

    ordered = tuple(int(seed) for seed in seeds)
    if not ordered or len(set(ordered)) != len(ordered) or set(by_seed) != set(ordered):
        raise ValueError("scores must exactly match the ordered non-empty seed set")
    values = {seed: binary_auroc(labels, by_seed[seed]) for seed in ordered}
    mean = sum(values.values()) / len(values)
    if len(values) == 1:
        sample_std = 0.0
    else:
        sample_std = math.sqrt(
            sum((value - mean) ** 2 for value in values.values()) / (len(values) - 1)
        )
    return values, mean, sample_std


def seed_mean_auroc_implementation_hash() -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "metric": "binary_auroc",
                "metric_implementation_hash": function_implementation_hash(binary_auroc),
                "aggregation": "arithmetic_mean_across_ordered_independent_seeds",
                "aggregation_implementation_hash": function_implementation_hash(seed_mean_auroc),
            }
        )
    ).hexdigest()


_SEED_MEAN_CHOICE_SEAL = object()


@dataclass(frozen=True, init=False)
class FrozenSeedMeanChoice:
    candidate_name: str
    validation_score: float
    validation_seed_auroc: tuple[tuple[int, float], ...]
    seeds: tuple[int, ...]
    dataset_id: str
    validation_manifest_hash: str
    expected_test_manifest_hash: str
    candidate_configs: tuple[tuple[str, str], ...]
    candidate_set_digest: str
    candidate_metrics_digest: str
    metric_id: str
    metric_implementation_hash: str
    protocol_digest: str
    selected_on: Partition

    def __init__(
        self,
        *,
        _seal: object | None = None,
        candidate_name: str = "",
        validation_score: float = math.nan,
        validation_seed_auroc: tuple[tuple[int, float], ...] = (),
        seeds: tuple[int, ...] = (),
        dataset_id: str = "",
        validation_manifest_hash: str = "",
        expected_test_manifest_hash: str = "",
        candidate_configs: tuple[tuple[str, str], ...] = (),
        candidate_set_digest: str = "",
        candidate_metrics_digest: str = "",
        metric_implementation_hash: str = "",
        protocol_digest: str = "",
    ) -> None:
        if _seal is not _SEED_MEAN_CHOICE_SEAL:
            raise ExperimentProtocolError(
                "FrozenSeedMeanChoice can only be created by validation selection"
            )
        object.__setattr__(self, "candidate_name", candidate_name)
        object.__setattr__(self, "validation_score", float(validation_score))
        object.__setattr__(self, "validation_seed_auroc", validation_seed_auroc)
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "validation_manifest_hash", validation_manifest_hash)
        object.__setattr__(self, "expected_test_manifest_hash", expected_test_manifest_hash)
        object.__setattr__(self, "candidate_configs", candidate_configs)
        object.__setattr__(self, "candidate_set_digest", candidate_set_digest)
        object.__setattr__(self, "candidate_metrics_digest", candidate_metrics_digest)
        object.__setattr__(self, "metric_id", "mean_of_seed_binary_auroc")
        object.__setattr__(self, "metric_implementation_hash", metric_implementation_hash)
        object.__setattr__(self, "protocol_digest", protocol_digest)
        object.__setattr__(self, "selected_on", Partition.VALIDATION)


def _seed_candidate_metrics(
    scores: SeededCandidateScoreSet,
) -> tuple[dict[str, dict[str, object]], str]:
    summaries: dict[str, dict[str, object]] = {}
    for name in sorted(scores.candidates):
        by_seed, mean, sample_std = seed_mean_auroc(
            scores.labels,
            scores.candidates[name],
            scores.seeds,
        )
        summaries[name] = {
            "per_seed_auroc_hex": [
                [seed, float(by_seed[seed]).hex()] for seed in scores.seeds
            ],
            "mean_hex": float(mean).hex(),
            "sample_std_hex": float(sample_std).hex(),
        }
    digest = hashlib.sha256(
        _canonical_json(
            {
                "schema": "scinumbench-v2-seed-candidate-metrics/1",
                "seeds": list(scores.seeds),
                "candidates": summaries,
            }
        )
    ).hexdigest()
    return summaries, digest


def _seed_mean_choice_digest(
    *,
    candidate_name: str,
    validation_score: float,
    validation_seed_auroc: Sequence[tuple[int, float]],
    seeds: Sequence[int],
    dataset_id: str,
    validation_manifest_hash: str,
    expected_test_manifest_hash: str,
    candidate_configs: Sequence[tuple[str, str]],
    candidate_set_digest: str,
    candidate_metrics_digest: str,
    metric_implementation_hash: str,
) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "protocol_version": EXPERIMENT_PROTOCOL_VERSION,
                "kind": "validation-seed-mean-auroc-selection",
                "candidate_name": candidate_name,
                "validation_score_hex": float(validation_score).hex(),
                "validation_seed_auroc_hex": [
                    [int(seed), float(value).hex()]
                    for seed, value in validation_seed_auroc
                ],
                "seeds": [int(seed) for seed in seeds],
                "dataset_id": dataset_id,
                "validation_manifest_hash": validation_manifest_hash,
                "expected_test_manifest_hash": expected_test_manifest_hash,
                "candidate_configs": list(candidate_configs),
                "candidate_set_digest": candidate_set_digest,
                "candidate_metrics_digest": candidate_metrics_digest,
                "metric_id": "mean_of_seed_binary_auroc",
                "metric_implementation_hash": metric_implementation_hash,
            }
        )
    ).hexdigest()


def select_seed_mean_auroc_on_validation(
    validation: SeededCandidateScoreSet,
    *,
    expected_test_manifest_hash: str,
) -> FrozenSeedMeanChoice:
    if validation.partition is not Partition.VALIDATION:
        raise ExperimentProtocolError("seed-mean candidate selection requires validation")
    _require_sha256(expected_test_manifest_hash, "expected_test_manifest_hash")
    summaries, metrics_digest = _seed_candidate_metrics(validation)
    best_name: str | None = None
    best_score = -math.inf
    for name in sorted(summaries):
        score = float.fromhex(str(summaries[name]["mean_hex"]))
        if score > best_score:
            best_name, best_score = name, score
    assert best_name is not None
    seed_values = tuple(
        (seed, binary_auroc(validation.labels, validation.candidates[best_name][seed]))
        for seed in validation.seeds
    )
    configs = tuple(sorted(validation.candidate_config_hashes.items()))
    metric_hash = seed_mean_auroc_implementation_hash()
    digest = _seed_mean_choice_digest(
        candidate_name=best_name,
        validation_score=best_score,
        validation_seed_auroc=seed_values,
        seeds=validation.seeds,
        dataset_id=validation.dataset_id,
        validation_manifest_hash=validation.labeled_manifest_hash,
        expected_test_manifest_hash=expected_test_manifest_hash,
        candidate_configs=configs,
        candidate_set_digest=validation.candidate_set_digest,
        candidate_metrics_digest=metrics_digest,
        metric_implementation_hash=metric_hash,
    )
    return FrozenSeedMeanChoice(
        _seal=_SEED_MEAN_CHOICE_SEAL,
        candidate_name=best_name,
        validation_score=best_score,
        validation_seed_auroc=seed_values,
        seeds=validation.seeds,
        dataset_id=validation.dataset_id,
        validation_manifest_hash=validation.labeled_manifest_hash,
        expected_test_manifest_hash=expected_test_manifest_hash,
        candidate_configs=configs,
        candidate_set_digest=validation.candidate_set_digest,
        candidate_metrics_digest=metrics_digest,
        metric_implementation_hash=metric_hash,
        protocol_digest=digest,
    )


@dataclass(frozen=True)
class SeedMeanEvaluation:
    candidate_name: str
    per_seed_auroc: Mapping[int, float]
    mean_auroc: float
    sample_std_auroc: float
    metric_id: str
    metric_implementation_hash: str
    test_manifest_hash: str
    test_labeled_manifest_hash: str
    protocol_digest: str


def evaluate_seed_mean_choice_on_test(
    choice: FrozenSeedMeanChoice,
    test: SeededCandidateScoreSet,
) -> SeedMeanEvaluation:
    if not isinstance(choice, FrozenSeedMeanChoice):
        raise ExperimentProtocolError("choice is not a sealed seed-mean validation receipt")
    if test.partition is not Partition.TEST:
        raise ExperimentProtocolError("seed-mean final evaluation requires test scores")
    if test.dataset_id != choice.dataset_id:
        raise ExperimentProtocolError("test dataset differs from seed-mean selection")
    if not hmac.compare_digest(test.dataset_manifest_hash, choice.expected_test_manifest_hash):
        raise ExperimentProtocolError("test manifest differs from seed-mean precommitment")
    if test.seeds != choice.seeds:
        raise ExperimentProtocolError("test seed set differs from seed-mean selection")
    configs = tuple(sorted(test.candidate_config_hashes.items()))
    if configs != choice.candidate_configs or test.candidate_set_digest != choice.candidate_set_digest:
        raise ExperimentProtocolError("test candidate configurations differ from validation")
    expected_digest = _seed_mean_choice_digest(
        candidate_name=choice.candidate_name,
        validation_score=choice.validation_score,
        validation_seed_auroc=choice.validation_seed_auroc,
        seeds=choice.seeds,
        dataset_id=choice.dataset_id,
        validation_manifest_hash=choice.validation_manifest_hash,
        expected_test_manifest_hash=choice.expected_test_manifest_hash,
        candidate_configs=choice.candidate_configs,
        candidate_set_digest=choice.candidate_set_digest,
        candidate_metrics_digest=choice.candidate_metrics_digest,
        metric_implementation_hash=choice.metric_implementation_hash,
    )
    if not hmac.compare_digest(expected_digest, choice.protocol_digest):
        raise ExperimentProtocolError("seed-mean selection receipt digest mismatch")
    if not hmac.compare_digest(
        choice.metric_implementation_hash,
        seed_mean_auroc_implementation_hash(),
    ):
        raise ExperimentProtocolError("seed-mean metric implementation mismatch")
    per_seed, mean, sample_std = seed_mean_auroc(
        test.labels,
        test.candidates[choice.candidate_name],
        choice.seeds,
    )
    return SeedMeanEvaluation(
        candidate_name=choice.candidate_name,
        per_seed_auroc=MappingProxyType(per_seed),
        mean_auroc=mean,
        sample_std_auroc=sample_std,
        metric_id=choice.metric_id,
        metric_implementation_hash=choice.metric_implementation_hash,
        test_manifest_hash=test.dataset_manifest_hash,
        test_labeled_manifest_hash=test.labeled_manifest_hash,
        protocol_digest=choice.protocol_digest,
    )


def fit_probe_candidates(
    archive: FeatureArchive,
    *,
    archive_name: str,
    dataset_id: str,
    c_values: Sequence[float],
    seeds: Sequence[int],
    max_iter: int = 2000,
) -> ProbeSearch:
    """Fit every probe on train and emit validation predictions only."""

    import numpy as np
    import warnings
    import sklearn
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if not archive_name or not dataset_id:
        raise ValueError("archive_name and dataset_id must be non-empty")
    c_grid = tuple(float(value) for value in c_values)
    seed_grid = tuple(int(value) for value in seeds)
    if not c_grid or any(not math.isfinite(value) or value <= 0 for value in c_grid):
        raise ValueError("all C values must be finite and positive")
    if len(set(c_grid)) != len(c_grid):
        raise ValueError("C values must be unique")
    if not seed_grid or len(set(seed_grid)) != len(seed_grid):
        raise ValueError("seeds must be non-empty and unique")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")

    train_indices = np.asarray(archive.indices(Partition.TRAIN), dtype=np.int64)
    validation_indices = np.asarray(archive.indices(Partition.VALIDATION), dtype=np.int64)
    y = np.asarray(archive.labels, dtype=np.int8)
    validation_ids = tuple(archive.example_ids[index] for index in validation_indices)
    validation_labels = tuple(int(y[index]) for index in validation_indices)
    candidates: dict[str, ProbeCandidate] = {}

    for feature_key in archive.feature_keys:
        match = _FEATURE_KEY.fullmatch(feature_key)
        assert match is not None
        pooling, layer_text = match.groups()
        matrix = archive.features[feature_key]
        x_train = matrix[train_indices]
        x_validation = matrix[validation_indices]
        y_train = y[train_indices]
        for c_value in c_grid:
            name = f"{archive_name}:{feature_key}:C={c_value.hex()}"
            per_seed: dict[int, tuple[float, ...]] = {}
            estimators: dict[int, tuple[Any, Any]] = {}
            n_iter_by_seed: dict[int, int] = {}
            for seed in seed_grid:
                scaler = StandardScaler(copy=True).fit(x_train)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    classifier = LogisticRegression(
                        C=c_value,
                        solver="liblinear",
                        max_iter=max_iter,
                        random_state=seed,
                    ).fit(scaler.transform(x_train), y_train)
                iterations = int(np.max(classifier.n_iter_))
                if any(issubclass(item.category, ConvergenceWarning) for item in caught):
                    raise ExperimentProtocolError(
                        f"probe candidate {name!r}, seed {seed} emitted ConvergenceWarning"
                    )
                if iterations >= max_iter:
                    raise ExperimentProtocolError(
                        f"probe candidate {name!r}, seed {seed} reached max_iter={max_iter}"
                    )
                scores = classifier.predict_proba(scaler.transform(x_validation))[:, 1]
                per_seed[seed] = tuple(float(value) for value in scores)
                estimators[seed] = (scaler, classifier)
                n_iter_by_seed[seed] = iterations
            stacked = np.asarray([per_seed[seed] for seed in seed_grid], dtype=np.float64)
            mean_scores = tuple(float(value) for value in stacked.mean(axis=0))
            config: dict[str, object] = {
                "schema": "scinumbench-v2-linear-probe/1",
                "archive_name": archive_name,
                "archive_sha256": archive.archive_sha256,
                "extraction_metadata_sha256": archive.extraction_metadata_sha256,
                "model_id": archive.model_id,
                "model_revision": archive.model_revision,
                "model_weight_manifest_sha256": archive.extraction_metadata[
                    "model_weight_manifest_sha256"
                ],
                "dataset_manifest_sha256": archive.dataset_manifest_sha256,
                "feature_key": feature_key,
                "pooling": pooling,
                "layer": int(layer_text),
                "C": c_value,
                "seeds": list(seed_grid),
                "standardizer": "train-only StandardScaler",
                "classifier": {
                    "type": "LogisticRegression",
                    "penalty": "l2",
                    "solver": "liblinear",
                    "max_iter": max_iter,
                    "class_weight": None,
                },
                "numpy_version": np.__version__,
                "scikit_learn_version": sklearn.__version__,
                "primary_seed_aggregation": "mean of independently evaluated seed metrics",
                "secondary_probability_aggregation": "arithmetic mean of probabilities",
                "validation_selection_estimand": "mean of per-seed AUROC",
            }
            config_hash = candidate_config_hash(config)
            candidates[name] = ProbeCandidate(
                name=name,
                config=MappingProxyType(config),
                config_hash=config_hash,
                validation_scores=mean_scores,
                per_seed_validation_scores=MappingProxyType(per_seed),
                n_iter_by_seed=MappingProxyType(n_iter_by_seed),
                fitted_estimators=MappingProxyType(estimators),
            )

    return ProbeSearch(
        archive_name=archive_name,
        dataset_id=dataset_id,
        validation_ids=validation_ids,
        validation_labels=validation_labels,
        candidates=MappingProxyType(candidates),
    )


@dataclass(frozen=True)
class ProbeTestPredictions:
    candidate_name: str
    config_hash: str
    mean_scores: tuple[float, ...]
    per_seed_scores: Mapping[int, tuple[float, ...]]


def predict_probe_on_test(
    search: ProbeSearch,
    archive: FeatureArchive,
    test_partition: UnlockedTestPartition,
    candidate_name: str,
) -> ProbeTestPredictions:
    """Score test only after the global ledger has unlocked this archive."""

    import numpy as np

    try:
        candidate = search.candidates[candidate_name]
    except KeyError as exc:
        raise ExperimentProtocolError("candidate was not present in validation search") from exc
    if (
        test_partition.archive_sha256 != archive.archive_sha256
        or test_partition.dataset_manifest_sha256 != archive.dataset_manifest_sha256
    ):
        raise ExperimentProtocolError("test grant/partition is bound to another feature archive")
    feature_key = str(candidate.config["feature_key"])
    test_indices = np.asarray(test_partition.indices, dtype=np.int64)
    x_test = archive.features[feature_key][test_indices]
    per_seed: dict[int, tuple[float, ...]] = {}
    for seed, (scaler, classifier) in candidate.fitted_estimators.items():
        scores = classifier.predict_proba(scaler.transform(x_test))[:, 1]
        per_seed[seed] = tuple(float(value) for value in scores)
    stacked = np.asarray(list(per_seed.values()), dtype=np.float64)
    return ProbeTestPredictions(
        candidate_name=candidate_name,
        config_hash=candidate.config_hash,
        mean_scores=tuple(float(value) for value in stacked.mean(axis=0)),
        per_seed_scores=MappingProxyType(per_seed),
    )


@dataclass(frozen=True)
class FusionCandidates:
    scores: Mapping[str, tuple[float, ...]]
    config_hashes: Mapping[str, str]
    members: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class SeededFusionCandidates:
    per_seed_scores: Mapping[str, Mapping[int, tuple[float, ...]]]
    probability_ensemble_scores: Mapping[str, tuple[float, ...]]
    config_hashes: Mapping[str, str]
    members: Mapping[str, tuple[str, ...]]
    seeds: tuple[int, ...]


def build_probability_fusions(
    base_scores: Mapping[str, Sequence[float]],
    base_config_hashes: Mapping[str, str],
) -> FusionCandidates:
    """Enumerate predeclared equal-weight probability fusions deterministically."""

    if not base_scores or set(base_scores) != set(base_config_hashes):
        raise ValueError("base scores and config hashes must share non-empty names")
    names = tuple(sorted(base_scores))
    if len(names) > 10:
        raise ValueError("refuse exponential fusion search over more than 10 base models")
    lengths = {len(base_scores[name]) for name in names}
    if len(lengths) != 1 or next(iter(lengths)) == 0:
        raise ValueError("base score vectors must share a non-zero length")
    for name in names:
        _require_sha256(str(base_config_hashes[name]), f"config hash for {name}")
        if any(not math.isfinite(float(value)) for value in base_scores[name]):
            raise ValueError(f"base scores for {name!r} contain non-finite values")

    scores: dict[str, tuple[float, ...]] = {}
    configs: dict[str, str] = {}
    members: dict[str, tuple[str, ...]] = {}
    for size in range(1, len(names) + 1):
        for combination in itertools.combinations(names, size):
            name = "single:" + combination[0] if size == 1 else "mean:" + "+".join(combination)
            fused = tuple(
                sum(float(base_scores[item][index]) for item in combination) / size
                for index in range(next(iter(lengths)))
            )
            config = {
                "schema": "scinumbench-v2-probability-fusion/1",
                "operation": "identity" if size == 1 else "equal_probability_mean",
                "members": [
                    {"name": item, "config_hash": str(base_config_hashes[item])}
                    for item in combination
                ],
                "weights": [1.0 / size] * size,
            }
            scores[name] = fused
            configs[name] = candidate_config_hash(config)
            members[name] = combination
    return FusionCandidates(
        scores=MappingProxyType(scores),
        config_hashes=MappingProxyType(configs),
        members=MappingProxyType(members),
    )


def build_seeded_probability_fusions(
    base_scores: Mapping[str, Mapping[int, Sequence[float]]],
    base_config_hashes: Mapping[str, str],
    *,
    seeds: Sequence[int],
) -> SeededFusionCandidates:
    """Fuse models within each seed; keep cross-seed probability mean secondary."""

    ordered_seeds = tuple(int(seed) for seed in seeds)
    if not ordered_seeds or len(set(ordered_seeds)) != len(ordered_seeds):
        raise ValueError("seeds must be non-empty and unique")
    if not base_scores or set(base_scores) != set(base_config_hashes):
        raise ValueError("base scores and config hashes must share non-empty names")
    for name, by_seed in base_scores.items():
        if set(by_seed) != set(ordered_seeds):
            raise ValueError(f"base model {name!r} does not contain the frozen seeds")

    by_seed_fusions: dict[int, FusionCandidates] = {}
    for seed in ordered_seeds:
        by_seed_fusions[seed] = build_probability_fusions(
            {name: values[seed] for name, values in base_scores.items()},
            base_config_hashes,
        )
    reference = by_seed_fusions[ordered_seeds[0]]
    for seed in ordered_seeds[1:]:
        current = by_seed_fusions[seed]
        if current.config_hashes != reference.config_hashes or current.members != reference.members:
            raise ExperimentProtocolError("seeded fusion configuration drift")

    seeded: dict[str, Mapping[int, tuple[float, ...]]] = {}
    ensembles: dict[str, tuple[float, ...]] = {}
    for name in sorted(reference.scores):
        seed_map = {
            seed: by_seed_fusions[seed].scores[name]
            for seed in ordered_seeds
        }
        length = len(seed_map[ordered_seeds[0]])
        ensembles[name] = tuple(
            sum(seed_map[seed][index] for seed in ordered_seeds) / len(ordered_seeds)
            for index in range(length)
        )
        seeded[name] = MappingProxyType(seed_map)
    return SeededFusionCandidates(
        per_seed_scores=MappingProxyType(seeded),
        probability_ensemble_scores=MappingProxyType(ensembles),
        config_hashes=reference.config_hashes,
        members=reference.members,
        seeds=ordered_seeds,
    )


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Average precision with threshold-grouped ties (sklearn-compatible form)."""

    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must have equal non-zero length")
    pairs = [(float(score), int(label)) for label, score in zip(labels, scores)]
    if any(label not in (0, 1) for _, label in pairs):
        raise ValueError("labels must be binary")
    if any(not math.isfinite(score) for score, _ in pairs):
        raise ValueError("scores must be finite")
    positives = sum(label for _, label in pairs)
    if positives == 0:
        raise ValueError("average precision requires a positive example")
    pairs.sort(key=lambda item: item[0], reverse=True)
    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    result = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        true_positive += sum(label for _, label in pairs[index:end])
        false_positive += (end - index) - sum(label for _, label in pairs[index:end])
        recall = true_positive / positives
        precision = true_positive / (true_positive + false_positive)
        result += (recall - previous_recall) * precision
        previous_recall = recall
        index = end
    return result


def binary_f1(labels: Sequence[int], scores: Sequence[float], threshold: float) -> float:
    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must have equal non-zero length")
    threshold = float(threshold)
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    tp = fp = fn = 0
    for label, score in zip(labels, scores):
        if int(label) not in (0, 1) or not math.isfinite(float(score)):
            raise ValueError("labels must be binary and scores finite")
        predicted = float(score) >= threshold
        tp += int(predicted and int(label) == 1)
        fp += int(predicted and int(label) == 0)
        fn += int(not predicted and int(label) == 1)
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2 * tp / denominator


@dataclass(frozen=True)
class SeedMetricReport:
    per_seed: Mapping[int, Mapping[str, float]]
    seed_mean: Mapping[str, float]
    seed_sample_std: Mapping[str, float]
    probability_ensemble: Mapping[str, float]


def _mean_and_sample_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("at least one value is required")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def summarize_seed_metrics(
    labels: Sequence[int],
    per_seed_scores: Mapping[int, Sequence[float]],
    per_seed_thresholds: Mapping[int, float],
    *,
    probability_ensemble_scores: Sequence[float],
    probability_ensemble_threshold: float,
) -> SeedMetricReport:
    """Keep the 10-seed estimand distinct from a probability ensemble."""

    if not per_seed_scores or set(per_seed_scores) != set(per_seed_thresholds):
        raise ValueError("seed scores and thresholds must share non-empty seed keys")
    per_seed: dict[int, Mapping[str, float]] = {}
    for seed in sorted(per_seed_scores):
        scores = per_seed_scores[seed]
        if len(scores) != len(labels):
            raise ValueError(f"seed {seed} score length differs from labels")
        per_seed[seed] = MappingProxyType(
            {
                "auroc": binary_auroc(labels, scores),
                "average_precision": average_precision(labels, scores),
                "f1": binary_f1(labels, scores, per_seed_thresholds[seed]),
                "validation_selected_threshold": float(per_seed_thresholds[seed]),
            }
        )
    metric_names = ("auroc", "average_precision", "f1")
    means: dict[str, float] = {}
    standard_deviations: dict[str, float] = {}
    for metric in metric_names:
        values = [float(per_seed[seed][metric]) for seed in sorted(per_seed)]
        means[metric], standard_deviations[metric] = _mean_and_sample_std(values)
    ensemble = {
        "auroc": binary_auroc(labels, probability_ensemble_scores),
        "average_precision": average_precision(labels, probability_ensemble_scores),
        "f1": binary_f1(labels, probability_ensemble_scores, probability_ensemble_threshold),
        "validation_selected_threshold": float(probability_ensemble_threshold),
    }
    return SeedMetricReport(
        per_seed=MappingProxyType(per_seed),
        seed_mean=MappingProxyType(means),
        seed_sample_std=MappingProxyType(standard_deviations),
        probability_ensemble=MappingProxyType(ensemble),
    )


_THRESHOLD_SEAL = object()


@dataclass(frozen=True, init=False)
class FrozenThresholdChoice:
    """F1 threshold receipt constructible only from validation labels."""

    candidate_name: str
    candidate_config_hash: str
    threshold: float
    validation_f1: float
    dataset_id: str
    validation_labeled_manifest_hash: str
    expected_test_manifest_hash: str
    metric_id: str
    metric_implementation_hash: str
    protocol_digest: str
    selected_on: Partition

    def __init__(
        self,
        *,
        _seal: object | None = None,
        candidate_name: str = "",
        candidate_config_hash: str = "",
        threshold: float = math.nan,
        validation_f1: float = math.nan,
        dataset_id: str = "",
        validation_labeled_manifest_hash: str = "",
        expected_test_manifest_hash: str = "",
        metric_id: str = "",
        metric_implementation_hash: str = "",
        protocol_digest: str = "",
    ) -> None:
        if _seal is not _THRESHOLD_SEAL:
            raise ExperimentProtocolError(
                "FrozenThresholdChoice can only be created by validation selection"
            )
        object.__setattr__(self, "candidate_name", candidate_name)
        object.__setattr__(self, "candidate_config_hash", candidate_config_hash)
        object.__setattr__(self, "threshold", float(threshold))
        object.__setattr__(self, "validation_f1", float(validation_f1))
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(
            self, "validation_labeled_manifest_hash", validation_labeled_manifest_hash
        )
        object.__setattr__(self, "expected_test_manifest_hash", expected_test_manifest_hash)
        object.__setattr__(self, "metric_id", metric_id)
        object.__setattr__(self, "metric_implementation_hash", metric_implementation_hash)
        object.__setattr__(self, "protocol_digest", protocol_digest)
        object.__setattr__(self, "selected_on", Partition.VALIDATION)


def _threshold_digest(
    *,
    candidate_name: str,
    candidate_config_hash: str,
    threshold: float,
    validation_f1: float,
    dataset_id: str,
    validation_labeled_manifest_hash: str,
    expected_test_manifest_hash: str,
    metric_id: str,
    metric_implementation_hash: str,
) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "protocol_version": EXPERIMENT_PROTOCOL_VERSION,
                "kind": "validation-f1-threshold",
                "candidate_name": candidate_name,
                "candidate_config_hash": candidate_config_hash,
                "threshold_hex": float(threshold).hex(),
                "validation_f1_hex": float(validation_f1).hex(),
                "dataset_id": dataset_id,
                "validation_labeled_manifest_hash": validation_labeled_manifest_hash,
                "expected_test_manifest_hash": expected_test_manifest_hash,
                "metric_id": metric_id,
                "metric_implementation_hash": metric_implementation_hash,
            }
        )
    ).hexdigest()


def select_f1_threshold_on_validation(
    validation: CandidateScoreSet,
    *,
    candidate_name: str,
    expected_test_manifest_hash: str,
) -> FrozenThresholdChoice:
    if validation.partition is not Partition.VALIDATION:
        raise ExperimentProtocolError("threshold selection requires validation scores")
    _require_sha256(expected_test_manifest_hash, "expected_test_manifest_hash")
    try:
        scores = validation.candidates[candidate_name]
        config_hash = validation.candidate_config_hashes[candidate_name]
    except KeyError as exc:
        raise ExperimentProtocolError("threshold candidate is absent from validation") from exc
    thresholds = sorted(set(float(score) for score in scores), reverse=True)
    best_threshold = thresholds[0]
    best_f1 = -1.0
    for threshold in thresholds:
        value = binary_f1(validation.labels, scores, threshold)
        # Higher threshold wins exact ties, yielding a deterministic conservative rule.
        if (value, threshold) > (best_f1, best_threshold):
            best_f1, best_threshold = value, threshold
    metric_id = "binary_f1_at_threshold"
    metric_hash = function_implementation_hash(binary_f1)
    digest = _threshold_digest(
        candidate_name=candidate_name,
        candidate_config_hash=config_hash,
        threshold=best_threshold,
        validation_f1=best_f1,
        dataset_id=validation.dataset_id,
        validation_labeled_manifest_hash=validation.labeled_manifest_hash,
        expected_test_manifest_hash=expected_test_manifest_hash,
        metric_id=metric_id,
        metric_implementation_hash=metric_hash,
    )
    return FrozenThresholdChoice(
        _seal=_THRESHOLD_SEAL,
        candidate_name=candidate_name,
        candidate_config_hash=config_hash,
        threshold=best_threshold,
        validation_f1=best_f1,
        dataset_id=validation.dataset_id,
        validation_labeled_manifest_hash=validation.labeled_manifest_hash,
        expected_test_manifest_hash=expected_test_manifest_hash,
        metric_id=metric_id,
        metric_implementation_hash=metric_hash,
        protocol_digest=digest,
    )


def evaluate_f1_with_frozen_threshold(
    choice: FrozenThresholdChoice,
    test: CandidateScoreSet,
) -> float:
    if not isinstance(choice, FrozenThresholdChoice):
        raise ExperimentProtocolError("threshold choice is not a sealed validation receipt")
    if test.partition is not Partition.TEST:
        raise ExperimentProtocolError("frozen threshold may be evaluated only on test")
    if test.dataset_id != choice.dataset_id:
        raise ExperimentProtocolError("test dataset differs from threshold selection dataset")
    if not hmac.compare_digest(test.dataset_manifest_hash, choice.expected_test_manifest_hash):
        raise ExperimentProtocolError("test manifest differs from threshold precommitment")
    if test.candidate_config_hashes.get(choice.candidate_name) != choice.candidate_config_hash:
        raise ExperimentProtocolError("test candidate configuration differs from validation")
    expected = _threshold_digest(
        candidate_name=choice.candidate_name,
        candidate_config_hash=choice.candidate_config_hash,
        threshold=choice.threshold,
        validation_f1=choice.validation_f1,
        dataset_id=choice.dataset_id,
        validation_labeled_manifest_hash=choice.validation_labeled_manifest_hash,
        expected_test_manifest_hash=choice.expected_test_manifest_hash,
        metric_id=choice.metric_id,
        metric_implementation_hash=choice.metric_implementation_hash,
    )
    if not hmac.compare_digest(expected, choice.protocol_digest):
        raise ExperimentProtocolError("threshold receipt digest mismatch")
    if (
        choice.metric_id != "binary_f1_at_threshold"
        or not hmac.compare_digest(
            choice.metric_implementation_hash,
            function_implementation_hash(binary_f1),
        )
    ):
        raise ExperimentProtocolError("threshold metric implementation mismatch")
    return binary_f1(
        test.labels,
        test.candidates[choice.candidate_name],
        choice.threshold,
    )


@dataclass(frozen=True)
class GroupedBootstrapResult:
    grouping: str
    group_count: int
    replicates: int
    confidence: float
    seed: int
    point: Mapping[str, float]
    lower: Mapping[str, float]
    upper: Mapping[str, float]


@dataclass(frozen=True)
class GroupedBootstrapDifference:
    grouping: str
    metric: str
    group_count: int
    replicates: int
    confidence: float
    seed: int
    point_difference: float
    lower: float
    upper: float


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def grouped_bootstrap_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    group_ids: Sequence[Hashable],
    *,
    threshold: float,
    grouping: str,
    replicates: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20261026,
) -> GroupedBootstrapResult:
    """Bootstrap whole sources or papers, preserving every within-group row."""

    if len(labels) != len(scores) or len(labels) != len(group_ids) or not labels:
        raise ValueError("labels, scores, and group_ids must have equal non-zero length")
    if replicates <= 0 or not (0.0 < confidence < 1.0):
        raise ValueError("replicates must be positive and confidence in (0,1)")
    groups: dict[Hashable, list[int]] = {}
    for index, group in enumerate(group_ids):
        try:
            hash(group)
        except TypeError as exc:
            raise TypeError("bootstrap group IDs must be hashable") from exc
        groups.setdefault(group, []).append(index)
    if len(groups) < 2:
        raise ValueError("grouped bootstrap requires at least two groups")
    for group, indices in groups.items():
        if {int(labels[index]) for index in indices} != {0, 1}:
            raise ExperimentProtocolError(
                f"{grouping} group {group!r} does not preserve both paired labels"
            )

    def metrics(indices: Sequence[int]) -> dict[str, float]:
        selected_labels = [int(labels[index]) for index in indices]
        selected_scores = [float(scores[index]) for index in indices]
        return {
            "auroc": binary_auroc(selected_labels, selected_scores),
            "average_precision": average_precision(selected_labels, selected_scores),
            "f1": binary_f1(selected_labels, selected_scores, threshold),
        }

    all_indices = list(range(len(labels)))
    point = metrics(all_indices)
    keys = sorted(groups, key=lambda value: (type(value).__name__, repr(value)))
    generator = random.Random(seed)
    distributions = {metric: [] for metric in point}
    for _ in range(replicates):
        sampled_indices: list[int] = []
        for _ in keys:
            sampled_indices.extend(groups[generator.choice(keys)])
        values = metrics(sampled_indices)
        for metric, value in values.items():
            distributions[metric].append(value)
    alpha = (1.0 - confidence) / 2.0
    return GroupedBootstrapResult(
        grouping=grouping,
        group_count=len(keys),
        replicates=replicates,
        confidence=confidence,
        seed=seed,
        point=MappingProxyType(point),
        lower=MappingProxyType(
            {metric: _percentile(values, alpha) for metric, values in distributions.items()}
        ),
        upper=MappingProxyType(
            {
                metric: _percentile(values, 1.0 - alpha)
                for metric, values in distributions.items()
            }
        ),
    )


def grouped_bootstrap_auroc_difference(
    labels: Sequence[int],
    left_scores: Sequence[float],
    right_scores: Sequence[float],
    group_ids: Sequence[Hashable],
    *,
    grouping: str,
    replicates: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20261026,
) -> GroupedBootstrapDifference:
    """Paired cluster bootstrap for ``AUROC(left) - AUROC(right)``."""

    if (
        len(labels) != len(left_scores)
        or len(labels) != len(right_scores)
        or len(labels) != len(group_ids)
        or not labels
    ):
        raise ValueError("labels, both scores, and group_ids must have equal non-zero length")
    if replicates <= 0 or not (0.0 < confidence < 1.0):
        raise ValueError("replicates must be positive and confidence in (0,1)")
    if any(
        not math.isfinite(float(left)) or not math.isfinite(float(right))
        for left, right in zip(left_scores, right_scores)
    ):
        raise ValueError("scores must be finite")
    groups: dict[Hashable, list[int]] = {}
    for index, group in enumerate(group_ids):
        try:
            hash(group)
        except TypeError as exc:
            raise TypeError("bootstrap group IDs must be hashable") from exc
        groups.setdefault(group, []).append(index)
    if len(groups) < 2:
        raise ValueError("grouped bootstrap requires at least two groups")
    for group, indices in groups.items():
        if {int(labels[index]) for index in indices} != {0, 1}:
            raise ExperimentProtocolError(
                f"{grouping} group {group!r} does not preserve both paired labels"
            )

    def difference(indices: Sequence[int]) -> float:
        selected_labels = [int(labels[index]) for index in indices]
        return binary_auroc(
            selected_labels,
            [float(left_scores[index]) for index in indices],
        ) - binary_auroc(
            selected_labels,
            [float(right_scores[index]) for index in indices],
        )

    keys = sorted(groups, key=lambda value: (type(value).__name__, repr(value)))
    generator = random.Random(seed)
    distribution: list[float] = []
    for _ in range(replicates):
        sampled_indices: list[int] = []
        for _ in keys:
            sampled_indices.extend(groups[generator.choice(keys)])
        distribution.append(difference(sampled_indices))
    alpha = (1.0 - confidence) / 2.0
    return GroupedBootstrapDifference(
        grouping=grouping,
        metric="auroc_difference_left_minus_right",
        group_count=len(keys),
        replicates=replicates,
        confidence=confidence,
        seed=seed,
        point_difference=difference(list(range(len(labels)))),
        lower=_percentile(distribution, alpha),
        upper=_percentile(distribution, 1.0 - alpha),
    )


def grouped_bootstrap_seed_mean_metrics(
    labels: Sequence[int],
    per_seed_scores: Mapping[int, Sequence[float]],
    per_seed_thresholds: Mapping[int, float],
    group_ids: Sequence[Hashable],
    *,
    seeds: Sequence[int],
    grouping: str,
    replicates: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20261026,
) -> GroupedBootstrapResult:
    """Cluster-bootstrap the primary mean-of-seed metric estimand."""

    ordered_seeds = tuple(int(value) for value in seeds)
    if (
        not ordered_seeds
        or len(set(ordered_seeds)) != len(ordered_seeds)
        or set(per_seed_scores) != set(ordered_seeds)
        or set(per_seed_thresholds) != set(ordered_seeds)
    ):
        raise ValueError("scores/thresholds must exactly match the ordered seed set")
    if len(labels) != len(group_ids) or not labels:
        raise ValueError("labels and group_ids must have equal non-zero length")
    if any(len(per_seed_scores[value]) != len(labels) for value in ordered_seeds):
        raise ValueError("every seed score vector must match labels")
    if replicates <= 0 or not (0.0 < confidence < 1.0):
        raise ValueError("replicates must be positive and confidence in (0,1)")
    groups: dict[Hashable, list[int]] = {}
    for index, group in enumerate(group_ids):
        try:
            hash(group)
        except TypeError as exc:
            raise TypeError("bootstrap group IDs must be hashable") from exc
        groups.setdefault(group, []).append(index)
    if len(groups) < 2:
        raise ValueError("grouped bootstrap requires at least two groups")
    for group, indices in groups.items():
        if {int(labels[index]) for index in indices} != {0, 1}:
            raise ExperimentProtocolError(
                f"{grouping} group {group!r} does not preserve both paired labels"
            )

    def metrics(indices: Sequence[int]) -> dict[str, float]:
        selected_labels = [int(labels[index]) for index in indices]
        seed_metrics: dict[str, list[float]] = {
            "auroc": [],
            "average_precision": [],
            "f1": [],
        }
        for training_seed in ordered_seeds:
            selected_scores = [
                float(per_seed_scores[training_seed][index]) for index in indices
            ]
            seed_metrics["auroc"].append(binary_auroc(selected_labels, selected_scores))
            seed_metrics["average_precision"].append(
                average_precision(selected_labels, selected_scores)
            )
            seed_metrics["f1"].append(
                binary_f1(
                    selected_labels,
                    selected_scores,
                    per_seed_thresholds[training_seed],
                )
            )
        return {
            metric: sum(values) / len(values)
            for metric, values in seed_metrics.items()
        }

    point = metrics(list(range(len(labels))))
    keys = sorted(groups, key=lambda value: (type(value).__name__, repr(value)))
    generator = random.Random(seed)
    distributions = {metric: [] for metric in point}
    for _ in range(replicates):
        sampled_indices: list[int] = []
        for _ in keys:
            sampled_indices.extend(groups[generator.choice(keys)])
        values = metrics(sampled_indices)
        for metric, value in values.items():
            distributions[metric].append(value)
    alpha = (1.0 - confidence) / 2.0
    return GroupedBootstrapResult(
        grouping=grouping,
        group_count=len(keys),
        replicates=replicates,
        confidence=confidence,
        seed=seed,
        point=MappingProxyType(point),
        lower=MappingProxyType(
            {metric: _percentile(values, alpha) for metric, values in distributions.items()}
        ),
        upper=MappingProxyType(
            {
                metric: _percentile(values, 1.0 - alpha)
                for metric, values in distributions.items()
            }
        ),
    )


def grouped_bootstrap_seed_mean_auroc_difference(
    labels: Sequence[int],
    left_per_seed_scores: Mapping[int, Sequence[float]],
    right_per_seed_scores: Mapping[int, Sequence[float]],
    group_ids: Sequence[Hashable],
    *,
    seeds: Sequence[int],
    grouping: str,
    replicates: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20261026,
) -> GroupedBootstrapDifference:
    """Paired cluster bootstrap for mean-seed AUROC(left)-AUROC(right)."""

    ordered_seeds = tuple(int(value) for value in seeds)
    if (
        not ordered_seeds
        or len(set(ordered_seeds)) != len(ordered_seeds)
        or set(left_per_seed_scores) != set(ordered_seeds)
        or set(right_per_seed_scores) != set(ordered_seeds)
    ):
        raise ValueError("left/right scores must exactly match the ordered seed set")
    if len(labels) != len(group_ids) or not labels:
        raise ValueError("labels and group_ids must have equal non-zero length")
    if any(
        len(left_per_seed_scores[value]) != len(labels)
        or len(right_per_seed_scores[value]) != len(labels)
        for value in ordered_seeds
    ):
        raise ValueError("every seed score vector must match labels")
    if replicates <= 0 or not (0.0 < confidence < 1.0):
        raise ValueError("replicates must be positive and confidence in (0,1)")
    groups: dict[Hashable, list[int]] = {}
    for index, group in enumerate(group_ids):
        try:
            hash(group)
        except TypeError as exc:
            raise TypeError("bootstrap group IDs must be hashable") from exc
        groups.setdefault(group, []).append(index)
    if len(groups) < 2:
        raise ValueError("grouped bootstrap requires at least two groups")
    for group, indices in groups.items():
        if {int(labels[index]) for index in indices} != {0, 1}:
            raise ExperimentProtocolError(
                f"{grouping} group {group!r} does not preserve both paired labels"
            )

    def difference(indices: Sequence[int]) -> float:
        selected_labels = [int(labels[index]) for index in indices]
        differences = []
        for training_seed in ordered_seeds:
            left = [
                float(left_per_seed_scores[training_seed][index]) for index in indices
            ]
            right = [
                float(right_per_seed_scores[training_seed][index]) for index in indices
            ]
            differences.append(
                binary_auroc(selected_labels, left) - binary_auroc(selected_labels, right)
            )
        return sum(differences) / len(differences)

    keys = sorted(groups, key=lambda value: (type(value).__name__, repr(value)))
    generator = random.Random(seed)
    distribution: list[float] = []
    for _ in range(replicates):
        sampled_indices: list[int] = []
        for _ in keys:
            sampled_indices.extend(groups[generator.choice(keys)])
        distribution.append(difference(sampled_indices))
    alpha = (1.0 - confidence) / 2.0
    return GroupedBootstrapDifference(
        grouping=grouping,
        metric="mean_seed_auroc_difference_left_minus_right",
        group_count=len(keys),
        replicates=replicates,
        confidence=confidence,
        seed=seed,
        point_difference=difference(list(range(len(labels)))),
        lower=_percentile(distribution, alpha),
        upper=_percentile(distribution, 1.0 - alpha),
    )


_LEDGER_BINDING_SCHEMA = "scinumbench-v2-ledger-binding/1"
_LEDGER_ROOT_MARKER_SCHEMA = "scinumbench-v2-ledger-namespace/1"
_LEDGER_DATASET_MARKER_SCHEMA = "scinumbench-v2-ledger-dataset/1"
_LEDGER_ROOT_MARKER = ".scinumbench-ledger-namespace.json"
_LEDGER_DATASET_MARKER = ".scinumbench-ledger-dataset.json"
_LEDGER_MARKER_MAX_BYTES = 16 * 1024


def _secure_directory_flags() -> int:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise ExperimentProtocolError(
            "secure test ledger requires O_DIRECTORY and O_NOFOLLOW support"
        )
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _canonical_ledger_root_input(ledger_root: str | os.PathLike[str]) -> Path:
    raw = os.fspath(ledger_root)
    if not raw:
        raise ExperimentProtocolError("test ledger root must be non-empty")
    absolute = Path(os.path.abspath(raw))
    if absolute == Path("/"):
        raise ExperimentProtocolError("test ledger root cannot be the filesystem root")
    resolved = Path(os.path.realpath(absolute))
    if absolute != resolved:
        raise ExperimentProtocolError(
            "test ledger root must be a canonical path with no symbolic-link component"
        )
    return absolute


def _open_absolute_directory_nofollow(path: Path, *, create: bool) -> int:
    """Open an absolute directory one component at a time without symlinks."""

    if not path.is_absolute() or path == Path("/"):
        raise ExperimentProtocolError("secure ledger directory must be a non-root absolute path")
    flags = _secure_directory_flags()
    descriptor = os.open("/", flags)
    try:
        for component in path.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, 0o700, dir_fd=descriptor)
                    os.fsync(descriptor)
                except FileExistsError:
                    # Another initializer may have created it. The no-follow
                    # open below determines whether the winner was safe.
                    pass
                child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_file_at(directory_fd: int, name: str) -> bytes:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_NOFOLLOW,
        dir_fd=directory_fd,
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat_module.S_ISREG(metadata.st_mode):
            raise ExperimentProtocolError(f"ledger entry {name!r} is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            payload = handle.read(_LEDGER_MARKER_MAX_BYTES + 1)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(payload) > _LEDGER_MARKER_MAX_BYTES:
        raise ExperimentProtocolError(f"ledger entry {name!r} is unexpectedly large")
    return payload


def _write_exclusive_file_at(
    directory_fd: int,
    name: str,
    payload: bytes,
    *,
    mode: int = 0o600,
) -> None:
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        mode,
        dir_fd=directory_fd,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.fsync(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _parse_ledger_marker(
    payload: bytes,
    *,
    schema: str,
    marker_name: str,
) -> dict[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExperimentProtocolError(f"ledger marker {marker_name!r} is invalid JSON") from exc
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ExperimentProtocolError(f"ledger marker {marker_name!r} has the wrong schema")
    if payload != _canonical_json(value) + b"\n":
        raise ExperimentProtocolError(f"ledger marker {marker_name!r} is not canonical")
    return value


def _read_ledger_marker(
    directory_fd: int,
    name: str,
    *,
    schema: str,
) -> tuple[dict[str, object], bytes]:
    payload = _read_file_at(directory_fd, name)
    return _parse_ledger_marker(payload, schema=schema, marker_name=name), payload


def _initialize_root_marker(root_fd: int) -> tuple[dict[str, object], bytes]:
    try:
        return _read_ledger_marker(
            root_fd,
            _LEDGER_ROOT_MARKER,
            schema=_LEDGER_ROOT_MARKER_SCHEMA,
        )
    except FileNotFoundError:
        entries = os.listdir(root_fd)
        if entries:
            # A concurrent initializer may have won between open and listdir.
            if entries == [_LEDGER_ROOT_MARKER]:
                return _read_ledger_marker(
                    root_fd,
                    _LEDGER_ROOT_MARKER,
                    schema=_LEDGER_ROOT_MARKER_SCHEMA,
                )
            raise ExperimentProtocolError(
                "unbound test ledger root must be empty before it is frozen"
            )
        marker = {
            "schema": _LEDGER_ROOT_MARKER_SCHEMA,
            "namespace_id": secrets.token_hex(32),
        }
        payload = _canonical_json(marker) + b"\n"
        try:
            _write_exclusive_file_at(root_fd, _LEDGER_ROOT_MARKER, payload)
            return marker, payload
        except FileExistsError:
            return _read_ledger_marker(
                root_fd,
                _LEDGER_ROOT_MARKER,
                schema=_LEDGER_ROOT_MARKER_SCHEMA,
            )


def _initialize_dataset_marker(
    dataset_fd: int,
    *,
    namespace_id: str,
    dataset_manifest_sha256: str,
) -> tuple[dict[str, object], bytes]:
    try:
        marker, payload = _read_ledger_marker(
            dataset_fd,
            _LEDGER_DATASET_MARKER,
            schema=_LEDGER_DATASET_MARKER_SCHEMA,
        )
    except FileNotFoundError:
        entries = os.listdir(dataset_fd)
        if entries:
            if entries == [_LEDGER_DATASET_MARKER]:
                marker, payload = _read_ledger_marker(
                    dataset_fd,
                    _LEDGER_DATASET_MARKER,
                    schema=_LEDGER_DATASET_MARKER_SCHEMA,
                )
            else:
                raise ExperimentProtocolError(
                    "unbound dataset ledger directory must be empty before it is frozen"
                )
        else:
            marker = {
                "schema": _LEDGER_DATASET_MARKER_SCHEMA,
                "namespace_id": namespace_id,
                "dataset_manifest_sha256": dataset_manifest_sha256,
            }
            payload = _canonical_json(marker) + b"\n"
            try:
                _write_exclusive_file_at(
                    dataset_fd,
                    _LEDGER_DATASET_MARKER,
                    payload,
                )
            except FileExistsError:
                marker, payload = _read_ledger_marker(
                    dataset_fd,
                    _LEDGER_DATASET_MARKER,
                    schema=_LEDGER_DATASET_MARKER_SCHEMA,
                )
    if marker.get("namespace_id") != namespace_id:
        raise ExperimentProtocolError("dataset ledger belongs to another namespace")
    if marker.get("dataset_manifest_sha256") != dataset_manifest_sha256:
        raise ExperimentProtocolError("dataset ledger marker has the wrong manifest digest")
    if set(marker) != {"schema", "namespace_id", "dataset_manifest_sha256"}:
        raise ExperimentProtocolError("dataset ledger marker has unexpected fields")
    return marker, payload


def _normalize_ledger_binding(binding: Mapping[str, object]) -> dict[str, object]:
    expected_fields = {
        "schema",
        "canonical_root",
        "namespace_id",
        "root_device",
        "root_inode",
        "root_marker_sha256",
        "dataset_manifest_sha256",
        "dataset_device",
        "dataset_inode",
        "dataset_marker_sha256",
    }
    if not isinstance(binding, Mapping) or set(binding) != expected_fields:
        raise ExperimentProtocolError("test ledger binding has unexpected fields")
    normalized = dict(binding)
    if normalized.get("schema") != _LEDGER_BINDING_SCHEMA:
        raise ExperimentProtocolError("test ledger binding has the wrong schema")
    canonical_root = normalized.get("canonical_root")
    if not isinstance(canonical_root, str) or not canonical_root.startswith("/"):
        raise ExperimentProtocolError("test ledger binding root must be absolute")
    if str(_canonical_ledger_root_input(canonical_root)) != canonical_root:
        raise ExperimentProtocolError("test ledger binding root is not canonical")
    namespace_id = normalized.get("namespace_id")
    if not isinstance(namespace_id, str) or re.fullmatch(r"[0-9a-f]{64}", namespace_id) is None:
        raise ExperimentProtocolError("test ledger namespace ID is invalid")
    manifest_sha = normalized.get("dataset_manifest_sha256")
    if not isinstance(manifest_sha, str):
        raise ExperimentProtocolError("test ledger manifest digest is invalid")
    _require_sha256(manifest_sha, "test ledger dataset_manifest_sha256")
    for field_name in ("root_marker_sha256", "dataset_marker_sha256"):
        digest = normalized.get(field_name)
        if not isinstance(digest, str):
            raise ExperimentProtocolError(f"{field_name} is invalid")
        _require_sha256(digest, field_name)
    for field_name in ("root_device", "root_inode", "dataset_device", "dataset_inode"):
        value = normalized.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ExperimentProtocolError(f"{field_name} must be a non-negative integer")
    return normalized


def prepare_test_ledger_binding(
    ledger_root: str | os.PathLike[str],
    *,
    dataset_manifest_sha256: str,
) -> dict[str, object]:
    """Create and freeze the stable filesystem identity of a test ledger.

    This function is called while producing the immutable run specification,
    before any test evaluation. Both the root and dataset directory are opened
    without following symbolic links; their device/inode identities and random
    namespace marker are then committed into the run specification.
    """

    _require_sha256(dataset_manifest_sha256, "dataset_manifest_sha256")
    canonical_root = _canonical_ledger_root_input(ledger_root)
    try:
        root_fd = _open_absolute_directory_nofollow(canonical_root, create=True)
    except OSError as exc:
        raise ExperimentProtocolError("cannot securely open test ledger root") from exc
    dataset_fd = -1
    try:
        root_stat = os.fstat(root_fd)
        if not stat_module.S_ISDIR(root_stat.st_mode):
            raise ExperimentProtocolError("test ledger root is not a directory")
        root_marker, root_payload = _initialize_root_marker(root_fd)
        if set(root_marker) != {"schema", "namespace_id"}:
            raise ExperimentProtocolError("test ledger namespace marker has unexpected fields")
        namespace_id = root_marker.get("namespace_id")
        if not isinstance(namespace_id, str) or re.fullmatch(r"[0-9a-f]{64}", namespace_id) is None:
            raise ExperimentProtocolError("test ledger namespace marker is invalid")
        try:
            os.mkdir(dataset_manifest_sha256, 0o700, dir_fd=root_fd)
            os.fsync(root_fd)
        except FileExistsError:
            pass
        try:
            dataset_fd = os.open(
                dataset_manifest_sha256,
                _secure_directory_flags(),
                dir_fd=root_fd,
            )
        except OSError as exc:
            raise ExperimentProtocolError(
                "dataset ledger entry is not a secure ordinary directory"
            ) from exc
        dataset_stat = os.fstat(dataset_fd)
        if not stat_module.S_ISDIR(dataset_stat.st_mode):
            raise ExperimentProtocolError("dataset ledger entry is not a directory")
        _, dataset_payload = _initialize_dataset_marker(
            dataset_fd,
            namespace_id=namespace_id,
            dataset_manifest_sha256=dataset_manifest_sha256,
        )
        binding = {
            "schema": _LEDGER_BINDING_SCHEMA,
            "canonical_root": str(canonical_root),
            "namespace_id": namespace_id,
            "root_device": root_stat.st_dev,
            "root_inode": root_stat.st_ino,
            "root_marker_sha256": hashlib.sha256(root_payload).hexdigest(),
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "dataset_device": dataset_stat.st_dev,
            "dataset_inode": dataset_stat.st_ino,
            "dataset_marker_sha256": hashlib.sha256(dataset_payload).hexdigest(),
        }
        return _normalize_ledger_binding(binding)
    finally:
        if dataset_fd >= 0:
            os.close(dataset_fd)
        os.close(root_fd)


def _open_bound_ledger_directories(
    ledger_root: str | os.PathLike[str],
    binding: Mapping[str, object],
    *,
    dataset_manifest_sha256: str,
) -> tuple[int, int, dict[str, object]]:
    normalized = _normalize_ledger_binding(binding)
    canonical_root = _canonical_ledger_root_input(ledger_root)
    if str(canonical_root) != normalized["canonical_root"]:
        raise ExperimentProtocolError("test ledger root differs from the frozen binding")
    if dataset_manifest_sha256 != normalized["dataset_manifest_sha256"]:
        raise ExperimentProtocolError("test ledger binding belongs to another dataset")
    try:
        root_fd = _open_absolute_directory_nofollow(canonical_root, create=False)
    except OSError as exc:
        raise ExperimentProtocolError("cannot securely reopen bound test ledger root") from exc
    dataset_fd = -1
    try:
        root_stat = os.fstat(root_fd)
        if (
            root_stat.st_dev != normalized["root_device"]
            or root_stat.st_ino != normalized["root_inode"]
        ):
            raise ExperimentProtocolError("test ledger root identity changed after freezing")
        root_marker, root_payload = _read_ledger_marker(
            root_fd,
            _LEDGER_ROOT_MARKER,
            schema=_LEDGER_ROOT_MARKER_SCHEMA,
        )
        if root_marker.get("namespace_id") != normalized["namespace_id"]:
            raise ExperimentProtocolError("test ledger namespace marker changed after freezing")
        if not hmac.compare_digest(
            hashlib.sha256(root_payload).hexdigest(),
            str(normalized["root_marker_sha256"]),
        ):
            raise ExperimentProtocolError("test ledger namespace marker was modified")
        try:
            dataset_fd = os.open(
                dataset_manifest_sha256,
                _secure_directory_flags(),
                dir_fd=root_fd,
            )
        except OSError as exc:
            raise ExperimentProtocolError(
                "cannot securely reopen bound dataset ledger directory"
            ) from exc
        dataset_stat = os.fstat(dataset_fd)
        if (
            dataset_stat.st_dev != normalized["dataset_device"]
            or dataset_stat.st_ino != normalized["dataset_inode"]
        ):
            raise ExperimentProtocolError(
                "dataset ledger directory identity changed after freezing"
            )
        dataset_marker, dataset_payload = _read_ledger_marker(
            dataset_fd,
            _LEDGER_DATASET_MARKER,
            schema=_LEDGER_DATASET_MARKER_SCHEMA,
        )
        if (
            dataset_marker.get("namespace_id") != normalized["namespace_id"]
            or dataset_marker.get("dataset_manifest_sha256")
            != dataset_manifest_sha256
        ):
            raise ExperimentProtocolError("dataset ledger marker changed after freezing")
        if not hmac.compare_digest(
            hashlib.sha256(dataset_payload).hexdigest(),
            str(normalized["dataset_marker_sha256"]),
        ):
            raise ExperimentProtocolError("dataset ledger marker was modified")
        return root_fd, dataset_fd, normalized
    except BaseException:
        if dataset_fd >= 0:
            os.close(dataset_fd)
        os.close(root_fd)
        raise


def validate_test_ledger_binding(
    ledger_root: str | os.PathLike[str],
    binding: Mapping[str, object],
    *,
    dataset_manifest_sha256: str,
) -> dict[str, object]:
    """Verify a frozen ledger binding without creating or modifying anything."""

    root_fd, dataset_fd, normalized = _open_bound_ledger_directories(
        ledger_root,
        binding,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )
    os.close(dataset_fd)
    os.close(root_fd)
    return normalized


def acquire_test_access_guard(
    ledger_root: str | os.PathLike[str],
    *,
    ledger_binding: Mapping[str, object],
    study_id: str,
    dataset_manifest_sha256: str,
    protocol_digest: str,
    selection_receipts_sha256: str,
    feature_archive_hashes: Mapping[str, str],
    source_revision: str,
) -> TestAccessGrant:
    """Atomically consume one test run scoped by dataset and study.

    The ledger location is independent of result/output directories. Hence a
    caller cannot obtain another evaluation simply by changing output path.
    """

    if re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", study_id) is None:
        raise ExperimentProtocolError("study_id must use 1-128 safe identifier characters")
    for value, field_name in (
        (dataset_manifest_sha256, "dataset_manifest_sha256"),
        (protocol_digest, "protocol_digest"),
        (selection_receipts_sha256, "selection_receipts_sha256"),
        (source_revision, "source_revision"),
    ):
        _require_sha256(value, field_name)
    if not feature_archive_hashes:
        raise ExperimentProtocolError("feature archive hashes must be non-empty")
    normalized_archives = {
        str(name): str(digest) for name, digest in sorted(feature_archive_hashes.items())
    }
    for name, digest in normalized_archives.items():
        if not name:
            raise ExperimentProtocolError("feature archive alias must be non-empty")
        _require_sha256(digest, f"feature archive hash for {name}")

    root_fd, directory_fd, normalized_binding = _open_bound_ledger_directories(
        ledger_root,
        ledger_binding,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )
    study_key = hashlib.sha256(study_id.encode("utf-8")).hexdigest()
    record_name = f"{study_key}.json"
    path = (
        Path(str(normalized_binding["canonical_root"]))
        / dataset_manifest_sha256
        / record_name
    )
    payload = _canonical_json(
        {
            "schema": "scinumbench-v2-global-test-ledger/3",
            "study_id": study_id,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "ledger_binding_sha256": hashlib.sha256(
                _canonical_json(normalized_binding)
            ).hexdigest(),
            "protocol_digest": protocol_digest,
            "selection_receipts_sha256": selection_receipts_sha256,
            "feature_archive_hashes": normalized_archives,
            "source_revision": source_revision,
            "state": "CONSUMED_BEFORE_TEST_JSONL_OPEN",
        }
    )
    payload_sha = hashlib.sha256(payload).hexdigest()
    try:
        descriptor = os.open(
            record_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
    except FileExistsError as exc:
        os.close(directory_fd)
        os.close(root_fd)
        raise ExperimentProtocolError(
            "test evaluation was already consumed for this dataset/study"
        ) from exc
    except BaseException:
        os.close(directory_fd)
        os.close(root_fd)
        raise
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
        os.close(root_fd)
    return TestAccessGrant(
        _seal=_TEST_ACCESS_SEAL,
        ledger_path=path,
        study_id=study_id,
        dataset_manifest_sha256=dataset_manifest_sha256,
        protocol_digest=protocol_digest,
        ledger_payload_sha256=payload_sha,
        feature_archive_sha256=tuple(sorted(normalized_archives.values())),
        ledger_binding=normalized_binding,
    )


def unlock_test_partition(
    archive: FeatureArchive,
    grant: TestAccessGrant,
    *,
    split_dir: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
) -> UnlockedTestPartition:
    """Open and validate test labels only after a valid global-ledger grant."""

    if not isinstance(grant, TestAccessGrant):
        raise ExperimentProtocolError("test partition requires a sealed global-ledger grant")
    if archive.dataset_manifest_sha256 != grant.dataset_manifest_sha256:
        raise ExperimentProtocolError("test grant belongs to another frozen dataset")
    if archive.archive_sha256 not in grant.feature_archive_sha256:
        raise ExperimentProtocolError("feature archive was not included in test precommitment")
    try:
        root_fd, dataset_fd, _ = _open_bound_ledger_directories(
            str(grant.ledger_binding.get("canonical_root", "")),
            grant.ledger_binding,
            dataset_manifest_sha256=grant.dataset_manifest_sha256,
        )
        try:
            study_key = hashlib.sha256(grant.study_id.encode("utf-8")).hexdigest()
            ledger_bytes = _read_file_at(dataset_fd, f"{study_key}.json")
        finally:
            os.close(dataset_fd)
            os.close(root_fd)
    except (OSError, ExperimentProtocolError) as exc:
        raise ExperimentProtocolError(
            "test ledger binding or record changed after the grant"
        ) from exc
    if not ledger_bytes.endswith(b"\n"):
        raise ExperimentProtocolError("test ledger record is not canonical")
    payload = ledger_bytes[:-1]
    if not hmac.compare_digest(hashlib.sha256(payload).hexdigest(), grant.ledger_payload_sha256):
        raise ExperimentProtocolError("test ledger record was modified")

    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest_sha = hashlib.sha256(manifest_json(manifest).encode("utf-8")).hexdigest()
    if manifest_sha != grant.dataset_manifest_sha256:
        raise ExperimentProtocolError("test manifest differs from global-ledger grant")
    split_root = Path(split_dir)
    split_paths = {
        name: split_root / f"{name}.jsonl"
        for name in ("train", "validation", "test")
    }
    # This deliberately is the first full verification in the runner. It opens
    # the test JSONL only after the immutable ledger record already exists.
    verify_manifest(manifest, split_paths)
    test_rows = _read_partition(split_paths["test"], "test")
    indices = archive.indices(Partition.TEST)
    if len(indices) != len(test_rows):
        raise ExperimentProtocolError("test archive/JSONL row counts differ")
    expected_ids = tuple(str(row["example_id"]) for row in test_rows)
    expected_sources = tuple(str(row["source_id"]) for row in test_rows)
    expected_papers = tuple(
        str(row.get("paper_id") or row.get("arxiv_id") or "") for row in test_rows
    )
    expected_hashes = tuple(
        hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest() for row in test_rows
    )
    if tuple(archive.example_ids[index] for index in indices) != expected_ids:
        raise ExperimentProtocolError("test example IDs differ from feature archive")
    if tuple(archive.source_ids[index] for index in indices) != expected_sources:
        raise ExperimentProtocolError("test source IDs differ from feature archive")
    if tuple(archive.paper_ids[index] for index in indices) != expected_papers:
        raise ExperimentProtocolError("test paper IDs differ from feature archive")
    if tuple(archive.text_sha256[index] for index in indices) != expected_hashes:
        raise ExperimentProtocolError("test text hashes differ from feature archive")
    labels = tuple(int(row["label"]) for row in test_rows)
    if set(labels) != {0, 1}:
        raise ExperimentProtocolError("test lacks one binary class")
    return UnlockedTestPartition(
        archive_sha256=archive.archive_sha256,
        dataset_manifest_sha256=archive.dataset_manifest_sha256,
        ledger_payload_sha256=grant.ledger_payload_sha256,
        indices=indices,
        example_ids=expected_ids,
        source_ids=expected_sources,
        paper_ids=expected_papers,
        labels=labels,
    )


def candidate_choice_payload(choice: FrozenCandidateChoice) -> dict[str, object]:
    return {
        "schema": "scinumbench-v2-validation-selection-receipt/1",
        "candidate_name": choice.candidate_name,
        "validation_score": choice.validation_score,
        "dataset_id": choice.dataset_id,
        "validation_manifest_hash": choice.validation_manifest_hash,
        "expected_test_manifest_hash": choice.expected_test_manifest_hash,
        "candidate_configs": [list(item) for item in choice.candidate_configs],
        "candidate_set_digest": choice.candidate_set_digest,
        "metric_id": choice.metric_id,
        "metric_implementation_hash": choice.metric_implementation_hash,
        "protocol_digest": choice.protocol_digest,
        "selected_on": choice.selected_on.value,
    }


def seed_mean_choice_payload(choice: FrozenSeedMeanChoice) -> dict[str, object]:
    return {
        "schema": "scinumbench-v2-seed-mean-validation-selection-receipt/1",
        "candidate_name": choice.candidate_name,
        "validation_seed_auroc": {
            str(seed): value for seed, value in choice.validation_seed_auroc
        },
        "validation_seed_auroc_mean": choice.validation_score,
        "seeds": list(choice.seeds),
        "aggregation": "arithmetic mean of per-seed AUROC",
        "dataset_id": choice.dataset_id,
        "validation_manifest_hash": choice.validation_manifest_hash,
        "expected_test_manifest_hash": choice.expected_test_manifest_hash,
        "candidate_configs": [list(item) for item in choice.candidate_configs],
        "candidate_set_digest": choice.candidate_set_digest,
        "candidate_metrics_digest": choice.candidate_metrics_digest,
        "metric_id": choice.metric_id,
        "metric_implementation_hash": choice.metric_implementation_hash,
        "protocol_digest": choice.protocol_digest,
        "selected_on": choice.selected_on.value,
    }


def threshold_choice_payload(choice: FrozenThresholdChoice) -> dict[str, object]:
    return {
        "schema": "scinumbench-v2-validation-threshold-receipt/1",
        "candidate_name": choice.candidate_name,
        "candidate_config_hash": choice.candidate_config_hash,
        "threshold": choice.threshold,
        "validation_f1": choice.validation_f1,
        "dataset_id": choice.dataset_id,
        "validation_labeled_manifest_hash": choice.validation_labeled_manifest_hash,
        "expected_test_manifest_hash": choice.expected_test_manifest_hash,
        "metric_id": choice.metric_id,
        "metric_implementation_hash": choice.metric_implementation_hash,
        "protocol_digest": choice.protocol_digest,
        "selected_on": choice.selected_on.value,
    }


__all__ = [
    "EXPERIMENT_PROTOCOL_VERSION",
    "PRIMARY_PROBE_SEEDS",
    "ExperimentProtocolError",
    "FeatureArchive",
    "FrozenSeedMeanChoice",
    "FrozenThresholdChoice",
    "FusionCandidates",
    "GroupedBootstrapDifference",
    "GroupedBootstrapResult",
    "ProbeCandidate",
    "ProbeSearch",
    "ProbeTestPredictions",
    "SeedMetricReport",
    "SeedMeanEvaluation",
    "SeededCandidateScoreSet",
    "SeededFusionCandidates",
    "TestAccessGrant",
    "UnlockedTestPartition",
    "acquire_test_access_guard",
    "assert_expected_model_identity",
    "assert_feature_archives_aligned",
    "average_precision",
    "binary_f1",
    "build_probability_fusions",
    "build_seeded_probability_fusions",
    "candidate_choice_payload",
    "evaluate_f1_with_frozen_threshold",
    "evaluate_seed_mean_choice_on_test",
    "fit_probe_candidates",
    "function_implementation_hash",
    "grouped_bootstrap_metrics",
    "grouped_bootstrap_auroc_difference",
    "grouped_bootstrap_seed_mean_auroc_difference",
    "grouped_bootstrap_seed_mean_metrics",
    "prepare_test_ledger_binding",
    "load_verified_feature_archive",
    "predict_probe_on_test",
    "seed_mean_auroc",
    "seed_mean_auroc_implementation_hash",
    "seed_mean_choice_payload",
    "select_f1_threshold_on_validation",
    "select_seed_mean_auroc_on_validation",
    "summarize_seed_metrics",
    "threshold_choice_payload",
    "unlock_test_partition",
    "validate_test_ledger_binding",
]
