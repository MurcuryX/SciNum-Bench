"""Fail-closed dataset freezing and integrity verification for SciNumBench V2.

The public manifest intentionally contains logical split names only.  Local
absolute paths are supplied by the caller at freeze/verification time and are
never serialized, so the same manifest can be checked on another machine.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from .split_protocol import SPLIT_NAMES, ProtocolViolation, content_hash


SCHEMA_VERSION = "scinumbench-freeze-manifest-v2"
NORMALIZATION_VERSION = "table-text-nfkc-markdown-whitespace-v1"
BLIND_TEST_SCHEMA_VERSION = "scinumbench-v2-label-blind-test-inputs-v1"
BLIND_TEST_FIELDS = ("example_id", "source_id", "paper_id", "split", "text")


class ManifestError(ValueError):
    """Raised when data cannot be frozen or no longer matches its manifest."""


IdentityKey = tuple[str, str]
SourceKey = IdentityKey
PaperKey = IdentityKey


@dataclass
class _SourceState:
    split: str
    labels: Counter[int] = field(default_factory=Counter)
    rows: int = 0


@dataclass(frozen=True)
class _FileStats:
    sha256: str
    byte_count: int
    row_count: int
    label_counts: Mapping[str, int]
    source_count: int
    paper_count: int
    content_hash_count: int

    def as_manifest_dict(self, split: str) -> dict[str, Any]:
        return {
            "logical_path": f"{split}.jsonl",
            "sha256": self.sha256,
            "bytes": self.byte_count,
            "rows": self.row_count,
            "label_counts": dict(self.label_counts),
            "source_count": self.source_count,
            "paper_count": self.paper_count,
            "content_hash_count": self.content_hash_count,
        }


@dataclass(frozen=True)
class _BlindFileStats:
    sha256: str
    byte_count: int
    row_count: int

    def as_manifest_dict(self, logical_path: str) -> dict[str, Any]:
        return {
            "schema_version": BLIND_TEST_SCHEMA_VERSION,
            "logical_path": logical_path,
            "sha256": self.sha256,
            "bytes": self.byte_count,
            "rows": self.row_count,
            "fields": list(BLIND_TEST_FIELDS),
            "contains_labels": False,
        }


def _identity_key(value: object, *, kind: str, location: str) -> IdentityKey:
    """Return a type-stable key for a JSON scalar identity."""

    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ManifestError(f"{location}: {kind} ID must be a string or integer")
    if isinstance(value, str):
        if not value.strip():
            raise ManifestError(f"{location}: {kind} ID must be non-empty")
        return ("str", value)
    return ("int", str(value))


def _source_key(value: object, *, location: str) -> SourceKey:
    return _identity_key(value, kind="source", location=location)


def _paper_key(value: object, *, location: str) -> PaperKey:
    return _identity_key(value, kind="paper", location=location)


def _record_source(record: Mapping[str, object], *, location: str) -> SourceKey:
    has_source = "source_id" in record
    has_legacy = "src_table_id" in record
    if not has_source and not has_legacy:
        raise ManifestError(f"{location}: missing source_id (or src_table_id)")

    selected = record["source_id"] if has_source else record["src_table_id"]
    key = _source_key(selected, location=location)
    if has_source and has_legacy:
        legacy_key = _source_key(record["src_table_id"], location=location)
        if key != legacy_key:
            raise ManifestError(
                f"{location}: source_id and src_table_id disagree"
            )
    return key


def _record_paper(record: Mapping[str, object], *, location: str) -> PaperKey:
    """Read the required document identity, accepting its arXiv alias.

    Release rows must carry a paper-level identity.  Supporting both names
    keeps the manifest compatible with source databases while ensuring that a
    builder cannot silently omit the document-level leakage key.
    """

    has_paper = "paper_id" in record
    has_arxiv = "arxiv_id" in record
    if not has_paper and not has_arxiv:
        raise ManifestError(f"{location}: missing paper_id (or arxiv_id)")

    selected = record["paper_id"] if has_paper else record["arxiv_id"]
    key = _paper_key(selected, location=location)
    if has_paper and has_arxiv:
        arxiv_key = _paper_key(record["arxiv_id"], location=location)
        if key != arxiv_key:
            raise ManifestError(f"{location}: paper_id and arxiv_id disagree")
    return key


def _blind_projection(record: Mapping[str, object], *, location: str) -> dict[str, object]:
    """Return the only fields permitted in the public test-inference view."""

    example_id = record.get("example_id")
    if not isinstance(example_id, str) or not example_id.strip():
        raise ManifestError(f"{location}: example_id must be a non-empty string")
    if example_id != example_id.strip():
        raise ManifestError(f"{location}: example_id may not have edge whitespace")
    text = record.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ManifestError(f"{location}: text must be a non-empty string")
    if record.get("split") != "test":
        raise ManifestError(f"{location}: blind projection requires split='test'")

    _record_source(record, location=location)
    _record_paper(record, location=location)
    source = record["source_id"] if "source_id" in record else record["src_table_id"]
    paper = record["paper_id"] if "paper_id" in record else record["arxiv_id"]
    return {
        "example_id": example_id,
        "source_id": source,
        "paper_id": paper,
        "split": "test",
        "text": text,
    }


def write_label_blind_test_inputs(
    labeled_test_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
) -> Path:
    """Atomically derive a fixed-schema test inference view without labels.

    This helper belongs to dataset construction/freeze time. Experiment code
    must consume the returned file and must not derive it from the labeled test
    file itself.
    """

    source = Path(labeled_test_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    seen: set[str] = set()
    try:
        with source.open(encoding="utf-8") as input_handle, temporary.open(
            "w", encoding="utf-8", newline=""
        ) as output_handle:
            for line_number, line in enumerate(input_handle, 1):
                location = f"test.jsonl:{line_number}"
                if not line.strip():
                    raise ManifestError(f"{location}: blank lines are not allowed")
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ManifestError(f"{location}: invalid JSON: {exc.msg}") from exc
                if not isinstance(record, dict):
                    raise ManifestError(f"{location}: each line must be a JSON object")
                projected = _blind_projection(record, location=location)
                example_id = str(projected["example_id"])
                if example_id in seen:
                    raise ManifestError(f"{location}: duplicate example_id {example_id!r}")
                seen.add(example_id)
                output_handle.write(
                    json.dumps(
                        projected,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
            if not seen:
                raise ManifestError("test.jsonl is empty")
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return destination


def _scan_blind_test_inputs(path: Path) -> tuple[_BlindFileStats, list[dict[str, object]]]:
    if not path.is_file():
        raise ManifestError(f"label-blind test input file does not exist: {path}")
    digest = hashlib.sha256()
    byte_count = 0
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            digest.update(raw_line)
            byte_count += len(raw_line)
            location = f"{path.name}:{line_number}"
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ManifestError(f"{location}: invalid UTF-8") from exc
            if not line.strip():
                raise ManifestError(f"{location}: blank lines are not allowed")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"{location}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ManifestError(f"{location}: each line must be a JSON object")
            if set(record) != set(BLIND_TEST_FIELDS):
                raise ManifestError(
                    f"{location}: blind test fields must be exactly {BLIND_TEST_FIELDS}"
                )
            projected = _blind_projection(record, location=location)
            if projected != record:
                raise ManifestError(f"{location}: blind test record is not canonical")
            example_id = str(record["example_id"])
            if example_id in seen:
                raise ManifestError(f"{location}: duplicate example_id {example_id!r}")
            seen.add(example_id)
            try:
                content_hash(str(record["text"]))
            except ProtocolViolation as exc:
                raise ManifestError(
                    f"{location}: text is not a canonical parseable Markdown table: {exc}"
                ) from exc
            rows.append(record)
    if not rows:
        raise ManifestError("label-blind test input file is empty")
    return (
        _BlindFileStats(
            sha256=digest.hexdigest(),
            byte_count=byte_count,
            row_count=len(rows),
        ),
        rows,
    )


def _verify_blind_matches_labeled_test(
    labeled_test_path: Path,
    blind_rows: list[dict[str, object]],
) -> None:
    expected: list[dict[str, object]] = []
    with labeled_test_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            location = f"test.jsonl:{line_number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"{location}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ManifestError(f"{location}: each line must be a JSON object")
            expected.append(_blind_projection(record, location=location))
    if expected != blind_rows:
        raise ManifestError("label-blind test inputs do not exactly match labeled test rows")


def _validate_ratios(ratios: Mapping[str, float]) -> dict[str, float]:
    if set(ratios) != set(SPLIT_NAMES):
        raise ManifestError(
            f"ratios must have exactly these keys: {', '.join(SPLIT_NAMES)}"
        )
    normalized: dict[str, float] = {}
    for split in SPLIT_NAMES:
        value = float(ratios[split])
        if not math.isfinite(value) or value <= 0.0:
            raise ManifestError(f"ratio for {split} must be finite and positive")
        normalized[split] = value
    if not math.isclose(sum(normalized.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ManifestError("split ratios must sum to 1")
    return normalized


def _validate_created_at(created_at: str) -> str:
    if not isinstance(created_at, str) or not created_at.strip():
        raise ManifestError("created_at must be a non-empty ISO-8601 timestamp")
    candidate = created_at.strip()
    try:
        parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ManifestError("created_at must be a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ManifestError("created_at must include a timezone")
    return candidate


def _scan_jsonl(
    path: Path,
    split: str,
    *,
    seen_example_ids: set[str],
    sources: dict[SourceKey, _SourceState],
    paper_owner: dict[PaperKey, str],
    hash_owner: dict[str, str],
) -> _FileStats:
    if not path.is_file():
        raise ManifestError(f"{split}: JSONL file does not exist: {path}")

    digest = hashlib.sha256()
    byte_count = 0
    row_count = 0
    labels: Counter[int] = Counter()
    local_sources: set[SourceKey] = set()
    local_papers: set[PaperKey] = set()
    local_hashes: set[str] = set()

    try:
        handle = path.open("rb")
    except OSError as exc:
        raise ManifestError(f"{split}: cannot open JSONL file: {exc}") from exc

    with handle:
        for line_number, raw_line in enumerate(handle, start=1):
            digest.update(raw_line)
            byte_count += len(raw_line)
            location = f"{split}.jsonl:{line_number}"
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ManifestError(f"{location}: invalid UTF-8") from exc
            if not line.strip():
                raise ManifestError(f"{location}: blank lines are not allowed")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"{location}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ManifestError(f"{location}: each line must be a JSON object")

            example_id = record.get("example_id")
            if not isinstance(example_id, str) or not example_id.strip():
                raise ManifestError(f"{location}: example_id must be a non-empty string")
            if example_id != example_id.strip():
                raise ManifestError(f"{location}: example_id may not have edge whitespace")
            if example_id in seen_example_ids:
                raise ManifestError(f"{location}: duplicate example_id {example_id!r}")
            seen_example_ids.add(example_id)

            label = record.get("label")
            if isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1):
                raise ManifestError(f"{location}: label must be integer 0 or 1")
            text = record.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ManifestError(f"{location}: text must be a non-empty string")
            if "split" in record and record["split"] != split:
                raise ManifestError(
                    f"{location}: embedded split {record['split']!r} does not match {split!r}"
                )

            source = _record_source(record, location=location)
            state = sources.get(source)
            if state is None:
                state = _SourceState(split=split)
                sources[source] = state
            elif state.split != split:
                raise ManifestError(
                    f"{location}: source appears in both {state.split} and {split}"
                )
            state.labels[label] += 1
            state.rows += 1

            paper = _record_paper(record, location=location)
            paper_split = paper_owner.setdefault(paper, split)
            if paper_split != split:
                raise ManifestError(
                    f"{location}: paper appears in both {paper_split} and {split}"
                )

            try:
                table_hash = content_hash(text)
            except ProtocolViolation as exc:
                raise ManifestError(
                    f"{location}: text is not a canonical parseable Markdown table: {exc}"
                ) from exc
            owner = hash_owner.setdefault(table_hash, split)
            if owner != split:
                raise ManifestError(
                    f"{location}: normalized content also appears in {owner}"
                )

            row_count += 1
            labels[label] += 1
            local_sources.add(source)
            local_papers.add(paper)
            local_hashes.add(table_hash)

    if row_count == 0:
        raise ManifestError(f"{split}: JSONL file is empty")
    return _FileStats(
        sha256=digest.hexdigest(),
        byte_count=byte_count,
        row_count=row_count,
        label_counts={"0": labels[0], "1": labels[1]},
        source_count=len(local_sources),
        paper_count=len(local_papers),
        content_hash_count=len(local_hashes),
    )


def freeze_manifest(
    split_paths: Mapping[str, str | os.PathLike[str]],
    *,
    seed: int,
    ratios: Mapping[str, float],
    code_revision: str,
    created_at: str,
    test_inputs_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Validate split JSONL files and return a deterministic logical manifest.

    The three files are streamed once.  No example text or absolute file path
    is retained in the returned object.
    """

    if set(split_paths) != set(SPLIT_NAMES):
        raise ManifestError(
            f"split_paths must have exactly these keys: {', '.join(SPLIT_NAMES)}"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ManifestError("seed must be an integer")
    normalized_ratios = _validate_ratios(ratios)
    if not isinstance(code_revision, str) or not code_revision.strip():
        raise ManifestError("code_revision must be a non-empty string")
    revision = code_revision.strip()
    timestamp = _validate_created_at(created_at)

    seen_example_ids: set[str] = set()
    sources: dict[SourceKey, _SourceState] = {}
    paper_owner: dict[PaperKey, str] = {}
    hash_owner: dict[str, str] = {}
    file_stats: dict[str, dict[str, Any]] = {}

    for split in SPLIT_NAMES:
        stats = _scan_jsonl(
            Path(split_paths[split]),
            split,
            seen_example_ids=seen_example_ids,
            sources=sources,
            paper_owner=paper_owner,
            hash_owner=hash_owner,
        )
        file_stats[split] = stats.as_manifest_dict(split)

    for source, state in sources.items():
        if state.rows != 2 or state.labels != Counter({0: 1, 1: 1}):
            printable = f"{source[0]}:{source[1]}"
            raise ManifestError(
                f"source {printable} in {state.split} must have exactly one "
                f"negative and one positive twin; observed labels={dict(state.labels)}"
            )

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "created_at": timestamp,
        "code_revision": revision,
        "split_policy": {
            "seed": seed,
            "ratios": normalized_ratios,
            "grouping_keys": [
                "source_id",
                "paper_id_or_arxiv_id",
                "normalized_content_sha256",
            ],
        },
        "files": file_stats,
        "totals": {
            "rows": len(seen_example_ids),
            "examples": len(seen_example_ids),
            "sources": len(sources),
            "papers": len(paper_owner),
            "content_hashes": len(hash_owner),
        },
        "gates": {
            "utf8_jsonl": "PASS",
            "schema": "PASS",
            "unique_example_ids": "PASS",
            "complete_twins": "PASS",
            "source_isolation": "PASS",
            "paper_isolation": "PASS",
            "content_isolation": "PASS",
        },
    }
    if test_inputs_path is not None:
        blind_path = Path(test_inputs_path)
        logical_path = blind_path.name
        if (
            Path(logical_path).is_absolute()
            or ".." in Path(logical_path).parts
            or logical_path in {"train.jsonl", "validation.jsonl", "test.jsonl"}
        ):
            raise ManifestError("label-blind test input logical path is unsafe")
        blind_stats, blind_rows = _scan_blind_test_inputs(blind_path)
        _verify_blind_matches_labeled_test(Path(split_paths["test"]), blind_rows)
        if blind_stats.row_count != int(file_stats["test"]["rows"]):
            raise ManifestError("label-blind and labeled test row counts differ")
        result["test_inputs"] = blind_stats.as_manifest_dict(logical_path)
        result["gates"]["label_blind_test_inputs"] = "PASS"
    return result


def manifest_json(manifest: Mapping[str, object]) -> str:
    """Return canonical, human-readable JSON for storage or hashing."""

    return json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def verify_manifest(
    manifest: Mapping[str, object],
    split_paths: Mapping[str, str | os.PathLike[str]],
    *,
    test_inputs_path: str | os.PathLike[str] | None = None,
) -> bool:
    """Re-scan files and require exact equality with a frozen manifest."""

    if not isinstance(manifest, Mapping):
        raise ManifestError("manifest must be a mapping")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError("unsupported or missing manifest schema_version")
    if manifest.get("normalization_version") != NORMALIZATION_VERSION:
        raise ManifestError("unsupported or missing normalization_version")

    policy = manifest.get("split_policy")
    if not isinstance(policy, Mapping):
        raise ManifestError("manifest split_policy must be an object")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ManifestError("manifest files must be an object")
    for split in SPLIT_NAMES:
        entry = files.get(split)
        if not isinstance(entry, Mapping):
            raise ManifestError(f"manifest is missing file entry for {split}")
        logical_path = entry.get("logical_path")
        if logical_path != f"{split}.jsonl":
            raise ManifestError(f"manifest has invalid logical_path for {split}")
        if Path(str(logical_path)).is_absolute() or ".." in Path(str(logical_path)).parts:
            raise ManifestError("manifest logical paths may not be absolute or traverse parents")

    seed = policy.get("seed")
    ratios = policy.get("ratios")
    if not isinstance(ratios, Mapping):
        raise ManifestError("manifest split ratios must be an object")
    revision = manifest.get("code_revision")
    timestamp = manifest.get("created_at")
    if not isinstance(revision, str) or not isinstance(timestamp, str):
        raise ManifestError("manifest code_revision and created_at must be strings")

    blind_entry = manifest.get("test_inputs")
    if blind_entry is not None:
        if not isinstance(blind_entry, Mapping):
            raise ManifestError("manifest test_inputs must be an object")
        logical_path = blind_entry.get("logical_path")
        if not isinstance(logical_path, str) or not logical_path:
            raise ManifestError("manifest test_inputs logical_path must be non-empty")
        logical = Path(logical_path)
        if logical.is_absolute() or ".." in logical.parts or len(logical.parts) != 1:
            raise ManifestError("manifest test_inputs logical_path is unsafe")
        if test_inputs_path is None:
            test_inputs_path = Path(split_paths["test"]).parent / logical_path
    elif test_inputs_path is not None:
        raise ManifestError("test_inputs_path supplied for a manifest without test_inputs")

    rebuilt = freeze_manifest(
        split_paths,
        seed=seed,  # type: ignore[arg-type]
        ratios=ratios,  # type: ignore[arg-type]
        code_revision=revision,
        created_at=timestamp,
        test_inputs_path=test_inputs_path,
    )
    if rebuilt != dict(manifest):
        mismatched: list[str] = []
        for key in sorted(set(rebuilt) | set(manifest)):
            if rebuilt.get(key) != manifest.get(key):
                mismatched.append(str(key))
        raise ManifestError(
            "frozen manifest does not match current files or metadata; "
            f"mismatched sections: {', '.join(mismatched)}"
        )
    return True


def _read_public_labeled_partition(
    path: Path,
    split: str,
    expected: Mapping[str, object],
) -> list[dict[str, object]]:
    digest = hashlib.sha256()
    byte_count = 0
    rows: list[dict[str, object]] = []
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            digest.update(raw_line)
            byte_count += len(raw_line)
            location = f"{split}.jsonl:{line_number}"
            try:
                record = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ManifestError(f"{location}: invalid UTF-8 JSON") from exc
            if not isinstance(record, dict):
                raise ManifestError(f"{location}: record must be an object")
            if record.get("split") != split:
                raise ManifestError(f"{location}: embedded split mismatch")
            label = record.get("label")
            if isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1):
                raise ManifestError(f"{location}: label must be integer 0 or 1")
            _blind_projection({**record, "split": "test"}, location=location)
            rows.append(record)
    observed = {
        "sha256": digest.hexdigest(),
        "bytes": byte_count,
        "rows": len(rows),
    }
    for field_name, value in observed.items():
        if expected.get(field_name) != value:
            raise ManifestError(f"{split} differs from its frozen manifest ({field_name})")
    if not rows:
        raise ManifestError(f"{split} public extraction partition is empty")
    return rows


def load_public_extraction_rows(
    manifest: Mapping[str, object],
    split_dir: str | os.PathLike[str],
) -> dict[str, list[dict[str, object]]]:
    """Load train/validation and the frozen blind test view, never labeled test.

    The implementation deliberately has no labeled-test path parameter. It is
    safe to call before the global test-access ledger is consumed.
    """

    if not isinstance(manifest, Mapping):
        raise ManifestError("manifest must be a mapping")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError("unsupported or missing manifest schema_version")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ManifestError("manifest files must be an object")
    root = Path(split_dir)
    result: dict[str, list[dict[str, object]]] = {}
    for split in ("train", "validation"):
        entry = files.get(split)
        if not isinstance(entry, Mapping):
            raise ManifestError(f"manifest is missing file entry for {split}")
        if entry.get("logical_path") != f"{split}.jsonl":
            raise ManifestError(f"manifest has invalid logical_path for {split}")
        result[split] = _read_public_labeled_partition(
            root / f"{split}.jsonl",
            split,
            entry,
        )

    test_entry = files.get("test")
    if not isinstance(test_entry, Mapping):
        raise ManifestError("manifest is missing file entry for test")
    if test_entry.get("logical_path") != "test.jsonl":
        raise ManifestError("manifest has invalid logical_path for test")
    blind_entry = manifest.get("test_inputs")
    if not isinstance(blind_entry, Mapping):
        raise ManifestError("manifest lacks frozen label-blind test_inputs")
    if blind_entry.get("schema_version") != BLIND_TEST_SCHEMA_VERSION:
        raise ManifestError("unsupported label-blind test input schema")
    if blind_entry.get("fields") != list(BLIND_TEST_FIELDS):
        raise ManifestError("manifest label-blind field list is invalid")
    if blind_entry.get("contains_labels") is not False:
        raise ManifestError("manifest does not guarantee label-blind test inputs")
    logical_path = blind_entry.get("logical_path")
    if not isinstance(logical_path, str) or not logical_path:
        raise ManifestError("manifest test_inputs logical_path must be non-empty")
    logical = Path(logical_path)
    if logical.is_absolute() or ".." in logical.parts or len(logical.parts) != 1:
        raise ManifestError("manifest test_inputs logical_path is unsafe")
    blind_stats, blind_rows = _scan_blind_test_inputs(root / logical_path)
    observed_blind = {
        "sha256": blind_stats.sha256,
        "bytes": blind_stats.byte_count,
        "rows": blind_stats.row_count,
    }
    for field_name, value in observed_blind.items():
        if blind_entry.get(field_name) != value:
            raise ManifestError(
                f"label-blind test inputs differ from manifest ({field_name})"
            )
    if blind_stats.row_count != test_entry.get("rows"):
        raise ManifestError("label-blind and labeled test manifest row counts differ")
    result["test"] = blind_rows
    return result


__all__ = [
    "BLIND_TEST_FIELDS",
    "BLIND_TEST_SCHEMA_VERSION",
    "ManifestError",
    "NORMALIZATION_VERSION",
    "SCHEMA_VERSION",
    "freeze_manifest",
    "load_public_extraction_rows",
    "manifest_json",
    "verify_manifest",
    "write_label_blind_test_inputs",
]

