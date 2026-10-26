"""Leakage-resistant split construction and sealed evaluation protocols.

The hard track is paired: every source contributes exactly one unedited
negative and one edited positive. Split groups are connected components of
source identity, paper/document identity, and canonical table-grid identity,
and are never divided.

Validation-only model selection emits an auditable receipt containing dataset
manifests, every candidate's config hash, the metric implementation hash, and
a protocol digest. This is an accidental-misuse guard, not a security boundary
against a malicious Python caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import hmac
import json
import marshal
import math
import random
import re
from types import MappingProxyType
from typing import Any, Callable, Hashable, Iterable, Iterator, Mapping, Sequence
import unicodedata


SPLIT_NAMES = ("train", "validation", "test")
PROTOCOL_VERSION = "scinumbench-v2-selection/1"


class ProtocolViolation(ValueError):
    """Raised when an operation would violate the frozen protocol."""


class LeakageError(ProtocolViolation):
    """Raised when examples, sources, or canonical grids cross splits."""


class Partition(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True)
class SplitRecord:
    """Minimal paired-track record used by the split builder."""

    example_id: str
    source_id: Hashable
    text: str
    label: int | None = None
    paper_id: Hashable | None = None

    def __post_init__(self) -> None:
        if not self.example_id:
            raise ValueError("example_id must be non-empty")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must be a non-empty string")
        if self.label not in (None, 0, 1):
            raise ValueError("label must be 0, 1, or None")
        try:
            hash(self.source_id)
        except TypeError as exc:
            raise TypeError("source_id must be hashable") from exc
        if self.paper_id is not None:
            if isinstance(self.paper_id, str) and not self.paper_id.strip():
                raise ValueError("paper_id must be non-empty when provided")
            try:
                hash(self.paper_id)
            except TypeError as exc:
                raise TypeError("paper_id must be hashable when provided") from exc


# ---------------------------------------------------------------------------
# Strict structural Markdown hashing
# ---------------------------------------------------------------------------

_SEPARATOR_CELL = re.compile(r"^:?-{3,}:?$")
_CAPTION_LINE = re.compile(r"^\s*caption\s*:", re.IGNORECASE)


def _trailing_pipe_is_unescaped(text: str) -> bool:
    if not text.endswith("|"):
        return False
    backslashes = 0
    index = len(text) - 2
    while index >= 0 and text[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 0


def _split_markdown_row(line: str) -> tuple[str, ...]:
    """Parse one Markdown row, respecting escaped pipes and outer pipes."""

    text = line.strip()
    if not text:
        raise ProtocolViolation("empty Markdown row")
    if text.startswith("|"):
        text = text[1:]
    if _trailing_pipe_is_unescaped(text):
        text = text[:-1]

    fields: list[str] = []
    current: list[str] = []
    index = 0
    escapes = {"\\": "\\", "|": "|", "n": "\n", "r": "\r", "t": "\t"}
    while index < len(text):
        char = text[index]
        if char == "\\":
            if index + 1 >= len(text):
                raise ProtocolViolation("dangling escape in Markdown row")
            next_char = text[index + 1]
            if next_char in escapes:
                current.append(escapes[next_char])
            else:
                current.extend((char, next_char))
            index += 2
            continue
        if char == "|":
            fields.append("".join(current).strip())
            current = []
        else:
            current.append(char)
        index += 1
    fields.append("".join(current).strip())
    if len(fields) < 2:
        raise ProtocolViolation("a Markdown table row needs at least two columns")
    return tuple(fields)


def _canonical_cell(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value.casefold()


def parse_table_grid(text: str) -> tuple[tuple[str, ...], ...]:
    """Extract exactly one strict Markdown grid, excluding its caption.

    Alignment colons and optional outer pipes are presentation only. Caption
    lines and prose outside the contiguous table block are ignored. Missing or
    ambiguous separators and ragged rows are hard errors; no raw-string hash
    fallback exists.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")

    separators: list[tuple[int, tuple[str, ...]]] = []
    for index, line in enumerate(lines):
        if not line.strip() or _CAPTION_LINE.match(line):
            continue
        try:
            cells = _split_markdown_row(line)
        except ProtocolViolation:
            continue
        if all(_SEPARATOR_CELL.fullmatch(cell) for cell in cells):
            separators.append((index, cells))
    if not separators:
        raise ProtocolViolation("no valid Markdown header separator found")
    if len(separators) != 1:
        raise ProtocolViolation("expected exactly one Markdown table separator")

    separator_index, separator = separators[0]
    if separator_index == 0 or not lines[separator_index - 1].strip():
        raise ProtocolViolation("Markdown header must immediately precede its separator")
    if _CAPTION_LINE.match(lines[separator_index - 1]):
        raise ProtocolViolation("Markdown table header is missing")
    try:
        header = _split_markdown_row(lines[separator_index - 1])
    except ProtocolViolation as exc:
        raise ProtocolViolation(f"invalid Markdown table header: {exc}") from exc
    if len(header) != len(separator):
        raise ProtocolViolation("Markdown header and separator widths differ")

    body: list[tuple[str, ...]] = []
    for line in lines[separator_index + 1 :]:
        if not line.strip() or _CAPTION_LINE.match(line):
            break
        try:
            row = _split_markdown_row(line)
        except ProtocolViolation as exc:
            if "|" in line:
                raise ProtocolViolation(f"invalid Markdown body row: {exc}") from exc
            break
        if len(row) != len(header):
            raise ProtocolViolation("Markdown table is not rectangular")
        if all(_SEPARATOR_CELL.fullmatch(cell) for cell in row):
            raise ProtocolViolation("unexpected separator inside Markdown body")
        body.append(row)
    if not body:
        raise ProtocolViolation("Markdown table needs at least one body row")

    return tuple(tuple(_canonical_cell(cell) for cell in row) for row in (header, *body))


def normalize_table_text(text: str) -> str:
    """Canonical JSON for the parsed grid and its explicit shape."""

    grid = parse_table_grid(text)
    payload = {
        "schema": "scinumbench-v2-grid/1",
        "shape": [len(grid), len(grid[0])],
        "cells": [list(row) for row in grid],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(text: str) -> str:
    """SHA-256 of the strict structured grid; captions never enter the hash."""

    return hashlib.sha256(normalize_table_text(text).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Paired, content-connected split construction
# ---------------------------------------------------------------------------

class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _paired_source_errors(records: Sequence[SplitRecord], context: str) -> list[str]:
    grouped: dict[Hashable, list[SplitRecord]] = {}
    for row in records:
        grouped.setdefault(row.source_id, []).append(row)
    errors: list[str] = []
    for source_id, rows in grouped.items():
        labels = [row.label for row in rows]
        if len(rows) != 2 or labels.count(0) != 1 or labels.count(1) != 1:
            errors.append(
                f"{context} source {source_id!r} must have exactly one positive and one negative"
            )
    return errors


def _require_paired_sources(records: Sequence[SplitRecord], context: str) -> None:
    errors = _paired_source_errors(records, context)
    if errors:
        raise ProtocolViolation("; ".join(errors))


def connected_split_groups(records: Sequence[SplitRecord]) -> tuple[tuple[SplitRecord, ...], ...]:
    """Return source/paper/content components after enforcing true pairs.

    ``paper_id`` remains optional for compatibility with callers that only use
    this module for source-paired utilities. Dataset builders must provide it;
    the freeze manifest independently requires a paper identity on every
    released row.
    """

    rows = tuple(records)
    if not rows:
        raise ValueError("at least one record is required")
    seen_ids: set[str] = set()
    for row in rows:
        if row.example_id in seen_ids:
            raise ValueError(f"duplicate example_id: {row.example_id}")
        seen_ids.add(row.example_id)
    _require_paired_sources(rows, "paired track")

    union_find = _UnionFind(len(rows))
    first_by_source: dict[Hashable, int] = {}
    first_by_paper: dict[Hashable, int] = {}
    first_by_hash: dict[str, int] = {}
    for index, row in enumerate(rows):
        union_find.union(index, first_by_source.setdefault(row.source_id, index))
        if row.paper_id is not None:
            union_find.union(index, first_by_paper.setdefault(row.paper_id, index))
        digest = content_hash(row.text)
        union_find.union(index, first_by_hash.setdefault(digest, index))

    components: dict[int, list[SplitRecord]] = {}
    for index, row in enumerate(rows):
        components.setdefault(union_find.find(index), []).append(row)
    groups = [tuple(sorted(group, key=lambda item: item.example_id)) for group in components.values()]
    groups.sort(key=lambda group: tuple(item.example_id for item in group))
    return tuple(groups)


def _stable_group_order(group: Sequence[SplitRecord], seed: int) -> str:
    signature = "\x1f".join(row.example_id for row in group)
    return hashlib.sha256(f"{seed}\x00{signature}".encode()).hexdigest()


def _group_stats(group: Sequence[SplitRecord]) -> dict[str, int]:
    return {
        "rows": len(group),
        "sources": len({row.source_id for row in group}),
        "papers": len({row.paper_id for row in group if row.paper_id is not None}),
        "positive": sum(row.label == 1 for row in group),
        "negative": sum(row.label == 0 for row in group),
    }


def _largest_remainder_quotas(
    total: int,
    fractions: Mapping[str, float],
) -> dict[str, int]:
    """Convert fractional paper targets to deterministic integer quotas.

    Every split receives at least one paper.  This lower bound is necessary
    because all released splits must contain both members of at least one
    source pair.  For ordinary dataset sizes this is exactly Hamilton's
    largest-remainder apportionment; the repair only affects tiny totals for
    which the unconstrained method would assign zero to a positive fraction.
    """

    if total < len(SPLIT_NAMES):
        raise ProtocolViolation(
            f"paper-isolated splitting needs at least {len(SPLIT_NAMES)} papers"
        )
    exact = {split: total * fractions[split] for split in SPLIT_NAMES}
    quotas = {split: math.floor(exact[split]) for split in SPLIT_NAMES}
    remaining = total - sum(quotas.values())
    order = sorted(
        SPLIT_NAMES,
        key=lambda split: (
            -(exact[split] - quotas[split]),
            SPLIT_NAMES.index(split),
        ),
    )
    for split in order[:remaining]:
        quotas[split] += 1

    for recipient in (split for split in SPLIT_NAMES if quotas[split] == 0):
        donors = [split for split in SPLIT_NAMES if quotas[split] > 1]
        if not donors:
            raise ProtocolViolation("cannot give every split a non-zero paper quota")
        donor = max(
            donors,
            key=lambda split: (
                quotas[split] - exact[split],
                quotas[split],
                -SPLIT_NAMES.index(split),
            ),
        )
        quotas[donor] -= 1
        quotas[recipient] += 1
    return quotas


def deterministic_group_split(
    records: Sequence[SplitRecord],
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, tuple[SplitRecord, ...]]:
    """Stratify connected components by paper, source, label, and row counts."""

    fractions = {
        "train": float(train_fraction),
        "validation": float(validation_fraction),
        "test": float(test_fraction),
    }
    if any(not math.isfinite(value) or value <= 0 for value in fractions.values()):
        raise ValueError("all three split fractions must be finite and positive")
    if not math.isclose(sum(fractions.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("split fractions must sum to 1")

    rows = tuple(records)
    groups = list(connected_split_groups(rows))
    if len(groups) < 3:
        raise ProtocolViolation("at least three disconnected paired groups are required")
    paper_constrained = all(row.paper_id is not None for row in rows)
    paper_quotas: dict[str, int] | None = None
    if paper_constrained:
        paper_total = len({row.paper_id for row in rows})
        paper_quotas = _largest_remainder_quotas(paper_total, fractions)
    groups.sort(
        key=lambda group: (
            -_group_stats(group)["papers"] if paper_constrained else 0,
            -_group_stats(group)["sources"],
            -len(group),
            _stable_group_order(group, seed),
        )
    )

    total_stats = _group_stats(tuple(row for group in groups for row in group))
    targets = {
        split: {metric: fractions[split] * value for metric, value in total_stats.items()}
        for split in SPLIT_NAMES
    }
    counts = {split: {metric: 0 for metric in total_stats} for split in SPLIT_NAMES}
    assigned: dict[str, list[SplitRecord]] = {split: [] for split in SPLIT_NAMES}
    assigned_groups: dict[str, list[tuple[SplitRecord, ...]]] = {
        split: [] for split in SPLIT_NAMES
    }

    for group_index, group in enumerate(groups):
        stats = _group_stats(group)
        remaining = len(groups) - group_index
        empty = [split for split in SPLIT_NAMES if counts[split]["sources"] == 0]
        eligible = list(SPLIT_NAMES)
        if paper_quotas is not None:
            within_paper_capacity = [
                split
                for split in SPLIT_NAMES
                if counts[split]["papers"] + stats["papers"] <= paper_quotas[split]
            ]
            if within_paper_capacity:
                eligible = within_paper_capacity
        if len(empty) >= remaining:
            empty_eligible = [split for split in eligible if split in empty]
            eligible = empty_eligible or empty

        def improvement(split: str) -> float:
            gain = 0.0
            for metric, addition in stats.items():
                if paper_quotas is not None and metric == "papers":
                    continue
                # Compare global squared-error reduction on one common scale
                # per metric.  Normalising by each split's own (smaller)
                # target wrongly sends high-source papers to validation/test:
                # a five-source paper then appears more valuable to a 15%
                # split than to the 70% training split.  A global denominator
                # preserves the requested proportions while paper quotas keep
                # document counts exact.
                target = targets[split][metric]
                scale = max(float(total_stats[metric]) ** 2, 1e-12)
                before = (counts[split][metric] - target) ** 2 / scale
                after = (counts[split][metric] + addition - target) ** 2 / scale
                gain += before - after
            return gain

        def assignment_score(split: str) -> tuple[float, float, int]:
            overflow = 0
            if paper_quotas is not None:
                overflow = max(
                    0,
                    counts[split]["papers"] + stats["papers"] - paper_quotas[split],
                )
            return (-float(overflow), improvement(split), -SPLIT_NAMES.index(split))

        chosen = max(eligible, key=assignment_score)
        assigned[chosen].extend(group)
        assigned_groups[chosen].append(group)
        for metric, addition in stats.items():
            counts[chosen][metric] += addition

    if paper_quotas is not None:
        # The capacity-constrained greedy pass fixes the exact paper counts,
        # but an early run showed that it could still put every multi-table
        # paper in train (or, with per-split normalisation, in validation/test).
        # Deterministic equal-paper-count swaps repair source mass without ever
        # changing a paper quota or breaking a connected component.
        source_targets = {split: targets[split]["sources"] for split in SPLIT_NAMES}
        while True:
            best: tuple[
                float,
                str,
                int,
                str,
                int,
                str,
                str,
            ] | None = None
            for left_index, left in enumerate(SPLIT_NAMES):
                for right in SPLIT_NAMES[left_index + 1 :]:
                    for left_group_index, left_group in enumerate(assigned_groups[left]):
                        left_stats = _group_stats(left_group)
                        for right_group_index, right_group in enumerate(assigned_groups[right]):
                            right_stats = _group_stats(right_group)
                            if left_stats["papers"] != right_stats["papers"]:
                                continue
                            before = (
                                (counts[left]["sources"] - source_targets[left]) ** 2
                                + (counts[right]["sources"] - source_targets[right]) ** 2
                            )
                            left_after = (
                                counts[left]["sources"]
                                - left_stats["sources"]
                                + right_stats["sources"]
                            )
                            right_after = (
                                counts[right]["sources"]
                                - right_stats["sources"]
                                + left_stats["sources"]
                            )
                            after = (
                                (left_after - source_targets[left]) ** 2
                                + (right_after - source_targets[right]) ** 2
                            )
                            delta = after - before
                            if delta >= -1e-12:
                                continue
                            candidate = (
                                delta,
                                left,
                                left_group_index,
                                right,
                                right_group_index,
                                _stable_group_order(left_group, seed),
                                _stable_group_order(right_group, seed),
                            )
                            if best is None or candidate < best:
                                best = candidate
            if best is None:
                break
            _, left, left_index, right, right_index, _, _ = best
            left_group = assigned_groups[left][left_index]
            right_group = assigned_groups[right][right_index]
            left_stats = _group_stats(left_group)
            right_stats = _group_stats(right_group)
            assigned_groups[left][left_index] = right_group
            assigned_groups[right][right_index] = left_group
            for metric in total_stats:
                counts[left][metric] += right_stats[metric] - left_stats[metric]
                counts[right][metric] += left_stats[metric] - right_stats[metric]

        assigned = {
            split: [row for group in assigned_groups[split] for row in group]
            for split in SPLIT_NAMES
        }

    result = {
        split: tuple(sorted(rows, key=lambda row: row.example_id))
        for split, rows in assigned.items()
    }
    if paper_quotas is not None and all(_group_stats(group)["papers"] == 1 for group in groups):
        observed = {
            split: len({row.paper_id for row in result[split]}) for split in SPLIT_NAMES
        }
        if observed != paper_quotas:
            raise ProtocolViolation(
                f"paper quota assignment failed: expected {paper_quotas}, observed {observed}"
            )
    assert_no_split_leakage(result)
    return result


@dataclass(frozen=True)
class SplitValidation:
    ok: bool
    errors: tuple[str, ...]
    counts: Mapping[str, int]
    source_counts: Mapping[str, int]
    paper_counts: Mapping[str, int]
    label_counts: Mapping[str, tuple[int, int]]


def validate_split_bundle(splits: Mapping[str, Sequence[SplitRecord]]) -> SplitValidation:
    """Audit parsing, pair completeness, class presence, and all leakage keys."""

    errors: list[str] = []
    supplied, expected = set(splits), set(SPLIT_NAMES)
    if supplied != expected:
        if expected - supplied:
            errors.append(f"missing splits: {sorted(expected - supplied)}")
        if supplied - expected:
            errors.append(f"unexpected splits: {sorted(supplied - expected)}")

    ids: dict[str, set[str]] = {}
    sources: dict[str, set[Hashable]] = {}
    papers: dict[str, set[Hashable]] = {}
    hashes: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    paper_counts: dict[str, int] = {}
    label_counts: dict[str, tuple[int, int]] = {}
    for split in SPLIT_NAMES:
        rows = tuple(splits.get(split, ()))
        counts[split] = len(rows)
        ids[split] = {row.example_id for row in rows}
        sources[split] = {row.source_id for row in rows}
        papers[split] = {row.paper_id for row in rows if row.paper_id is not None}
        hashes[split] = {content_hash(row.text) for row in rows}
        source_counts[split] = len(sources[split])
        paper_counts[split] = len(papers[split])
        label_counts[split] = (
            sum(row.label == 0 for row in rows),
            sum(row.label == 1 for row in rows),
        )
        if len(ids[split]) != len(rows):
            errors.append(f"duplicate example_id inside {split}")
        if not rows:
            errors.append(f"empty split: {split}")
        if {row.label for row in rows} != {0, 1}:
            errors.append(f"{split} must contain both labels")
        errors.extend(_paired_source_errors(rows, split))

    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1 :]:
            if ids[left] & ids[right]:
                errors.append(f"example_id overlap {left}/{right}: {len(ids[left] & ids[right])}")
            if sources[left] & sources[right]:
                errors.append(f"source_id overlap {left}/{right}: {len(sources[left] & sources[right])}")
            if papers[left] & papers[right]:
                errors.append(f"paper_id overlap {left}/{right}: {len(papers[left] & papers[right])}")
            if hashes[left] & hashes[right]:
                errors.append(f"content_hash overlap {left}/{right}: {len(hashes[left] & hashes[right])}")

    return SplitValidation(
        ok=not errors,
        errors=tuple(errors),
        counts=MappingProxyType(counts),
        source_counts=MappingProxyType(source_counts),
        paper_counts=MappingProxyType(paper_counts),
        label_counts=MappingProxyType(label_counts),
    )


def assert_no_split_leakage(splits: Mapping[str, Sequence[SplitRecord]]) -> None:
    result = validate_split_bundle(splits)
    if not result.ok:
        raise LeakageError("; ".join(result.errors))


# ---------------------------------------------------------------------------
# Metric and validation-frozen candidate protocol
# ---------------------------------------------------------------------------

def binary_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Binary AUROC using average ranks for ties (Mann-Whitney form)."""

    y = tuple(int(value) for value in labels)
    s = tuple(float(value) for value in scores)
    if len(y) != len(s) or not y:
        raise ValueError("labels and scores must have the same non-zero length")
    if any(value not in (0, 1) for value in y):
        raise ValueError("labels must be binary")
    if any(not math.isfinite(value) for value in s):
        raise ValueError("scores must be finite")
    positives = sum(y)
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires both classes")

    ordered = sorted(range(len(s)), key=lambda index: s[index])
    ranks = [0.0] * len(s)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and s[ordered[end]] == s[ordered[start]]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[ordered[position]] = average_rank
        start = end
    positive_rank_sum = sum(rank for rank, label in zip(ranks, y) if label == 1)
    statistic = positive_rank_sum - positives * (positives + 1) / 2.0
    return statistic / (positives * negatives)


def _canonical_json(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value must be finite JSON-serializable data") from exc
    return rendered.encode("utf-8")


def candidate_config_hash(config: Mapping[str, Any]) -> str:
    """Hash a complete, JSON-serializable candidate/fusion configuration."""

    return hashlib.sha256(_canonical_json(config)).hexdigest()


def _require_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")


def _callable_hash(function: Callable[[Sequence[int], Sequence[float]], float]) -> str:
    code = getattr(function, "__code__", None)
    if code is None:
        raise ValueError("metric function must expose Python bytecode")
    if getattr(function, "__closure__", None):
        raise ValueError("metric function must not close over mutable state")
    payload = marshal.dumps(code)
    payload += repr(getattr(function, "__defaults__", None)).encode()
    payload += repr(getattr(function, "__kwdefaults__", None)).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class MetricSpec:
    metric_id: str
    function: Callable[[Sequence[int], Sequence[float]], float] = field(repr=False, compare=False)
    implementation_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.metric_id or not re.fullmatch(r"[A-Za-z0-9_.:/-]+", self.metric_id):
            raise ValueError("metric_id must be a stable non-empty identifier")
        object.__setattr__(self, "implementation_hash", _callable_hash(self.function))


AUROC_METRIC = MetricSpec("binary_auroc", binary_auroc)


def compute_dataset_manifest_hash(
    partition: Partition | str,
    dataset_id: str,
    sample_ids: Sequence[str],
) -> str:
    """Bind release id, partition, and ordered sample ids without test labels."""

    partition = Partition(partition)
    payload = {
        "schema": "scinumbench-v2-score-manifest/1",
        "dataset_id": dataset_id,
        "partition": partition.value,
        "sample_ids": list(sample_ids),
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _labeled_manifest_hash(dataset_manifest_hash: str, labels: Sequence[int]) -> str:
    """Private/evaluation manifest that additionally commits ordered labels."""

    payload = {
        "schema": "scinumbench-v2-labeled-score-manifest/1",
        "dataset_manifest_hash": dataset_manifest_hash,
        "labels": [int(value) for value in labels],
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _candidate_set_digest(config_items: Sequence[tuple[str, str]]) -> str:
    return hashlib.sha256(
        _canonical_json({"schema": "candidate-set/1", "candidates": list(config_items)})
    ).hexdigest()


@dataclass(frozen=True)
class CandidateScoreSet:
    """Scores bound to a release manifest and complete candidate configs."""

    partition: Partition
    dataset_id: str
    sample_ids: tuple[str, ...]
    labels: tuple[int, ...]
    candidates: Mapping[str, tuple[float, ...]]
    candidate_config_hashes: Mapping[str, str]
    dataset_manifest_hash: str = field(init=False)
    labeled_manifest_hash: str = field(init=False)
    candidate_set_digest: str = field(init=False)

    def __post_init__(self) -> None:
        partition = Partition(self.partition)
        if not isinstance(self.dataset_id, str) or not self.dataset_id.strip():
            raise ValueError("dataset_id must be non-empty")
        sample_ids = tuple(str(value) for value in self.sample_ids)
        labels = tuple(int(value) for value in self.labels)
        candidates = {
            str(name): tuple(float(value) for value in values)
            for name, values in self.candidates.items()
        }
        configs = {str(name): str(value) for name, value in self.candidate_config_hashes.items()}
        if not sample_ids or len(sample_ids) != len(labels):
            raise ValueError("sample_ids and labels must have equal non-zero length")
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("sample_ids must be unique")
        if any(label not in (0, 1) for label in labels) or set(labels) != {0, 1}:
            raise ValueError("score sets must contain both binary labels")
        if not candidates or set(candidates) != set(configs):
            raise ValueError("candidate scores and config hashes must have identical non-empty names")
        for name, values in candidates.items():
            if len(values) != len(labels):
                raise ValueError(f"candidate {name!r} length does not match labels")
            if any(not math.isfinite(value) for value in values):
                raise ValueError(f"candidate {name!r} contains non-finite scores")
            _require_sha256(configs[name], f"config hash for {name!r}")

        config_items = tuple(sorted(configs.items()))
        object.__setattr__(self, "partition", partition)
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "candidates", MappingProxyType(candidates))
        object.__setattr__(self, "candidate_config_hashes", MappingProxyType(configs))
        manifest = compute_dataset_manifest_hash(partition, self.dataset_id, sample_ids)
        object.__setattr__(self, "dataset_manifest_hash", manifest)
        object.__setattr__(self, "labeled_manifest_hash", _labeled_manifest_hash(manifest, labels))
        object.__setattr__(self, "candidate_set_digest", _candidate_set_digest(config_items))


def _selection_protocol_digest(
    *,
    dataset_id: str,
    validation_manifest_hash: str,
    expected_test_manifest_hash: str,
    candidate_configs: Sequence[tuple[str, str]],
    candidate_set_digest: str,
    metric_id: str,
    metric_implementation_hash: str,
    candidate_name: str,
    validation_score: float,
) -> str:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset_id": dataset_id,
        "validation_manifest_hash": validation_manifest_hash,
        "expected_test_manifest_hash": expected_test_manifest_hash,
        "candidate_configs": list(candidate_configs),
        "candidate_set_digest": candidate_set_digest,
        "metric_id": metric_id,
        "metric_implementation_hash": metric_implementation_hash,
        "selected_candidate": candidate_name,
        "validation_score_hex": float(validation_score).hex(),
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


_CHOICE_SEAL = object()


@dataclass(frozen=True, init=False)
class FrozenCandidateChoice:
    """Selection receipt created only by ``select_candidate_on_validation``."""

    candidate_name: str
    validation_score: float
    dataset_id: str
    validation_manifest_hash: str
    expected_test_manifest_hash: str
    candidate_configs: tuple[tuple[str, str], ...]
    candidate_set_digest: str
    metric_id: str
    metric_implementation_hash: str
    protocol_digest: str
    selected_on: Partition
    _metric_spec: MetricSpec = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        _seal: object | None = None,
        candidate_name: str = "",
        validation_score: float = math.nan,
        dataset_id: str = "",
        validation_manifest_hash: str = "",
        expected_test_manifest_hash: str = "",
        candidate_configs: tuple[tuple[str, str], ...] = (),
        candidate_set_digest: str = "",
        metric_spec: MetricSpec | None = None,
        protocol_digest: str = "",
    ) -> None:
        if _seal is not _CHOICE_SEAL:
            raise ProtocolViolation("FrozenCandidateChoice can only be created by validation selection")
        assert metric_spec is not None
        object.__setattr__(self, "candidate_name", candidate_name)
        object.__setattr__(self, "validation_score", float(validation_score))
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "validation_manifest_hash", validation_manifest_hash)
        object.__setattr__(self, "expected_test_manifest_hash", expected_test_manifest_hash)
        object.__setattr__(self, "candidate_configs", tuple(candidate_configs))
        object.__setattr__(self, "candidate_set_digest", candidate_set_digest)
        object.__setattr__(self, "metric_id", metric_spec.metric_id)
        object.__setattr__(self, "metric_implementation_hash", metric_spec.implementation_hash)
        object.__setattr__(self, "protocol_digest", protocol_digest)
        object.__setattr__(self, "selected_on", Partition.VALIDATION)
        object.__setattr__(self, "_metric_spec", metric_spec)


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_name: str
    test_score: float
    metric_id: str
    metric_implementation_hash: str
    validation_manifest_hash: str
    test_manifest_hash: str
    test_labeled_manifest_hash: str
    candidate_config_hash: str
    protocol_digest: str
    selected_on: Partition = field(default=Partition.VALIDATION, init=False)
    evaluated_on: Partition = field(default=Partition.TEST, init=False)


def select_candidate_on_validation(
    validation: CandidateScoreSet,
    *,
    expected_test_manifest_hash: str,
    metric: MetricSpec = AUROC_METRIC,
) -> FrozenCandidateChoice:
    """Select on validation and emit a fully bound, sealed receipt."""

    if validation.partition is not Partition.VALIDATION:
        raise ProtocolViolation(
            f"candidate selection requires validation scores, got {validation.partition.value}"
        )
    _require_sha256(expected_test_manifest_hash, "expected_test_manifest_hash")
    if not isinstance(metric, MetricSpec):
        raise TypeError("metric must be a MetricSpec")
    if _callable_hash(metric.function) != metric.implementation_hash:
        raise ProtocolViolation("metric implementation no longer matches its frozen hash")

    best_name: str | None = None
    best_score = -math.inf
    for name in sorted(validation.candidates):
        score = float(metric.function(validation.labels, validation.candidates[name]))
        if not math.isfinite(score):
            raise ValueError(f"metric returned a non-finite score for {name!r}")
        if score > best_score:
            best_name, best_score = name, score
    assert best_name is not None
    configs = tuple(sorted(validation.candidate_config_hashes.items()))
    digest = _selection_protocol_digest(
        dataset_id=validation.dataset_id,
        validation_manifest_hash=validation.labeled_manifest_hash,
        expected_test_manifest_hash=expected_test_manifest_hash,
        candidate_configs=configs,
        candidate_set_digest=validation.candidate_set_digest,
        metric_id=metric.metric_id,
        metric_implementation_hash=metric.implementation_hash,
        candidate_name=best_name,
        validation_score=best_score,
    )
    return FrozenCandidateChoice(
        _seal=_CHOICE_SEAL,
        candidate_name=best_name,
        validation_score=best_score,
        dataset_id=validation.dataset_id,
        validation_manifest_hash=validation.labeled_manifest_hash,
        expected_test_manifest_hash=expected_test_manifest_hash,
        candidate_configs=configs,
        candidate_set_digest=validation.candidate_set_digest,
        metric_spec=metric,
        protocol_digest=digest,
    )


def evaluate_frozen_on_test(
    choice: FrozenCandidateChoice,
    test: CandidateScoreSet,
    *,
    metric: MetricSpec | None = None,
) -> CandidateEvaluation:
    """Evaluate only the frozen candidate with its frozen metric and configs."""

    if metric is not None:
        raise ProtocolViolation("the evaluation metric is frozen at validation selection")
    if not isinstance(choice, FrozenCandidateChoice):
        raise ProtocolViolation("choice is not a frozen validation receipt")
    if choice.selected_on is not Partition.VALIDATION:
        raise ProtocolViolation("candidate choice was not selected on validation")
    if test.partition is not Partition.TEST:
        raise ProtocolViolation(f"final evaluation requires test scores, got {test.partition.value}")
    if test.dataset_id != choice.dataset_id:
        raise ProtocolViolation("test dataset release differs from the frozen validation release")
    if not hmac.compare_digest(test.dataset_manifest_hash, choice.expected_test_manifest_hash):
        raise ProtocolViolation("test dataset manifest differs from the precommitted manifest")

    test_configs = tuple(sorted(test.candidate_config_hashes.items()))
    if test_configs != choice.candidate_configs:
        raise ProtocolViolation("test candidate names/configurations differ from the frozen set")
    if not hmac.compare_digest(test.candidate_set_digest, choice.candidate_set_digest):
        raise ProtocolViolation("test candidate-set digest differs from the frozen set")
    if choice.candidate_name not in test.candidates:
        raise ProtocolViolation(f"frozen candidate {choice.candidate_name!r} is absent from test")
    if _callable_hash(choice._metric_spec.function) != choice.metric_implementation_hash:
        raise ProtocolViolation("frozen metric implementation hash mismatch")
    if (
        choice._metric_spec.metric_id != choice.metric_id
        or choice._metric_spec.implementation_hash != choice.metric_implementation_hash
    ):
        raise ProtocolViolation("frozen metric identity mismatch")

    expected_digest = _selection_protocol_digest(
        dataset_id=choice.dataset_id,
        validation_manifest_hash=choice.validation_manifest_hash,
        expected_test_manifest_hash=choice.expected_test_manifest_hash,
        candidate_configs=choice.candidate_configs,
        candidate_set_digest=choice.candidate_set_digest,
        metric_id=choice.metric_id,
        metric_implementation_hash=choice.metric_implementation_hash,
        candidate_name=choice.candidate_name,
        validation_score=choice.validation_score,
    )
    if not hmac.compare_digest(expected_digest, choice.protocol_digest):
        raise ProtocolViolation("frozen selection protocol digest mismatch")

    score = float(choice._metric_spec.function(test.labels, test.candidates[choice.candidate_name]))
    return CandidateEvaluation(
        candidate_name=choice.candidate_name,
        test_score=score,
        metric_id=choice.metric_id,
        metric_implementation_hash=choice.metric_implementation_hash,
        validation_manifest_hash=choice.validation_manifest_hash,
        test_manifest_hash=test.dataset_manifest_hash,
        test_labeled_manifest_hash=test.labeled_manifest_hash,
        candidate_config_hash=test.candidate_config_hashes[choice.candidate_name],
        protocol_digest=choice.protocol_digest,
    )


# ---------------------------------------------------------------------------
# Correct reference diagnostic and source-paired bootstrap
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PairedFeatures:
    source_id: Hashable
    original: tuple[float, ...]
    edited: tuple[float, ...]

    def __post_init__(self) -> None:
        original = tuple(float(value) for value in self.original)
        edited = tuple(float(value) for value in self.edited)
        if not original or len(original) != len(edited):
            raise ValueError("original and edited features must have equal non-zero dimension")
        if any(not math.isfinite(value) for value in original + edited):
            raise ValueError("features must be finite")
        object.__setattr__(self, "original", original)
        object.__setattr__(self, "edited", edited)


@dataclass(frozen=True)
class ReferenceDiagnosticRow:
    source_id: Hashable
    label: int
    current: tuple[float, ...]
    reference: tuple[float, ...]
    delta_from_reference: tuple[float, ...]


def build_reference_diagnostic_rows(
    pairs: Iterable[PairedFeatures],
) -> tuple[ReferenceDiagnosticRow, ...]:
    """Use the true original for both rows; the negative delta is zero."""

    rows: list[ReferenceDiagnosticRow] = []
    seen_sources: set[Hashable] = set()
    for pair in pairs:
        if pair.source_id in seen_sources:
            raise ValueError(f"duplicate paired source: {pair.source_id!r}")
        seen_sources.add(pair.source_id)
        rows.append(
            ReferenceDiagnosticRow(
                pair.source_id,
                1,
                pair.edited,
                pair.original,
                tuple(new - old for new, old in zip(pair.edited, pair.original)),
            )
        )
        rows.append(
            ReferenceDiagnosticRow(
                pair.source_id,
                0,
                pair.original,
                pair.original,
                tuple(0.0 for _ in pair.original),
            )
        )
    if not rows:
        raise ValueError("at least one pair is required")
    return tuple(rows)


def _source_sort_key(value: Hashable) -> tuple[str, str]:
    return type(value).__name__, repr(value)


def iter_source_bootstrap_indices(
    source_ids: Sequence[Hashable],
    *,
    n_resamples: int = 1_000,
    seed: int = 12_345,
) -> Iterator[tuple[int, ...]]:
    """Yield paired-source cluster bootstrap row indices."""

    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not source_ids:
        raise ValueError("source_ids must be non-empty")
    grouped: dict[Hashable, list[int]] = {}
    for index, source_id in enumerate(source_ids):
        try:
            grouped.setdefault(source_id, []).append(index)
        except TypeError as exc:
            raise TypeError("source_ids must be hashable") from exc
    if any(len(indices) != 2 for indices in grouped.values()):
        raise ProtocolViolation("paired bootstrap requires exactly two rows per source")
    sources = sorted(grouped, key=_source_sort_key)
    rng = random.Random(seed)
    for _ in range(n_resamples):
        sampled = [sources[rng.randrange(len(sources))] for _ in sources]
        yield tuple(index for source in sampled for index in grouped[source])


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass(frozen=True)
class PairedBootstrapResult:
    point_difference: float
    ci_low: float
    ci_high: float
    confidence: float
    n_resamples: int
    valid_resamples: int
    seed: int
    resampling_unit: str = field(default="source_id", init=False)


def source_paired_bootstrap_difference(
    labels: Sequence[int],
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    source_ids: Sequence[Hashable],
    *,
    metric: Callable[[Sequence[int], Sequence[float]], float] = binary_auroc,
    confidence: float = 0.95,
    n_resamples: int = 1_000,
    seed: int = 12_345,
) -> PairedBootstrapResult:
    """Paired percentile CI for a metric difference, clustered by source."""

    y = tuple(int(value) for value in labels)
    a = tuple(float(value) for value in scores_a)
    b = tuple(float(value) for value in scores_b)
    sources = tuple(source_ids)
    if not (len(y) == len(a) == len(b) == len(sources)) or not y:
        raise ValueError("labels, scores, and source_ids must have equal non-zero length")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    pseudo_records = [
        SplitRecord(str(index), source, "x", label)
        for index, (source, label) in enumerate(zip(sources, y))
    ]
    _require_paired_sources(pseudo_records, "paired bootstrap")

    point = float(metric(y, a)) - float(metric(y, b))
    differences: list[float] = []
    for indices in iter_source_bootstrap_indices(sources, n_resamples=n_resamples, seed=seed):
        sampled_y = tuple(y[index] for index in indices)
        sampled_a = tuple(a[index] for index in indices)
        sampled_b = tuple(b[index] for index in indices)
        try:
            difference = float(metric(sampled_y, sampled_a)) - float(metric(sampled_y, sampled_b))
        except ValueError:
            continue
        if math.isfinite(difference):
            differences.append(difference)
    minimum_valid = max(1, math.ceil(n_resamples * 0.95))
    if len(differences) < minimum_valid:
        raise ProtocolViolation(
            f"only {len(differences)}/{n_resamples} bootstrap replicates were valid; "
            f"need at least {minimum_valid}"
        )
    alpha = (1.0 - confidence) / 2.0
    return PairedBootstrapResult(
        point_difference=point,
        ci_low=_percentile(differences, alpha),
        ci_high=_percentile(differences, 1.0 - alpha),
        confidence=confidence,
        n_resamples=n_resamples,
        valid_resamples=len(differences),
        seed=seed,
    )
