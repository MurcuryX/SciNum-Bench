"""Archived statistical baselines used for the paper's legacy comparison.

This module extracts the Benford, terminal-digit, and rounding-bias paths from
the earlier experiment tree.  The formulae and decision thresholds are kept
unchanged; filesystem, database, plotting, and unrelated detector code are
intentionally omitted.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy import stats


_RE_MEANSD = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s*(?:±|\\pm|\+/-)\s*(\d+(?:\.\d+)?)\s*\*?\s*$"
)
_RE_CI = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s*[\(\[]\s*(-?\d+(?:\.\d+)?)\s*"
    r"[-–—~,;]\s*(-?\d+(?:\.\d+)?)\s*[\)\]]\s*$"
)
_RE_PVAL = re.compile(r"^\s*([<>]=?|=)?\s*(\d*\.\d+|\d+)\s*(\*+)?\s*$")
_RE_PCT = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*%\s*$")
_RE_COUNTPCT = re.compile(
    r"^\s*(\d+)\s*(?:/\s*(\d+))?\s*[\(\[]\s*(\d+(?:\.\d+)?)\s*%?\s*[\)\]]\s*$"
)
_RE_INT = re.compile(r"^\s*-?\d+\s*$")
_RE_NUM = re.compile(r"^\s*-?\d+\.\d+\s*$")
_RE_PVAL_HEADER = re.compile(
    r"(?i)(p[\s._-]?(value|val|sig|level)?\b|significance|p\s*[<>=])"
)
_NUMTOK = re.compile(r"-?\d+(?:\.\d+)?")
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍﻿  "), None)
_BENFORD = np.array([math.log10(1 + 1.0 / digit) for digit in range(1, 10)])


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).translate(_ZERO_WIDTH).strip()


def classify_cell(raw: Any, *, in_pval_col: bool = False) -> dict[str, Any]:
    """Classify a serialized table cell using the archived parser rules."""

    value = _clean(raw)
    if not value:
        return {"type": "empty", "raw": ""}

    match = _RE_MEANSD.match(value)
    if match:
        return {
            "type": "meansd",
            "raw": value,
            "mean": float(match.group(1)),
            "sd": float(match.group(2)),
        }

    match = _RE_CI.match(value)
    if match:
        return {
            "type": "ci",
            "raw": value,
            "point": float(match.group(1)),
            "lo": float(match.group(2)),
            "hi": float(match.group(3)),
        }

    if in_pval_col:
        match = _RE_PVAL.match(value)
        if match:
            return {
                "type": "pval",
                "raw": value,
                "op": match.group(1) or "=",
                "value": float(match.group(2)),
                "star": bool(match.group(3)),
            }

    match = _RE_PCT.match(value)
    if match:
        return {"type": "pct", "raw": value, "value": float(match.group(1))}

    match = _RE_COUNTPCT.match(value)
    if match and 0.0 <= float(match.group(3)) <= 100.0:
        return {
            "type": "countpct",
            "raw": value,
            "count": int(match.group(1)),
            "denom": int(match.group(2)) if match.group(2) else None,
            "pct": float(match.group(3)),
        }

    if _RE_INT.match(value):
        return {"type": "count", "raw": value, "value": int(value)}
    if _RE_NUM.match(value):
        return {"type": "num", "raw": value, "value": float(value)}
    return {"type": "text", "raw": value}


@dataclass
class ParsedTable:
    """Minimal typed-table representation required by the three baselines."""

    columns: list[str]
    data: list[list[str]]
    caption: str = ""

    def __post_init__(self) -> None:
        self.columns = [_clean(column) for column in self.columns]
        self.data = [[_clean(cell) for cell in row] for row in self.data]
        pval_columns = {
            index
            for index, header in enumerate(self.columns)
            if _RE_PVAL_HEADER.search(header)
        }
        self.cells: list[list[dict[str, Any]]] = []
        for row in self.data:
            typed_row = []
            for index in range(len(self.columns)):
                raw = row[index] if index < len(row) else ""
                typed_row.append(
                    classify_cell(raw, in_pval_col=index in pval_columns)
                )
            self.cells.append(typed_row)


def parse_serialized_table(text: str) -> ParsedTable | None:
    """Parse the archived ``Caption:`` plus Markdown-table serialization.

    This intentionally matches the parser used for the reported legacy rows.
    It expects the benchmark serializer's format; it is not a general Markdown
    parser, and literal, unescaped pipe characters inside cells are unsupported.
    """

    caption = ""
    rows: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Caption:"):
            caption = stripped[len("Caption:") :].strip()
            continue
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if cells and all(cell and set(cell) <= set("-: ") for cell in cells):
            continue
        rows.append(cells)
    if not rows:
        return None
    return ParsedTable(rows[0], rows[1:], caption=caption)


def _numbers_and_last_digits(table: ParsedTable) -> tuple[list[float], list[int]]:
    numbers: list[float] = []
    digits: list[int] = []
    typed = {"meansd", "ci", "pct", "num", "count", "countpct", "pval"}

    for row in table.cells:
        for cell in row:
            cell_type = cell["type"]
            if cell_type == "meansd":
                values: Iterable[float] = (cell["mean"], cell["sd"])
            elif cell_type == "ci":
                values = (cell["point"], cell["lo"], cell["hi"])
            elif cell_type in {"pct", "num", "count", "pval"}:
                values = (cell["value"],)
            elif cell_type == "countpct":
                values = (cell["count"], cell["pct"])
                if cell.get("denom"):
                    values = (*values, cell["denom"])
            else:
                values = tuple(float(token) for token in _NUMTOK.findall(cell["raw"]))
            numbers.extend(float(value) for value in values)

            if cell_type in typed:
                for token in _NUMTOK.findall(cell["raw"]):
                    if token[-1].isdigit():
                        digits.append(int(token[-1]))
    return numbers, digits


def _first_digit(value: float) -> int | None:
    number = abs(float(value))
    if number == 0:
        return None
    while number < 1:
        number *= 10
    while number >= 10:
        number /= 10
    return int(number)


def benford_score(table: ParsedTable | None) -> tuple[float, str]:
    """Return anomaly score ``1-p`` and the archived p<.05 decision."""

    if table is None:
        return 0.0, "PLAUSIBLE"
    numbers, _ = _numbers_and_last_digits(table)
    digits = [digit for value in numbers if (digit := _first_digit(value)) is not None]
    if len(digits) < 20:
        return 0.0, "PLAUSIBLE"
    observed = np.zeros(9)
    for digit in digits:
        observed[digit - 1] += 1
    expected = _BENFORD * len(digits)
    chi_square = float(((observed - expected) ** 2 / expected).sum())
    p_value = 1.0 - stats.chi2.cdf(chi_square, df=8)
    return 1.0 - p_value, "FABRICATED" if p_value < 0.05 else "PLAUSIBLE"


def terminal_digit_score(table: ParsedTable | None) -> tuple[float, str]:
    """Return terminal-digit anomaly score and the archived p<.05 decision."""

    if table is None:
        return 0.0, "PLAUSIBLE"
    _, digits = _numbers_and_last_digits(table)
    if len(digits) < 20:
        return 0.0, "PLAUSIBLE"
    observed = np.zeros(10)
    for digit in digits:
        observed[digit] += 1
    expected = len(digits) / 10.0
    chi_square = float(((observed - expected) ** 2 / expected).sum())
    p_value = 1.0 - stats.chi2.cdf(chi_square, df=9)
    return 1.0 - p_value, "FABRICATED" if p_value < 0.05 else "PLAUSIBLE"


def rounding_score(table: ParsedTable | None) -> tuple[float, str]:
    """Return rounding-bias score and the archived one-sided binomial decision."""

    if table is None:
        return 0.0, "PLAUSIBLE"
    final_digits: list[int] = []
    eligible = {"meansd", "ci", "pct", "num", "countpct"}
    for row in table.cells:
        for cell in row:
            if cell["type"] not in eligible:
                continue
            for token in _NUMTOK.findall(cell["raw"]):
                if "." in token and token[-1].isdigit():
                    final_digits.append(int(token[-1]))
    if len(final_digits) < 20:
        return 0.0, "PLAUSIBLE"
    successes = sum(digit in (0, 5) for digit in final_digits)
    p_value = stats.binomtest(
        successes, len(final_digits), 0.2, alternative="greater"
    ).pvalue
    return 1.0 - p_value, "FABRICATED" if p_value < 0.05 else "PLAUSIBLE"


LEGACY_STATISTICAL_BASELINES = {
    "benford": benford_score,
    "terminal_digit": terminal_digit_score,
    "rounding": rounding_score,
}

