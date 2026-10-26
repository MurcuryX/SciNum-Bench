import json

from scinumbench_v2.legacy_baselines import (
    benford_score,
    parse_serialized_table,
    rounding_score,
    terminal_digit_score,
)
from scinumbench_v2.legacy_zero_shot import parse_yes_no


def test_serialized_table_and_short_table_fallbacks():
    table = parse_serialized_table(
        "Caption: demo\n| Method | Score |\n| --- | --- |\n| A | 1.25 |\n"
    )
    assert table is not None
    assert table.caption == "demo"
    assert benford_score(table) == (0.0, "PLAUSIBLE")
    assert terminal_digit_score(table) == (0.0, "PLAUSIBLE")
    assert rounding_score(table) == (0.0, "PLAUSIBLE")


def test_yes_no_parser_matches_legacy_fallback():
    assert parse_yes_no("Yes") == "FABRICATED"
    assert parse_yes_no("No.") == "PLAUSIBLE"
    assert parse_yes_no("The values are inconsistent") == "FABRICATED"
    assert parse_yes_no("unclear") == "PLAUSIBLE"


def test_large_terminal_bias_is_flagged():
    values = " | ".join(str(index * 10) for index in range(1, 31))
    headers = " | ".join(f"C{index}" for index in range(30))
    separators = " | ".join("---" for _ in range(30))
    table = parse_serialized_table(
        f"| {headers} |\n| {separators} |\n| {values} |\n"
    )
    assert table is not None
    _, verdict = terminal_digit_score(table)
    assert verdict == "FABRICATED"

