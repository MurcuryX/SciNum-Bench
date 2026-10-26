import json
from pathlib import Path
import tempfile
import unittest

from scinumbench_v2.freeze_manifest import (
    BLIND_TEST_FIELDS,
    ManifestError,
    freeze_manifest,
    load_public_extraction_rows,
    manifest_json,
    verify_manifest,
    write_label_blind_test_inputs,
)


RATIOS = {"train": 0.7, "validation": 0.15, "test": 0.15}
CREATED_AT = "2026-10-26T12:00:00+08:00"
_DEFAULT_PAPER = object()


def _row(
    example_id,
    source_id,
    label,
    text,
    *,
    split=None,
    legacy=False,
    paper_id=_DEFAULT_PAPER,
):
    if "\n" not in text:
        text = f"| Field | Value |\n| --- | --- |\n| fixture | {text} |"
    row = {
        "example_id": example_id,
        "label": label,
        "text": text,
    }
    row["src_table_id" if legacy else "source_id"] = source_id
    if paper_id is _DEFAULT_PAPER:
        row["paper_id"] = f"paper-{source_id}"
    elif paper_id is not None:
        row["paper_id"] = paper_id
    if split is not None:
        row["split"] = split
    return row


class FreezeManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.paths = {name: self.root / f"input-{name}.jsonl" for name in RATIOS}

    def tearDown(self):
        self.temp.cleanup()

    def _write(self, split, rows):
        payload = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        )
        self.paths[split].write_text(payload, encoding="utf-8")

    def _valid_files(self):
        for index, split in enumerate(("train", "validation", "test"), start=1):
            self._write(
                split,
                [
                    _row(f"{split}-n", index, 0, f"| M | V |\n|---|---|\n| A{index} | 1 |", split=split),
                    _row(f"{split}-p", index, 1, f"| M | V |\n|---|---|\n| A{index} | 2 |", split=split),
                ],
            )

    def _freeze(self):
        return freeze_manifest(
            self.paths,
            seed=42,
            ratios=RATIOS,
            code_revision="abc123",
            created_at=CREATED_AT,
        )

    def test_freeze_has_complete_stats_and_no_private_paths(self):
        self._valid_files()
        manifest = self._freeze()

        self.assertEqual(manifest["totals"]["rows"], 6)
        self.assertEqual(manifest["totals"]["sources"], 3)
        self.assertEqual(manifest["totals"]["papers"], 3)
        self.assertEqual(manifest["files"]["train"]["label_counts"], {"0": 1, "1": 1})
        self.assertEqual(manifest["files"]["train"]["paper_count"], 1)
        self.assertEqual(
            manifest["split_policy"]["grouping_keys"],
            ["source_id", "paper_id_or_arxiv_id", "normalized_content_sha256"],
        )
        self.assertEqual(manifest["gates"]["paper_isolation"], "PASS")
        self.assertEqual(manifest["files"]["train"]["logical_path"], "train.jsonl")
        serialized = manifest_json(manifest)
        self.assertNotIn(str(self.root), serialized)
        self.assertTrue(verify_manifest(manifest, self.paths))

    def test_label_blind_test_inputs_are_exactly_bound_and_publicly_loadable(self):
        self._valid_files()
        canonical_paths = {
            split: self.root / f"{split}.jsonl" for split in RATIOS
        }
        for split, path in canonical_paths.items():
            path.write_bytes(self.paths[split].read_bytes())
        blind_path = self.root / "test_inputs.jsonl"
        write_label_blind_test_inputs(canonical_paths["test"], blind_path)
        manifest = freeze_manifest(
            canonical_paths,
            seed=42,
            ratios=RATIOS,
            code_revision="abc123",
            created_at=CREATED_AT,
            test_inputs_path=blind_path,
        )
        self.assertEqual(manifest["test_inputs"]["rows"], 2)
        self.assertFalse(manifest["test_inputs"]["contains_labels"])
        blind_rows = [json.loads(line) for line in blind_path.read_text().splitlines()]
        self.assertTrue(all(set(row) == set(BLIND_TEST_FIELDS) for row in blind_rows))
        self.assertTrue(all("label" not in row for row in blind_rows))
        loaded = load_public_extraction_rows(manifest, self.root)
        self.assertEqual([row["example_id"] for row in loaded["test"]], ["test-n", "test-p"])
        self.assertTrue(
            verify_manifest(manifest, canonical_paths, test_inputs_path=blind_path)
        )

        tampered = dict(blind_rows[0])
        tampered["text"] = str(tampered["text"]).replace("| A3 | 1 |", "| A3 | 9 |")
        blind_path.write_text(
            json.dumps(tampered, sort_keys=True) + "\n"
            + json.dumps(blind_rows[1], sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaises(ManifestError):
            load_public_extraction_rows(manifest, self.root)
        with self.assertRaisesRegex(ManifestError, "do not exactly match"):
            freeze_manifest(
                canonical_paths,
                seed=42,
                ratios=RATIOS,
                code_revision="abc123",
                created_at=CREATED_AT,
                test_inputs_path=blind_path,
            )

    def test_tampered_file_fails_verification(self):
        self._valid_files()
        manifest = self._freeze()
        with self.paths["test"].open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_row("extra", 99, 0, "different")) + "\n")

        with self.assertRaises(ManifestError):
            verify_manifest(manifest, self.paths)

    def test_source_cannot_cross_splits(self):
        self._valid_files()
        validation_rows = [
            _row("validation-n", 1, 0, "validation clean"),
            _row("validation-p", 1, 1, "validation edit"),
        ]
        self._write("validation", validation_rows)

        with self.assertRaisesRegex(ManifestError, "source appears in both"):
            self._freeze()

    def test_paper_cannot_cross_splits_even_when_sources_differ(self):
        self._valid_files()
        self._write(
            "validation",
            [
                _row("validation-n", 2, 0, "validation clean", paper_id="paper-1"),
                _row("validation-p", 2, 1, "validation edit", paper_id="paper-1"),
            ],
        )

        with self.assertRaisesRegex(ManifestError, "paper appears in both"):
            self._freeze()

    def test_paper_identity_is_required_and_aliases_must_agree(self):
        self._valid_files()
        self._write(
            "train",
            [
                _row("train-n", 1, 0, "clean", paper_id=None),
                _row("train-p", 1, 1, "edit", paper_id=None),
            ],
        )
        with self.assertRaisesRegex(ManifestError, "missing paper_id"):
            self._freeze()

        self._valid_files()
        bad = _row("train-n", 1, 0, "clean")
        bad["arxiv_id"] = "different-paper"
        self._write("train", [bad, _row("train-p", 1, 1, "edit")])
        with self.assertRaisesRegex(ManifestError, "paper_id and arxiv_id disagree"):
            self._freeze()

    def test_each_source_requires_exactly_one_twin_per_label(self):
        self._valid_files()
        self._write("train", [_row("train-n", 1, 0, "only clean")])

        with self.assertRaisesRegex(ManifestError, "exactly one negative and one positive"):
            self._freeze()

    def test_duplicate_example_id_is_global_error(self):
        self._valid_files()
        self._write(
            "validation",
            [
                _row("train-n", 2, 0, "validation clean"),
                _row("validation-p", 2, 1, "validation edit"),
            ],
        )

        with self.assertRaisesRegex(ManifestError, "duplicate example_id"):
            self._freeze()

    def test_normalized_content_cannot_cross_splits(self):
        self._valid_files()
        duplicate = "| M | V |\n| --- | --- |\n| A1 | 1 |"
        self._write(
            "validation",
            [
                _row("validation-n", 2, 0, duplicate.upper()),
                _row("validation-p", 2, 1, "validation edit"),
            ],
        )

        with self.assertRaisesRegex(ManifestError, "normalized content also appears"):
            self._freeze()

    def test_legacy_src_table_id_is_accepted(self):
        self._valid_files()
        self._write(
            "train",
            [
                _row("train-n", 1, 0, "legacy clean", legacy=True),
                _row("train-p", 1, 1, "legacy edit", legacy=True),
            ],
        )

        manifest = self._freeze()
        self.assertEqual(manifest["files"]["train"]["source_count"], 1)

    def test_invalid_utf8_and_invalid_json_fail_closed(self):
        self._valid_files()
        self.paths["train"].write_bytes(b"\xff\n")
        with self.assertRaisesRegex(ManifestError, "invalid UTF-8"):
            self._freeze()

        self.paths["train"].write_text("not-json\n", encoding="utf-8")
        with self.assertRaisesRegex(ManifestError, "invalid JSON"):
            self._freeze()

    def test_embedded_split_and_source_alias_must_agree(self):
        self._valid_files()
        bad = _row("train-n", 1, 0, "clean", split="test")
        bad["src_table_id"] = 2
        self._write("train", [bad, _row("train-p", 1, 1, "edit")])

        with self.assertRaises(ManifestError):
            self._freeze()


if __name__ == "__main__":
    unittest.main()

