from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scinumbench_v2.split_protocol import (  # noqa: E402
    CandidateScoreSet,
    FrozenCandidateChoice,
    LeakageError,
    MetricSpec,
    PairedFeatures,
    Partition,
    ProtocolViolation,
    SplitRecord,
    assert_no_split_leakage,
    binary_auroc,
    build_reference_diagnostic_rows,
    candidate_config_hash,
    connected_split_groups,
    content_hash,
    deterministic_group_split,
    evaluate_frozen_on_test,
    iter_source_bootstrap_indices,
    normalize_table_text,
    select_candidate_on_validation,
    source_paired_bootstrap_difference,
    validate_split_bundle,
)


def table(header: tuple[str, str], row: tuple[str, str], *, caption: str = "") -> str:
    prefix = f"Caption: {caption}\n\n" if caption else ""
    return prefix + f"| {header[0]} | {header[1]} |\n| --- | --- |\n| {row[0]} | {row[1]} |"


def alternative_metric(labels, scores):
    return 1.0 - binary_auroc(labels, scores)


def score_set(partition, candidates, configs, *, dataset_id="release-v2"):
    return CandidateScoreSet(
        partition=partition,
        dataset_id=dataset_id,
        sample_ids=(
            f"{partition.value}-0",
            f"{partition.value}-1",
            f"{partition.value}-2",
            f"{partition.value}-3",
        ),
        labels=(0, 0, 1, 1),
        candidates=candidates,
        candidate_config_hashes={name: candidate_config_hash(configs[name]) for name in candidates},
    )


class StructuralContentHashTests(unittest.TestCase):
    def test_caption_outer_pipe_and_alignment_differences_do_not_change_hash(self):
        left = (
            "Caption: first caption\n\n"
            "| Method | Score |\n"
            "| :--- | ---: |\n"
            "| A | 1.0 |"
        )
        right = (
            "Caption: unrelated wording with | a pipe\n\n"
            "Method | Score\n"
            "--- | :---\n"
            "A | 1.0\n"
            "\nmore caption prose"
        )
        self.assertEqual(normalize_table_text(left), normalize_table_text(right))
        self.assertEqual(content_hash(left), content_hash(right))
        self.assertNotEqual(content_hash(left), content_hash(right.replace("1.0", "1.00")))

    def test_grid_shape_is_part_of_hash(self):
        one_row = "|A|B|\n|---|---|\n|x|1|"
        two_rows = one_row + "\n|y|2|"
        self.assertNotEqual(content_hash(one_row), content_hash(two_rows))

    def test_parse_failure_never_falls_back_to_raw_text(self):
        with self.assertRaisesRegex(ProtocolViolation, "separator"):
            content_hash("| A | B |\n| x | 1 |")
        with self.assertRaisesRegex(ProtocolViolation, "rectangular"):
            content_hash("| A | B |\n| --- | --- |\n| x | 1 | extra |")

    def test_source_and_content_relations_are_transitively_closed(self):
        records = (
            SplitRecord("a-original", 1, table(("M", "X"), ("A", "1"), caption="one"), 0),
            SplitRecord("a-edited", 1, table(("M", "X"), ("A", "2")), 1),
            SplitRecord("b-duplicate", 2, table(("m", "x"), ("a", "1"), caption="two"), 0),
            SplitRecord("b-edited", 2, table(("M", "X"), ("B", "3")), 1),
            SplitRecord("c-original", 3, table(("M", "X"), ("C", "7")), 0),
            SplitRecord("c-edited", 3, table(("M", "X"), ("C", "8")), 1),
        )
        group_ids = [{row.example_id for row in group} for group in connected_split_groups(records)]
        self.assertIn({"a-original", "a-edited", "b-duplicate", "b-edited"}, group_ids)
        self.assertIn({"c-original", "c-edited"}, group_ids)

    def test_distinct_sources_from_same_paper_form_one_component(self):
        records = (
            SplitRecord("a-neg", 11, table(("M", "X"), ("A", "1")), 0, "paper-a"),
            SplitRecord("a-pos", 11, table(("M", "X"), ("A", "2")), 1, "paper-a"),
            SplitRecord("b-neg", 12, table(("M", "X"), ("B", "3")), 0, "paper-a"),
            SplitRecord("b-pos", 12, table(("M", "X"), ("B", "4")), 1, "paper-a"),
            SplitRecord("c-neg", 13, table(("M", "X"), ("C", "5")), 0, "paper-b"),
            SplitRecord("c-pos", 13, table(("M", "X"), ("C", "6")), 1, "paper-b"),
        )
        groups = connected_split_groups(records)
        self.assertEqual(len(groups), 2)
        self.assertIn(
            {"a-neg", "a-pos", "b-neg", "b-pos"},
            [{row.example_id for row in group} for group in groups],
        )


class SplitTests(unittest.TestCase):
    @staticmethod
    def records():
        rows = []
        for source in range(12):
            negative_value = "0" if source == 11 else str(source)
            negative_method = "x0" if source == 11 else f"x{source}"
            rows.append(
                SplitRecord(
                    f"s{source}-neg",
                    source,
                    table(("m", "v"), (negative_method, negative_value), caption=f"neg {source}"),
                    0,
                )
            )
            rows.append(
                SplitRecord(
                    f"s{source}-pos",
                    source,
                    table(("m", "v"), (f"x{source}", f"{source}.1"), caption=f"pos {source}"),
                    1,
                )
            )
        return tuple(rows)

    def test_deterministic_stratified_three_way_split_has_complete_pairs(self):
        first = deterministic_group_split(self.records(), seed=77)
        second = deterministic_group_split(self.records(), seed=77)
        self.assertEqual(
            {key: tuple(row.example_id for row in value) for key, value in first.items()},
            {key: tuple(row.example_id for row in value) for key, value in second.items()},
        )
        report = validate_split_bundle(first)
        self.assertTrue(report.ok, report.errors)
        for split in ("train", "validation", "test"):
            self.assertGreater(report.label_counts[split][0], 0)
            self.assertEqual(report.label_counts[split][0], report.label_counts[split][1])
        assert_no_split_leakage(first)

    def test_many_single_paper_components_follow_integer_paper_quotas(self):
        rows = []
        source = 0
        for paper in range(40):
            # Some papers contribute several table pairs.  They still count as
            # one allocation unit and may never be split across partitions.
            source_count = 3 if paper % 10 == 0 else 1
            for table_index in range(source_count):
                source += 1
                rows.extend(
                    (
                        SplitRecord(
                            f"p{paper}-s{source}-neg",
                            source,
                            table(("m", "v"), (f"p{paper}t{table_index}", "clean")),
                            0,
                            f"paper-{paper}",
                        ),
                        SplitRecord(
                            f"p{paper}-s{source}-pos",
                            source,
                            table(("m", "v"), (f"p{paper}t{table_index}", "edited")),
                            1,
                            f"paper-{paper}",
                        ),
                    )
                )

        split = deterministic_group_split(tuple(rows), seed=123)
        report = validate_split_bundle(split)
        self.assertTrue(report.ok, report.errors)
        self.assertEqual(dict(report.paper_counts), {"train": 28, "validation": 6, "test": 6})
        # Four papers carry three sources instead of one.  After fixing paper
        # counts exactly, source mass must still stay close to 70/15/15 rather
        # than concentrating all multi-source papers in the small splits.
        self.assertGreaterEqual(report.source_counts["train"], 32)
        self.assertLessEqual(report.source_counts["train"], 36)
        self.assertGreaterEqual(report.source_counts["validation"], 6)
        self.assertLessEqual(report.source_counts["validation"], 9)
        self.assertGreaterEqual(report.source_counts["test"], 6)
        self.assertLessEqual(report.source_counts["test"], 9)
        paper_owners = {}
        for split_name, split_rows in split.items():
            for row in split_rows:
                previous = paper_owners.setdefault(row.paper_id, split_name)
                self.assertEqual(previous, split_name)

    def test_manual_content_leak_is_rejected_with_other_contracts_valid(self):
        bundle = {
            "train": (
                SplitRecord("tr-n", 1, table(("a", "b"), ("c", "1"), caption="train"), 0),
                SplitRecord("tr-p", 1, table(("a", "b"), ("c", "2")), 1),
            ),
            "validation": (
                SplitRecord("va-n", 2, table(("a", "b"), ("d", "3")), 0),
                SplitRecord("va-p", 2, table(("a", "b"), ("d", "4")), 1),
            ),
            "test": (
                SplitRecord("te-n", 3, table(("A", "B"), ("C", "1"), caption="test"), 0),
                SplitRecord("te-p", 3, table(("a", "b"), ("e", "5")), 1),
            ),
        }
        report = validate_split_bundle(bundle)
        self.assertFalse(report.ok)
        self.assertTrue(any("content_hash overlap" in error for error in report.errors))
        with self.assertRaises(LeakageError):
            assert_no_split_leakage(bundle)

    def test_source_must_be_exact_pair_and_every_split_has_both_labels(self):
        bundle = {
            "train": (
                SplitRecord("tr-n1", 1, table(("m", "v"), ("a", "1")), 0),
                SplitRecord("tr-n2", 1, table(("m", "v"), ("a", "2")), 0),
            ),
            "validation": (
                SplitRecord("va-n", 2, table(("m", "v"), ("b", "1")), 0),
                SplitRecord("va-p", 2, table(("m", "v"), ("b", "2")), 1),
            ),
            "test": (
                SplitRecord("te-n", 3, table(("m", "v"), ("c", "1")), 0),
                SplitRecord("te-p", 3, table(("m", "v"), ("c", "2")), 1),
            ),
        }
        report = validate_split_bundle(bundle)
        self.assertFalse(report.ok)
        self.assertTrue(any("train must contain both labels" in error for error in report.errors))
        self.assertTrue(any("exactly one positive and one negative" in error for error in report.errors))

    def test_manual_paper_leak_with_distinct_sources_is_rejected(self):
        bundle = {
            "train": (
                SplitRecord("tr-n", 1, table(("m", "v"), ("a", "1")), 0, "paper-x"),
                SplitRecord("tr-p", 1, table(("m", "v"), ("a", "2")), 1, "paper-x"),
            ),
            "validation": (
                SplitRecord("va-n", 2, table(("m", "v"), ("b", "3")), 0, "paper-y"),
                SplitRecord("va-p", 2, table(("m", "v"), ("b", "4")), 1, "paper-y"),
            ),
            "test": (
                SplitRecord("te-n", 3, table(("m", "v"), ("c", "5")), 0, "paper-x"),
                SplitRecord("te-p", 3, table(("m", "v"), ("c", "6")), 1, "paper-x"),
            ),
        }
        report = validate_split_bundle(bundle)
        self.assertFalse(report.ok)
        self.assertTrue(any("paper_id overlap" in error for error in report.errors))
        with self.assertRaises(LeakageError):
            assert_no_split_leakage(bundle)


class FrozenSelectionTests(unittest.TestCase):
    def setUp(self):
        self.configs = {
            "validation-winner": {"kind": "probability_mean", "models": ["7b", "14b"]},
            "test-winner": {"kind": "feature_concat", "models": ["7b", "14b"]},
        }
        self.validation = score_set(
            Partition.VALIDATION,
            {
                "validation-winner": (0.1, 0.2, 0.8, 0.9),
                "test-winner": (0.9, 0.8, 0.2, 0.1),
            },
            self.configs,
        )
        self.test = score_set(
            Partition.TEST,
            {
                "validation-winner": (0.9, 0.8, 0.2, 0.1),
                "test-winner": (0.1, 0.2, 0.8, 0.9),
            },
            self.configs,
        )

    def choice(self):
        return select_candidate_on_validation(
            self.validation,
            expected_test_manifest_hash=self.test.dataset_manifest_hash,
        )

    def test_validation_choice_remains_frozen_when_other_candidate_wins_test(self):
        choice = self.choice()
        self.assertEqual(choice.candidate_name, "validation-winner")
        self.assertEqual(choice.validation_score, 1.0)
        self.assertEqual(choice.validation_manifest_hash, self.validation.labeled_manifest_hash)
        self.assertEqual(len(choice.candidate_configs), 2)
        result = evaluate_frozen_on_test(choice, self.test)
        self.assertEqual(result.candidate_name, "validation-winner")
        self.assertEqual(result.test_score, 0.0)
        with self.assertRaisesRegex(ProtocolViolation, "requires validation"):
            select_candidate_on_validation(
                self.test,
                expected_test_manifest_hash=self.test.dataset_manifest_hash,
            )

    def test_choice_cannot_be_publicly_forged(self):
        with self.assertRaisesRegex(ProtocolViolation, "only be created"):
            FrozenCandidateChoice(candidate_name="test-winner", validation_score=1.0)

    def test_digest_detects_post_construction_tampering(self):
        choice = self.choice()
        # frozen=True blocks ordinary assignment; this simulates a hostile
        # object-level bypass and verifies the digest is an independent gate.
        object.__setattr__(choice, "candidate_name", "test-winner")
        with self.assertRaisesRegex(ProtocolViolation, "protocol digest mismatch"):
            evaluate_frozen_on_test(choice, self.test)

    def test_same_candidate_name_with_different_formula_is_rejected(self):
        changed = dict(self.configs)
        changed["validation-winner"] = {
            "kind": "probability_mean",
            "models": ["7b", "14b"],
            "weights": [0.9, 0.1],
        }
        incompatible_test = score_set(Partition.TEST, self.test.candidates, changed)
        choice = select_candidate_on_validation(
            self.validation,
            expected_test_manifest_hash=incompatible_test.dataset_manifest_hash,
        )
        with self.assertRaisesRegex(ProtocolViolation, "configurations differ"):
            evaluate_frozen_on_test(choice, incompatible_test)

    def test_evaluation_cannot_substitute_another_metric(self):
        other = MetricSpec("inverse_auroc", alternative_metric)
        with self.assertRaisesRegex(ProtocolViolation, "metric is frozen"):
            evaluate_frozen_on_test(self.choice(), self.test, metric=other)

    def test_wrong_test_manifest_is_rejected(self):
        another_test = CandidateScoreSet(
            partition=Partition.TEST,
            dataset_id="release-v2",
            sample_ids=("other-0", "other-1", "other-2", "other-3"),
            labels=self.test.labels,
            candidates=self.test.candidates,
            candidate_config_hashes=self.test.candidate_config_hashes,
        )
        with self.assertRaisesRegex(ProtocolViolation, "precommitted manifest"):
            evaluate_frozen_on_test(self.choice(), another_test)

    def test_auroc_ties_use_average_ranks(self):
        self.assertEqual(binary_auroc((0, 1), (0.5, 0.5)), 0.5)


class ReferenceDiagnosticTests(unittest.TestCase):
    def test_negative_uses_its_true_original_and_zero_delta(self):
        positive, negative = build_reference_diagnostic_rows(
            [PairedFeatures(source_id=9, original=(1.0, 2.0), edited=(1.25, 1.5))]
        )
        self.assertEqual(positive.label, 1)
        self.assertEqual(positive.reference, (1.0, 2.0))
        self.assertEqual(positive.delta_from_reference, (0.25, -0.5))
        self.assertEqual(negative.label, 0)
        self.assertEqual(negative.current, negative.reference)
        self.assertEqual(negative.delta_from_reference, (0.0, 0.0))
        self.assertNotEqual(negative.reference, positive.current)


class SourcePairedBootstrapTests(unittest.TestCase):
    def test_resampling_keeps_both_rows_of_each_source_together(self):
        source_ids = (1, 1, 2, 2, 3, 3)
        samples = list(iter_source_bootstrap_indices(source_ids, n_resamples=8, seed=5))
        self.assertEqual(samples, list(iter_source_bootstrap_indices(source_ids, n_resamples=8, seed=5)))
        for indices in samples:
            self.assertEqual(len(indices), 6)
            for first, second in ((0, 1), (2, 3), (4, 5)):
                self.assertEqual(indices.count(first), indices.count(second))

    def test_bootstrap_rejects_non_pairs(self):
        with self.assertRaisesRegex(ProtocolViolation, "two rows per source"):
            list(iter_source_bootstrap_indices((1, 1, 1, 2, 2), n_resamples=1))

    def test_paired_difference_ci_is_deterministic(self):
        labels = (0, 1, 0, 1, 0, 1, 0, 1)
        source_ids = (10, 10, 11, 11, 12, 12, 13, 13)
        scores_a = (0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6)
        scores_b = (0.4, 0.6, 0.5, 0.5, 0.6, 0.4, 0.7, 0.3)
        first = source_paired_bootstrap_difference(
            labels, scores_a, scores_b, source_ids, n_resamples=200, seed=91
        )
        second = source_paired_bootstrap_difference(
            labels, scores_a, scores_b, source_ids, n_resamples=200, seed=91
        )
        self.assertEqual(first, second)
        self.assertEqual(first.resampling_unit, "source_id")
        self.assertEqual(first.valid_resamples, 200)
        self.assertGreater(first.point_difference, 0.0)
        self.assertLessEqual(first.ci_low, first.point_difference)
        self.assertGreaterEqual(first.ci_high, first.point_difference)


if __name__ == "__main__":
    unittest.main()
