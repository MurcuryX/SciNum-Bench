# SciNumBench Basic Experiment Code

This anonymous source package contains the code required for the hidden-state
probe experiment and the reusable baseline implementations inherited from the
earlier experiment tree. It intentionally excludes plotting code, paper
sources, model weights, datasets, cached features, saved predictions, logs, and
development history.

## Included code

- `scripts/extract_hidden_features.py`: extract mean-pooled and last-token
  hidden states from a frozen Qwen2.5 checkpoint.
- `scripts/freeze_probe_run_spec.py`: bind the dataset, feature archives,
  model commits, hyperparameter grid, seeds, and one-time test ledger.
- `scripts/run_probe_experiment_v2.py`: fit probes on train, select all
  configurations on validation, and evaluate the sealed test split.
- `scripts/run_basic_experiment.sh`: convenience entry point for the CPU probe
  stage when the released split bundle and frozen feature archives are present.
- `scripts/run_legacy_statistical_baselines.py`: archived Benford,
  terminal-digit, and rounding-bias baselines used in the paper's separate
  legacy comparison.
- `scripts/run_legacy_qwen_zero_shot.py`: archived Qwen first-token Yes/No
  self-judge.
- `scripts/run_legacy_gemma_zero_shot.py`: archived Gemma vLLM Yes/No judge
  with first-token probabilities.
- `scripts/score_legacy_predictions.py`: common ID-aligned AUROC, AUPRC, and
  binary-metric scorer for those predictions.
- `src/scinumbench_v2/`: data-manifest, split, and experiment-protocol logic.
- `src/scinumbench_v2/legacy_*.py`: path-free extraction of the earlier parser,
  statistical detectors, prompt, and verdict parsing logic.
- `tests/`: focused protocol and baseline tests; no plotting tests or figure
  code.

`LEGACY_BASELINES.md` records exactly which earlier source paths each extracted
implementation came from and the limits of that material.

## Environment

The reported CPU experiment used Python 3.11.15. Create an isolated
environment and install the package:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
```

The pinned numerical stack is also listed in `requirements.txt`. Feature
extraction additionally needs the packages in `requirements-gpu.txt` and a
compatible CUDA build of PyTorch 2.12.0; the reported extraction used
PyTorch 2.12.0+cu130.

## Smoke test

The smoke test does not need datasets, model weights, or a GPU:

```bash
python -m pytest -q \
  tests/test_split_protocol.py \
  tests/test_freeze_manifest.py \
  tests/test_experiment_protocol.py \
  tests/test_legacy_baselines.py
```

## Required external inputs

The code-only archive does not duplicate the separately released experiment
inputs. The CPU probe stage expects:

```text
RELEASE_ROOT/
  hard_splits/
    train.jsonl
    validation.jsonl
    test.jsonl
    test_inputs.jsonl
  manifests/
    hard_track_regenerated_manifest.json

FEATURE_ROOT/
  qwen2.5-7b.npz
  qwen2.5-7b.report.json
  qwen2.5-14b.npz
  qwen2.5-14b.report.json
```

The manifest and feature reports are verified before fitting. Test labels are
not stored in the feature archives.

## Reproduce the basic probe experiment

Use a new, empty work directory. The command creates a fresh run
specification and test ledger, searches the pre-specified layer/pooling/C grid
on validation, and writes the test report to `WORK_ROOT/results/`:

```bash
bash scripts/run_basic_experiment.sh RELEASE_ROOT FEATURE_ROOT WORK_ROOT
```

The fixed settings are:

- 7B layers: 14, 21, 28
- 14B layers: 20, 28, 36, 44
- pooling: attention-mask mean and last non-padding token
- inverse regularization: 0.001, 0.01, 0.1, 1, 10
- solver seeds: 13, 29, 42, 67, 101, 137, 173, 211, 251, 307
- grouped-bootstrap replicates: 10,000

Expected outputs include the validation leaderboard, selection receipts,
validation and test predictions, selected probe parameters, experiment report,
and result manifest.

## Optional feature extraction

To regenerate features, run `extract_hidden_features.py` once per model using
the exact model commits printed below. Model weights are not redistributed in
this archive.

```bash
PYTHONPATH=src python scripts/extract_hidden_features.py \
  --split-dir RELEASE_ROOT/hard_splits \
  --manifest RELEASE_ROOT/manifests/hard_track_regenerated_manifest.json \
  --model Qwen/Qwen2.5-7B-Instruct \
  --model-revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --layers 14,21,28 \
  --output FEATURE_ROOT/qwen2.5-7b.npz \
  --report FEATURE_ROOT/qwen2.5-7b.report.json \
  --batch-size 4 --max-length 1024 --device cuda:0 \
  --hf-home PATH_TO_HUGGINGFACE_CACHE
```

For 14B, use `Qwen/Qwen2.5-14B-Instruct`, commit
`cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`, layers
`20,28,36,44`, and the corresponding 14B output names.

## Archived baseline implementations

These implementations reproduce the methods used for the paper's legacy
baseline table. That table uses a separate earlier evaluation collection, so
its values are context only and must not be compared directly with the current
hard-track probe results.

The source-only package does not redistribute that earlier collection or its
saved predictions. The three statistical baselines expect one JSONL file whose
rows contain `example_id` (or `bench_id`) and `text`:

```bash
PYTHONPATH=src python scripts/run_legacy_statistical_baselines.py \
  --input LEGACY_TABLES.jsonl \
  --output-dir LEGACY_OUTPUTS
```

The Qwen runner uses the original first-token probability calculation:

```bash
PYTHONPATH=src python scripts/run_legacy_qwen_zero_shot.py \
  --input LEGACY_TABLES.jsonl \
  --output LEGACY_OUTPUTS/qwen_selfjudge.json \
  --hf-home PATH_TO_HUGGINGFACE_CACHE --offline
```

The Gemma runner requires vLLM in addition to the GPU requirements:

```bash
python -m pip install -r requirements-vllm.txt
PYTHONPATH=src python scripts/run_legacy_gemma_zero_shot.py \
  --input LEGACY_TABLES.jsonl \
  --output LEGACY_OUTPUTS/baseline_preds_gemma3_zeroshot.json \
  --hf-home PATH_TO_HUGGINGFACE_CACHE --offline
```

Score one or more output files against an explicit mapping JSONL:

```bash
PYTHONPATH=src python scripts/score_legacy_predictions.py \
  --labels LEGACY_MAPPING.jsonl \
  --prediction benford=LEGACY_OUTPUTS/baseline_preds_benford.json \
  --prediction qwen=LEGACY_OUTPUTS/qwen_selfjudge.json \
  --output LEGACY_OUTPUTS/metrics.json
```

The earlier project directory did not contain a standard Claude inference
program. It contained only saved Claude predictions and an internal workflow
that was tied to private orchestration infrastructure and did not fix a
complete public model configuration. Neither that workflow nor its outputs are
included here. Accordingly, the Claude row in the paper is not claimed as
end-to-end reproducible from this code-only archive.

