#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 RELEASE_ROOT FEATURE_ROOT WORK_ROOT" >&2
  exit 2
fi

release_root=$1
feature_root=$2
work_root=$3
project_root=$(cd "$(dirname "$0")/.." && pwd)
python_bin=${PYTHON:-python3}

split_dir="$release_root/hard_splits"
manifest="$release_root/manifests/hard_track_regenerated_manifest.json"
run_spec="$work_root/run_spec.json"
ledger_root="$work_root/test_ledger"
output_dir="$work_root/results"
study_id="scinumbench-hard-probe-reproduction"

required_files=(
  "$manifest"
  "$split_dir/train.jsonl"
  "$split_dir/validation.jsonl"
  "$split_dir/test.jsonl"
  "$split_dir/test_inputs.jsonl"
  "$feature_root/qwen2.5-7b.npz"
  "$feature_root/qwen2.5-7b.report.json"
  "$feature_root/qwen2.5-14b.npz"
  "$feature_root/qwen2.5-14b.report.json"
)
for required_file in "${required_files[@]}"; do
  if [[ ! -f "$required_file" ]]; then
    echo "missing required input: $required_file" >&2
    exit 2
  fi
done

if [[ -e "$run_spec" || -e "$output_dir" ]]; then
  echo "WORK_ROOT must not contain run_spec.json or results/" >&2
  exit 2
fi

mkdir -p "$work_root" "$ledger_root"
export PYTHONPATH="$project_root/src"
created_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)

"$python_bin" "$project_root/scripts/freeze_probe_run_spec.py" \
  --project-root "$project_root" \
  --split-dir "$split_dir" \
  --manifest "$manifest" \
  --feature "qwen7b=$feature_root/qwen2.5-7b.npz" \
  --feature "qwen14b=$feature_root/qwen2.5-14b.npz" \
  --feature-report "qwen7b=$feature_root/qwen2.5-7b.report.json" \
  --feature-report "qwen14b=$feature_root/qwen2.5-14b.report.json" \
  --expected-model qwen7b=Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28 \
  --expected-model qwen14b=Qwen/Qwen2.5-14B-Instruct@cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8 \
  --study-id "$study_id" \
  --test-ledger-root "$ledger_root" \
  --seeds 13,29,42,67,101,137,173,211,251,307 \
  --c-grid 0.001,0.01,0.1,1,10 \
  --max-iter 2000 \
  --bootstrap-replicates 10000 \
  --bootstrap-seed 20261026 \
  --created-at "$created_at" \
  --output "$run_spec"

run_spec_sha=$(sha256sum "$run_spec" | awk '{print $1}')

"$python_bin" "$project_root/scripts/run_probe_experiment_v2.py" \
  --split-dir "$split_dir" \
  --manifest "$manifest" \
  --feature "qwen7b=$feature_root/qwen2.5-7b.npz" \
  --feature "qwen14b=$feature_root/qwen2.5-14b.npz" \
  --feature-report "qwen7b=$feature_root/qwen2.5-7b.report.json" \
  --feature-report "qwen14b=$feature_root/qwen2.5-14b.report.json" \
  --expected-model qwen7b=Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28 \
  --expected-model qwen14b=Qwen/Qwen2.5-14B-Instruct@cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8 \
  --run-spec "$run_spec" \
  --run-spec-sha256 "$run_spec_sha" \
  --study-id "$study_id" \
  --test-ledger-root "$ledger_root" \
  --output-dir "$output_dir" \
  --seeds 13,29,42,67,101,137,173,211,251,307 \
  --c-grid 0.001,0.01,0.1,1,10 \
  --max-iter 2000 \
  --bootstrap-replicates 10000 \
  --bootstrap-seed 20261026

echo "results written to $output_dir"
