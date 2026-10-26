# Legacy baseline source provenance

The paper retains a small comparison table from an earlier experiment
collection. This package includes only the source needed to inspect or rerun
the corresponding reusable methods. The older project directory was not copied
wholesale because it also contains plotting programs, obsolete experiments,
database-bound utilities, local paths, and development artifacts.

## Included dependency closure

| Paper method | Earlier source | Packaged replacement |
|---|---|---|
| Benford first digit | `run_baselines_newtest.py`, `parser.py`, `llmfraud_common.py`, `detect_llmfraud.py` | `legacy_baselines.py`, `run_legacy_statistical_baselines.py` |
| Terminal digit | same statistical path | `legacy_baselines.py`, `run_legacy_statistical_baselines.py` |
| Rounding bias | same statistical path | `legacy_baselines.py`, `run_legacy_statistical_baselines.py` |
| Gemma zero-shot | `judge_gemma.py`, `judge_gemma_prob.py` | `legacy_zero_shot.py`, `run_legacy_gemma_zero_shot.py` |
| Qwen self-judge | `judge_gemma.py`, `qwen_selfjudge.py` | `legacy_zero_shot.py`, `run_legacy_qwen_zero_shot.py` |
| Metric calculation | `score_all_baselines.py` and Qwen's embedded scorer | `score_legacy_predictions.py` |

The extraction preserves the numeric statistics, minimum-sample rules,
decision thresholds, zero-shot instruction, chat-template use, Yes/No mapping,
and first-token probability calculations. Paths and inputs are command-line
arguments. The statistical runner no longer opens the old experiment database
or runs unrelated rule/dominance detectors.

## Deliberately excluded

- every plotting and figure-generation program;
- paper sources, PDFs, datasets, model weights, feature archives, and saved
  predictions;
- obsolete easy-track baselines and unrelated ablations;
- the old database-bound export/scoring utilities;
- caches, logs, backups, temporary files, and absolute local paths.

## Known boundary

No standard Claude inference source existed in the earlier paper directory.
The saved Claude row therefore cannot be regenerated from this source package.
The generic scorer can audit compatible saved predictions when supplied
separately, and reports missing and extra example IDs explicitly.
