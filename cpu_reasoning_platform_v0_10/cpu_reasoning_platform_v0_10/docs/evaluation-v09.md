# v0.9: evaluation you can defend

v0.9 implements the scientific-validity part of the v0.9 engineering plan (EVAL-01/02/03,
VERIFY-01). The rest of that plan (shared runtime, adaptive controller, performance sweeps,
CI and security hardening, dashboard) waits for the first real results, because its design
depends on them.

## 1. Freeze the splits (once)

```powershell
py -m pip install datasets
py scripts\fetch_gsm8k.py            # data/gsm8k_test.jsonl (200 sample) + data/gsm8k_test_full.jsonl (1319)
py -m cpu_platform.datasets freeze   # data/splits/dev.jsonl, eval.jsonl, manifest.json
py -m cpu_platform.datasets validate
```

- **dev** = the questions in `data/gsm8k_test.jsonl`, the ones pilots run on. Earlier pilot runs stay valid as development data. Tune prompts, thinking budget and checker policy here, as often as needed.
- **eval** = 500 other questions, frozen. Excluded first: exact duplicates, near duplicates (3-word-sequence Jaccard ≥ 0.7, including near duplicates of any dev question), questions echoed in a built-in prompt, and malformed gold answers. When two questions are duplicates, the dev (already used) one is kept.
- `fetch_gsm8k.py` never overwrites an existing `data/gsm8k_test.jsonl`, because that file defines dev.
- Re-freezing needs `--force`. `validate` checks every question, gold answer, id and SHA256.

## 2. The eval guard

As of v0.9.1, Experiment 6 recognizes splits by normalized question-content fingerprints, not IDs alone. New manifests store a fingerprint per question. Existing v0.9 manifests are accepted by computing fingerprints from the original, SHA256-pinned split files. An altered or missing frozen reference file makes the guard fail closed.

| Dataset file | Treated as |
|---|---|
| `data/splits/dev.jsonl` or `data/gsm8k_test.jsonl` (all questions in dev) | dev |
| `data/splits/eval.jsonl`, unchanged | eval: needs `--confirm-frozen-eval` |
| any other file containing even one eval question, including with changed IDs | eval, and **refused** (use the frozen file unchanged) |

A copied question is detected even after changes to row IDs, case, whitespace, or punctuation that the normalizer ignores. Paraphrased questions are **not** reliably detected; this guard does not prove absence of contamination. Do not use `freeze --force` to repair an old manifest: preserve the frozen split and run log. If a frozen split is missing or changed, restore the original file before running benchmarks.

Every eval run is appended to `data/splits/eval_runs.jsonl` *before* it starts (so aborted
runs count), and the run number is stored in the summary. One look is the goal; each extra
look weakens the held-out claim.

## 3. What every run now records

- `servers.<role>`: the GGUF path the server actually loaded, its **SHA256** (cached in `results/model_hashes.json` by path, size and mtime, so a 5 GB file is hashed once), llama.cpp `build_info`, quantization type and chat-template hash, all read from llama.cpp's `/props`.
- `peak_model_rss_mb`: peak memory of each model server (needs psutil and the launch manifest PID).
- Per arm: mean, p50, p95 and max cost per question, and cost per correct answer.
- `per_question_matched` in every comparison against a large model: on each question the arm may use only as many samples as fit within the large model's cost **on that question**. The mean-matched result can hide expensive outliers; this one can't.
- `evidence`: every item is MEASURED, RECORDED, FAILED or **NOT RUN**, never a silent pass. Energy and thermal throttling are NOT RUN (no sensors).

## 4. The checker, v2 (VERIFY-01)

With the question available, the checker also records **ungrounded operands**: numbers used
in an equation that are neither in the question (including number words and allowed unit
constants) nor results of earlier equations. That catches invented numbers (`70 - 36` when
the question says 50) and wrong conversions (`2 * 100` for hours to minutes).

| Policy | A chain may vote if... |
|---|---|
| `exclude_errors` (default) | no checked equation is wrong |
| `strict` | at least one equation checked, none wrong |
| `answer_supported` | no wrong equation, and the answer is the result of a checked equation |
| `grounded` (new) | answer_supported, and every operand is grounded |

Each checked sample carries `verification_evidence`, which says in plain labels what was
established and what was **NOT CHECKED**: operation semantics and problem interpretation are
never checked. The adversarial suite (`tests/test_v09.py`) pins this down. *Right arithmetic on
the wrong operation passes every rule*: a documented limitation, not a hidden one.

## 5. Reports

Each run writes, next to `<output>.jsonl` and `<output>.summary.json`:

- `<output>.failures.jsonl`: every wrong or missing final answer with a heuristic label: `unparseable_output`, `no_answer_after_thinking`, `abstained_checker_filtered_all`, `arithmetic_error`, `ungrounded_numbers`, `wrong_operation_or_interpretation`, `wrong_answer_unexplained`.
- `<output>.report.md`: settings and models, the evidence table, an arm table, matched-cost comparisons (mean and per-question), the checker effect with its policy table, failure counts per arm, and example traces showing exactly what the checker tested.

To regenerate a report from any earlier run, including v0.8 output or an interrupted run:

```powershell
py -m cpu_platform.report results\pilot_v08.jsonl
```
