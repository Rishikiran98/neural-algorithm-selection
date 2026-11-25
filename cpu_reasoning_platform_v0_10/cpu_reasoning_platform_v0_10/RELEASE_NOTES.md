# v0.10.0 release notes

New: local multi-domain Q&A endpoint (`/v1/ask`), conservative arithmetic checking with a bounded repair, Python syntax-only inspection, complete bounded propositional logic checker (`/v1/verify-logic`), SDK support, browser panels, and an explicitly limited cross-domain smoke evaluation.

All v0.9.1 modules, tests, dataset safety checks and the existing arithmetic/job endpoints are retained. New work does not execute arbitrary code, fetch live facts or prove general intelligence. The old 0.9.1 Q&A limitation has been reduced at the application layer, but the local Qwen model remains responsible for answer quality.

See `docs/general-v010.md` and the v0.9.1 documentation retained in the README.


# Release notes: v0.9.1

## Frozen evaluation guard correctness patch

**Fixed:** v0.9 could classify a copied frozen evaluation question as an unrelated dataset when its JSONL row ID was changed. Version 0.9.1 fingerprints normalized question text (SHA256) and checks copied, reordered or merged datasets. Any matching evaluation question triggers the frozen-evaluation guard. Only the original SHA256-matched evaluation file can run with explicit `--confirm-frozen-eval`.

- New split manifests include per-question fingerprints for both dev and eval, and `validate` checks them, as well as normalized question overlap across splits.
- Existing v0.9 manifests (schema v1) work unchanged: the guard derives hashes from original SHA256-pinned split JSONL files. No re-freeze or eval-log reset required.
- Missing, altered or invalid frozen references now fail closed with an actionable CLI error before contacting models.
- Matching a row ID without matching its question text is no longer considered an evaluation leak by itself.
- Added 10 regression tests for copied questions with altered IDs, mixed and normalized text, legacy manifests, corrupted references, and actual experiment CLI rejection.
- All earlier v0.9 platform features remain in the package. There are **no new real-model benchmark claims** from this patch.
- Limit: exact normalized question matching does not detect paraphrases, and GSM8K may have appeared in a model's pretraining data.

## Prior v0.9.0 release notes

# Release notes: v0.9.0

Implements the scientific-validity priorities of the v0.9 engineering plan (EVAL-01, EVAL-02, EVAL-03, VERIFY-01). The runtime, controller, performance, CI, security and UX items are deferred until real pilot results exist, because their design depends on them.

## Added

- `cpu_platform/datasets.py` (EVAL-01): `freeze`, `validate`, `status`. Dev = the pilot questions; eval = 500 frozen questions after removing exact and near duplicates (3-word shingles, Jaccard ≥ 0.7, including near duplicates of dev), prompt echoes and malformed gold answers. When two questions are duplicates the dev one is kept. The manifest records source SHA256 and provenance, ids, per-split SHA256 and removed ids. The eval-run log is written before each run. `split_of()` matches by content and by question id, so any file containing an eval question is treated as eval.
- `scripts/fetch_gsm8k.py` also writes the full 1319-question test set and `data/gsm8k_source.json` (dataset id, fingerprint, versions), and never overwrites the existing pilot sample.
- `cpu_platform/provenance.py` (EVAL-02): reads llama.cpp `/props` (model_path, build_info, model_ftype, chat_template), hashes the loaded GGUF with a cache, and builds the evidence table (MEASURED / RECORDED / FAILED / NOT RUN).
- Experiment 6: `--confirm-frozen-eval` guard (exit 6), split info and eval run number in the summary, `servers`, `peak_model_rss_mb`, `evidence`, p50/p95/max cost and cost per correct for every arm, and `per_question_matched` in every comparison against a large model. Writes the report bundle (`--no-report` to skip).
- Checker v2 (VERIFY-01): `audit_reasoning(text, question)` records ungrounded operands (numbers from neither the question nor earlier results), a new `grounded` policy, and `Audit.evidence()` with explicit NOT CHECKED labels for operation semantics and interpretation. `check_cot_samples(..., question)` stores `verification_evidence`.
- `cpu_platform/report.py` (EVAL-03): heuristic failure labels, `*.failures.jsonl`, `*.report.md` with example traces; works on v0.8 output and on interrupted runs.
- `docs/evaluation-v09.md`; `tests/test_v09.py` (16 tests: splits, eval guard and leak detection, model hashing, per-question matching, 8 adversarial verifier fixtures, report). The fake server now serves `/props`. Full suite: 205 tests.

## Changed

- Policy dictionaries now include `grounded`. It is False whenever the question isn't supplied, since grounding can't be checked without it.

# Release notes: v0.8.0

Built on v0.7.0 (paired CoT voting and the arithmetic checker).

## Added

- **Native thinking with a token budget.** Search kind `think`: each request sets `chat_template_kwargs.enable_thinking` and `reasoning_budget_tokens` (both read by llama.cpp's `server-common.cpp`). When the budget runs out the server forces the end of thinking and the model must answer. The thinking text (`reasoning_content`, or embedded `<think>` blocks) is stored per sample and audited by the checker. If the reply has no JSON, the last JSON answer inside the thinking is used.
- Experiment arms `small_think`, `small_think_vote`, `small_think_checked`, `large_think`; options `--think-samples` (4), `--think-budget` (512 tokens), `--check-policy`.
- **Generalized pairing**: every `*_checked` arm re-scores exactly the same samples as its `*_vote` twin (no extra model calls), for both prose reasoning and thinking. `paired_think_arithmetic_filter_effect` joins the v0.7 `paired_cot_arithmetic_filter_effect`, and both include a `policy_table`.
- **Checker policies** `exclude_errors` (new default: excluded only on evidence of a wrong equation), `strict` (the v0.7 rule) and `answer_supported`. All are stored per sample (`verification_policies`).
- **Checker normalization**: thousands commas (v0.7 read `1,200 * 2 = 2,400` as a false error), `$` amounts (missed in v0.7), LaTeX `\times` `\div` `\cdot` `\frac` `\( \)`, letter `x`, `N% of M` and `N%`, markdown, and rounding to the stated precision or with `≈`.
- Thinking preflight: confirms thinking switches on and off per request and that the budget is enforced, using its own question so it never overlaps a dataset item.
- `scripts/run_windows_pilot.ps1`, `scripts/run_linux_pilot.sh`, `docs/benchmark8.md`, `tests/test_v08.py` (14 tests). The fake server now writes checkable equations, with arithmetic slips on its wrong answers. Full suite: 189 tests.

## Changed

- Experiment 6 defaults: dataset `data/gsm8k_test.jsonl`, 100 questions; arms small_cot, small_think, large_cot, small_cot_vote, small_cot_checked, small_think_vote, small_think_checked. Missing dataset -> exit with fetch instructions. `scripts/fetch_gsm8k.py` now fetches 200 by default.
- Launcher no longer passes `--reasoning off` (it passes `--reasoning auto` where supported): thinking is decided per request. The client still sends `/no_think` to Qwen3 whenever thinking is off.
- Two v0.7 tests updated for deliberate changes: `10% of 240 = 24` is now checked, and the default policy no longer excludes chains just because no equation was found (the v0.7 behaviour remains as `--check-policy strict`).

# Release notes: v0.7.0

**Research pivot based on the first live v0.6 run:** on five bundled questions,
free-form CoT got 5/5 while grammar-constrained plans got 1/5. The grammar helped
format and cost, not the correctness of selected operations. This is an exploratory
observation, not a general benchmark result.

- Added `cpu_platform/cot_check.py`: conservative bounded arithmetic-equality
  checks on free-form visible text, using exact `Fraction` arithmetic and the
  existing AST evaluator. Generated content is never executed.
- Added matched arms `small_cot_vote` and `small_cot_checked` to Experiment 6.
  Both arms share **identical generated samples**: 2× N evaluations of stored
  results, only N actual model calls.
- Added checked equation counts, math-error flags, audit scope and checker CPU
  overhead to sample records. The summary reports the verifier contribution,
  coverage, abstentions, and paired accuracy deltas.
- Added `tests/test_v07.py`, [docs/benchmark7.md](docs/benchmark7.md), and Windows
  / Linux experiment7 convenience scripts.
- Preserved v0.6 constrained-plan arm as a reproducible comparison.

**Crucial limitation:** An arithmetically correct equality is not proof that
operations are relevant to the question, or that the final answer follows.
If no explicit equation is found, the new strict filter abstains. Both positives
and negatives must be reported on 100+ development problems before conclusions.

---

# Release notes: v0.6.0

**Driven by the first real run** (Qwen3-0.6B Q8_0, CPU-only llama.cpp on an i7-8850H, 6 threads, ~28 tokens/s): thinking was correctly off and the verifier rejected every bad plan for the right reason, but **0 of 8 free-form step plans were valid** (invented numbers, broken format, abstentions) at 13-35 CPU-seconds each, while short answers cost 1.5-4 CPU-seconds. The server also stopped when Ctrl+C was pressed in its window.

## Added

- `cpu_platform/constrained.py`: per-question **GBNF grammar** (sent as llama.cpp's `grammar` request field, confirmed in `tools/server/server-common.cpp`). It admits only `{"steps":[{"op":..,"a":..,"b":..},...]}` with grounded numbers, earlier-step references and at most 8 steps. The model chooses operations and operands; exact rational arithmetic computes results; steps that don't feed the answer are dropped; the v0.3 verifier replays the result.
- Search kinds `cplan` (constrained plan) and `cot` (reason in prose, then final JSON answer; the last JSON object wins).
- Experiment 6 arms `small_cot`, `large_cot` (strong reference) and `small_constrained`. Default arms: small_greedy, small_cot, large_greedy, large_cot, small_vote, small_constrained. Comparisons against both large references; verifier effect vs `small_vote`; `grammar_violation_rate`; `unparseable_rate` per arm. Per-kind token caps (answer 64, cplan 256, cot/plan 512).
- Preflight: one constrained request before the run. Exits with a clear message if the server rejects or ignores the grammar.
- Grounding: number words (one to twenty, tens, hundred, thousand, dozen, triple, quarter, ...) and unit-conversion constants that need both units in the question (hours+minutes -> 60, weeks+days -> 7, dollars+cents -> 100, ...).
- `tests/test_v06.py` (22 tests), including a small GBNF parser/matcher built from llama.cpp's grammar syntax. It checks that generated grammars are well-formed, accept valid plans, reject the v0.5.2 failure modes, and that 400 random strings generated from the grammar all pass the plan checker. Full suite: 163 tests.

## Changed

- Launcher: a "KEEP THIS WINDOW OPEN / Ctrl+C here STOPS the model server" banner, a message distinguishing Ctrl+C from a crash (with the last log lines), and auto-detection of `vendor/llama-cpu/llama-server(.exe)`.
- Experiment 6: a "no server at this URL, start it in a SEPARATE window with ..." message instead of a traceback.
- Hardware: the same GPU reported by nvidia-smi and Windows is listed once.

## Not yet known

Whether constrained plans pick the *right* operations, and so whether `small_constrained` beats `large_cot` at matched CPU-seconds. Experiment 6 measures exactly this.

# Release notes: v0.5.2

**Focus:** hardware-agnostic CPU-only setup, and the first experiment that directly tests the project thesis.

## Added

- `cpu_platform/hardware.py`: portable detection (Windows, Linux, macOS; standard library only) of physical cores, RAM, GPUs and single-thread memory bandwidth, plus a recommended profile: threads = physical cores, context size and feasible models by RAM, and a rough tokens/s estimate per model. `python -m cpu_platform.hardware`.
- `cpu_platform/launch.py`: one cross-platform llama.cpp launcher that replaces the hardcoded per-OS settings. It picks model presets (`qwen3-0.6b` to `qwen3-14b`, or `--hf`/`--model-file`), refuses models that don't fit in RAM, and hides every GPU (`--device none` when the build supports it, plus CUDA/ROCm/SYCL/Vulkan environment variables). It binds to loopback, saves the server log, scans it for GPU use, and writes `results/launch_<port>.json` (argv, version, PID, settings, hardware, GPU verdict). Optional flags are added only when that build's `--help` lists them (flag names checked against current llama.cpp source).
- `cpu_platform/search.py`: verifier-guided test-time search. Draws N seeded samples, checks plan candidates with exact replay, majority-votes (ties go to the first seen), and scores every prefix. Includes a psutil CPU-seconds meter for the model server process.
- `cpu_platform/experiment6.py`: paired arms `small_greedy`, `large_greedy`, `small_vote` (sampling without the checker) and `small_verified`. Produces accuracy-vs-cost curves, a matched-compute comparison against the larger model, and the verifier's effect at an equal number of samples. Cost is server CPU-seconds when metered, otherwise wall-clock, never mixed.
- `docs/hardware.md`, `docs/benchmark6.md`, `scripts/run_*_experiment6`.
- Tests: `tests/test_v052.py` (20 tests) and `tests/fake_llama.py`, an offline stand-in for llama.cpp's HTTP API that runs the real client code end to end. Full suite: 141 tests. Also fixes a race in the v0.4 `test_metrics_no_prompt_content` test that failed intermittently under load.

## Changed

- `scripts/serve_windows_cpu.ps1` and `scripts/serve_cpu.sh` now delegate to the launcher (previously `-t 4 -c 2048` on Windows and no thread setting on Linux).
- Product and API start scripts use `--max-inflight 1` (one llama.cpp process should serve one request at a time).
- Experiment 4 scripts read threads and context size from the launch manifest (`--launch-manifest`) instead of hardcoding `--threads 4 --context-size 2048`. The manifest is embedded in the summary.
- **Qwen3 thinking is disabled for every mode**: `--reasoning off` (or `--reasoning-budget 0`) at launch, `/no_think` in the system prompt for Qwen3 model ids, and `<think>` blocks stripped from replies. Without this, Qwen3 can spend the whole token budget thinking and produce no JSON, which would make every mode look broken.
- `LocalLlama.chat` accepts `temperature` (0–2) and `seed`, and only sends them when set.

## Still true

No real model has been run as part of this release (the build environment can't download weights). The first real numbers come from running experiment 6 on your own machine.

# Release notes: v0.5.1 (bug-fix patch)

Three fixes, each with regression tests in `tests/test_v051_fixes.py` (9 tests, all of which fail on v0.5.0). Full suite: 121 tests.

1. **Sentence-final numbers are now grounded** (`cpu_platform/grounding.py`). The number regex rejected any number followed by a period, so "pay with $50." never matched 50. This made the `purchase_change` semantic template impossible to pass and caused `adaptive`, `proof` and `semantic` modes to reject correct answers whenever a key number ended a sentence. Version strings and ordinals ("v1.2.3", "5th") are still rejected. A new test runs every supported smoke question through full `verify_semantic_plan`, not just `classify`.
2. **Job workers survive storage errors** (`cpu_platform/jobs.py`). Previously an exception while claiming or saving a job killed the worker thread and left the job stuck in `running` (uncancellable, counting against quota) until restart. Now claims back off and retry; result writes retry three times, then fall back to a bare `failed` / `result_storage_failed` record so the job leaves the active set. If storage is completely unavailable, the job stays `running` until restart recovery marks it `interrupted`, but the worker keeps running.
3. **One shared execution budget** (`cpu_platform/server.py`, `jobs.py`). Synchronous requests and queued jobs now share the `--max-inflight` semaphore, so together they never exceed it. Workers take a slot *before* claiming a job, so a job waiting for capacity stays `queued`, not `running`. With `--job-workers` greater than `--max-inflight`, the extra workers simply wait. While jobs hold every slot, synchronous requests get the existing 429 `capacity_exceeded`.

Recommended setup: `--max-inflight` equal to the number of parallel requests your llama.cpp server is configured for (often 1 on an 8 GB laptop).

# Release notes: v0.5.0 (cumulative)

**Focus:** transition from a one-shot local research prototype to an opt-in durable, CPU-native single-node job service.

## New in v0.5

- New `cpu_platform/jobs.py`: SQLite WAL queue with idempotent admission, atomic claims, single-owner lock, pending/running/terminal lifecycle, bounded worker pool, cancellation before execution, restart recovery, 24-hour default lazy retention and configurable quotas.
- HTTP `POST /v1/jobs`, `GET /v1/jobs/{id}`, `DELETE /v1/jobs/{id}` plus queue status counts in metrics. Optional tenant-specific keys with no cross-tenant job visibility.
- SDK `submit_job`, `get_job`, `cancel_job`, `wait_for_job`.
- Browser job console and one-command Windows/Linux startup scripts.
- `cpu_platform.experiment5` for reproducible local workload stress tests and queue timing.
- Offline tests for concurrency and backpressure, isolation, idempotency, cancellation, interrupted-job recovery, API validation and workloads.

## Preserved

- v0.1: Python SDK, local HTTP API, exact arithmetic, local llama.cpp model adapter, browser UI.
- v0.2: adaptive inference and bounded repair, comparison harness.
- v0.3: exact multi-step proof graphs and four-way evaluation.
- v0.4: restricted semantic-template checks, five-mode reproducible evaluation.

## Validation

- The offline tests exercise a fake model and exact arithmetic, not a real model.
- No inference accuracy, energy/cost advantage or frontier reasoning quality has been demonstrated.
- The API remains deliberately loopback-only. SQLite jobs store input/output content on disk, are not encrypted by this application, and are not intended for production internet exposure.
