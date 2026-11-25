# Experiment 007: Check free-form arithmetic reasoning rather than constrain its output

## Hypothesis

Given the **same model outputs**, removing chains with explicit arithmetic contradictions
from a self-consistency vote can improve accuracy per unit of CPU work. The counterhypothesis
is equally plausible: filtering introduces abstentions and discards correct answers.

This is **not** a proof that the model interpreted the problem correctly. Checking
`50 - 36 = 14` cannot establish whether 36 is the relevant quantity or whether the
final answer follows from the displayed equations. Model-generated words can also
omit critical calculations or invent incorrect rules without stating an equality.

## Two paired arms

- `small_cot_vote`: N sampled, unconstrained visible reasoning replies, majority vote
- `small_cot_checked`: the exact SAME N replies, no further inference. Reject a sample
  from voting if any supported equation is incorrect, if no explicit arithmetic
  equation is checkable, or if the audit exceeds its limits. Otherwise, vote.

Both arms use the same model, prompt, temperature, seed, token budget, and generated
text. The checker runs locally without a neural model. `small_cot` remains a
separate deterministic single-answer baseline. `small_constrained` remains a separate
negative-control architecture; it is **not** removed.

## Supported checks

Explicit numeric equalities of the form `12 * 3 = 36`, `50 - (12*3) = 14`,
`7 / 4 = 1.75` or `3 - 8 = -5`. Unicode `×`, `÷`, `−` are normalized.
The left side is evaluated by the existing bounded AST arithmetic evaluator.
The expression is never executed as Python source. Only a bounded portion of
model output (12,000 characters, up to 40 equations) is audited.

**Not supported:** percentages stated in natural language, implicit multiplication,
unit conversion prose, variables or algebra, inequality proofs, numerical claims
without `=`, alternate formats, and semantic validation of the source question.
A 'check passed' means **only the extracted equations** are arithmetically consistent.
The extractor can miss errors. Missing equations cause abstention, not approval.

## Windows quick run

Start the small model with the CPU-only llama.cpp launcher in another terminal
and leave it running on port 8080. Check `/health` before starting.

```powershell
py -m unittest discover -s tests -v
py -m pip install psutil
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_windows_experiment7.ps1 --limit 5 --samples 4
```

For 100 development questions:

```powershell
py -m pip install datasets
py scripts/fetch_gsm8k.py --limit 100
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_windows_experiment7.ps1 --dataset data/gsm8k_test.jsonl --limit 100 --samples 4
```

**Output:** `results/experiment7.jsonl`, `results/experiment7.summary.json`.
These JSONL files contain the full visible model-generated replies. They are
local research artifacts, not suitable for sensitive inputs or sharing without review.

## How to interpret

Check `results.comparisons.paired_cot_arithmetic_filter_effect` for:

- `same_model_outputs`: true by construction, enforcing a fair offline comparison
- `accuracy_before`, `accuracy_after`, `accuracy_difference`
- `sample_arithmetic_error_rate`
- `sample_checkable_rate` and `sample_vote_eligible_rate`
- `mean_checker_cpu_seconds_per_question`

Also compare the `curves` (accuracy, abstention rate, latency/CPU cost). A filtering
method that removes mistakes but abstains on everything has not succeeded.

CPU-seconds count llama-server process time plus checker process CPU time when
`psutil` is available; fallback wall-clock includes local checker time.
The checker overhead is **measured**, not assumed to be zero.

## Research cautions

- Five bundled questions are a software smoke test, not an accuracy claim.
  One is similar to the existing prompt example, so contamination is possible.
- Select 100 or more challenging held-out tasks, and eventually privately created
  tests unseen during prompt development. GSM8K is public and not truly unseen.
- Report how many chains had checkable equations. Small models often emit prose
  without equalities and will be excluded by this strict policy.
- Use the same prompt and N for the two paired arms. Compare against strong
  single-chain CoT, and later larger-model CoT at matched measured CPU cost.
- Report failures and abstentions openly. Do not call consistency of extracted
  equations an independently verified reasoning chain.
