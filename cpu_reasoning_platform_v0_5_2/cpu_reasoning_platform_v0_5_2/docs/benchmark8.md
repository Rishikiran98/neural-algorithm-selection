# v0.8: let the small model think, then check its arithmetic

## Why

The first real runs (Qwen3-0.6B, CPU only, ~28 tokens/s on an i7-8850H) showed:

| v0.6 smoke (5 template questions) | accuracy | CPU-s / question |
|---|---|---|
| one direct answer | 1/5 | 2.4 |
| 4 grammar-constrained plans | 1/5 | 23.4 |
| reason in prose, then answer | **5/5** | 11.7 |

Forcing structure removed the model's room to think. Qwen3 is trained to *think* before
answering, and v0.5.2 had switched that off (thinking used up the token budget). v0.8 turns
it back on **with a hard budget**: llama.cpp forces the end of thinking after
`--think-budget` tokens, so the model must answer. The arithmetic checker then audits the
thinking as well as the answer.

## Default arms (all paired on the same questions)

| Arm | What runs |
|---|---|
| `small_cot` | 0.6B reasons in prose once (temperature 0) |
| `small_think` | 0.6B thinks natively, budget-capped, once |
| `large_cot` | 8B reasons in prose once: **the reference** |
| `small_cot_vote` / `small_cot_checked` | 8 sampled prose chains, voted / the SAME chains with arithmetic-error chains excluded |
| `small_think_vote` / `small_think_checked` | 4 sampled thinking runs, voted / the SAME runs filtered |

Optional: `large_think`, and the v0.6 arms (`small_greedy`, `large_greedy`, `small_vote`, `small_constrained`).

## Checker rules (`--check-policy`)

| Policy | A chain may vote if... |
|---|---|
| `exclude_errors` (default) | no checked equation is wrong (excluded only on evidence) |
| `strict` (v0.7 rule) | at least one equation was checked and none is wrong |
| `answer_supported` | no wrong equation, and the final answer is the result of a checked equation |

All three are computed for every sample and reported in `policy_table`, so they can be
compared without re-running the model. **Choose the policy on the pilot, then freeze it before
the main run** (otherwise it is a post-hoc choice).

Checker coverage (v0.8): `12 * 3 = 36`, `$12 × 3 = $36`, `1,200 * 2 = 2,400`, `15 x 4 = 60`,
LaTeX `\times`, `\div`, `\frac{a}{b}`, `25% of 80 = 20`, `**bold**`, numbered lists, rounding to
the stated precision (`10 / 3 = 3.33`, or `≈`). Not checked: equations with units inside them
(`15 toys/hour * 4 hours = 60`), algebra (`2x = 10`), percentages on the right-hand side.

## Running the pilot

```powershell
# Window 1 (leave it alone):  py -m cpu_platform.launch --model qwen3-0.6b --port 8080
# Window 2 (leave it alone):  py -m cpu_platform.launch --model qwen3-8b  --port 8081
# Window 3:
py -m pip install datasets psutil
py scripts\fetch_gsm8k.py --limit 200
powershell -NoProfile -File .\scripts\run_windows_pilot.ps1 --limit 20     # first look, ~45 min
powershell -NoProfile -File .\scripts\run_windows_pilot.ps1                # 100 questions, ~4 h
```

**Restart the model servers with the v0.8 launcher**: older launchers passed `--reasoning off`,
and a preflight check stops the run with instructions if thinking can't be switched per request.

Rough time per question on a 6-core laptop: small_think + 4 thinking samples ≈ 5 × 19 s,
prose arms ≈ 9 × 2.5 s, 8B prose ≈ 25 s, so about 2.5 minutes. `--think-budget 256` and
`--think-samples 2` roughly halve it.

## What to read in `results/pilot_v08.summary.json`

- `small_think` vs `small_cot`: does native thinking help the 0.6B model?
- `paired_think_arithmetic_filter_effect` and `paired_cot_arithmetic_filter_effect`: what the checker adds on identical outputs, plus `policy_table`.
- `small_think_checked_vs_large_cot`: **the thesis at matched CPU-seconds**: `matched_k`, `accuracy_minus_reference`, and `first_k_reaching_reference_accuracy` with `cost_ratio_at_that_k`.
- `curves.*.abstain_rate`: if a checked arm abstains a lot, the checker is discarding too much.

The checker verifies arithmetic only, never the reading of the question. GSM8K is public and
may be in the models' training data, so treat the pilot as a development set.
