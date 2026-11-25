# Experiment 006: small model + verified search vs a larger model, at matched compute

## Why this experiment

The project goal is frontier-level reasoning on CPUs without repeatedly processing billions
of parameters. On a CPU, each generated token streams every weight through memory, so a
0.6B model is roughly 5–10× cheaper per token than a 4B–8B model. The thesis holds only if
spending that saving on **many cheap, exactly-checked attempts** gives more accuracy than
spending it on **one pass of a bigger model**. Experiment 6 measures this directly.

## Arms (paired, same questions)

| Arm | What runs | Role |
| --- | --- | --- |
| `small_greedy` | small model, 1 answer, temperature 0 | floor |
| `small_cot` | small model reasons step by step, then answers | small model's best single try |
| `large_greedy` | large model, 1 answer, temperature 0 | reference: a bigger model answering directly |
| `large_cot` | large model reasons step by step, then answers | **strong reference: what a reviewer will compare against** |
| `small_vote` | small model, N sampled answers, majority vote | control: sampling without structure or checking |
| `small_constrained` | small model, N grammar-constrained plans computed exactly, majority vote | **the hypothesis (v0.6)** |
| `small_verified` | free-form step plans (v0.5.2) | optional (`--arms`); 0/8 valid in the first real run |

Sampled arms are scored at every prefix k = 1..N from the same samples, which gives an
accuracy-vs-cost curve with no extra model calls.

## Running it

```powershell
# Window 1 and window 2 (from the project root):
powershell -NoProfile -File .\scripts\serve_windows_cpu.ps1 --model qwen3-0.6b --port 8080
powershell -NoProfile -File .\scripts\serve_windows_cpu.ps1 --model qwen3-4b  --port 8081
# Window 3 (optional, for CPU-seconds): py -m pip install psutil
py scripts/fetch_gsm8k.py --limit 100          # needs: py -m pip install datasets
powershell -NoProfile -File .\scripts\run_windows_experiment6.ps1 --dataset data/gsm8k_test.jsonl --limit 100 --samples 8
```

On Linux/macOS use `scripts/serve_cpu.sh` and `scripts/run_linux_experiment6.sh` with the same arguments.

Both servers stay loaded while the experiment sends **one request at a time**, so they never
compete for cores. Make sure both fit in RAM together (`--list-models`). On a 32 GB machine,
qwen3-8b is also a fair reference: `--model qwen3-8b --port 8081`.

## Cost

- **CPU-seconds** (preferred): with `psutil` installed, the experiment reads each server's PID
  from `results/launch_<port>.json` and records user+system CPU time per call.
- **Model wall-clock seconds** (fallback): used for every arm whenever any server can't be
  metered. The two units are never mixed. The unit is recorded in `cost_unit`.

## Reading the summary (`results/experiment6.summary.json`)

- `curves.<arm>`: accuracy, abstain rate, mean cost and tokens for each k.
- `comparisons.small_verified_vs_large_greedy`:
  - `matched_k`: the largest k whose mean cost is within the large model's mean cost.
  - `accuracy_minus_reference`: the paired accuracy difference at that k, with a bootstrap interval.
  - `first_k_reaching_reference_accuracy` and `cost_ratio_at_that_k`: how much compute the small model needs to equal the large one.
- `comparisons.<arm>_vs_large_cot`: the same comparison against the strong reference. **This is the one that matters for a paper.**
- `comparisons.verifier_effect_small_constrained_vs_small_vote`: with the same N, separates structured computation from just sampling more.
- `comparisons.grammar_violation_rate`: should be 0. Anything else means the server ignored the grammar (the preflight check normally stops the run before that).

**The thesis is supported** (on this data and hardware) only if `small_constrained` matches or beats
`large_cot` at `matched_k`, the interval excludes 0 on 100+ held-out questions, and
`verifier_effect` is positive. Anything else is also a result. Report it the same way.

## Caveats

- The verifier checks arithmetic and that numbers come from the question. It does **not** check
  that the plan answers the right question: an arithmetically valid but wrong plan can win
  the vote. Expect this to be the main failure mode on GSM8K.
- Plans are longer than answers, so `small_verified` costs more per sample than `small_vote`.
  Compare along the cost axis, not just k.
- GSM8K is public and may be in the models' training data. Treat it as a development set and
  confirm any positive result on held-out problems.
- The bundled offline tests use a fake server whose "plans" are correct by construction.
  They check the software, not the hypothesis.
