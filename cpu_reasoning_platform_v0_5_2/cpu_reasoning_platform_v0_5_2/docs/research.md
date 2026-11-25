# Experiment 003 research protocol

## Hypothesis

For restricted arithmetic word problems, small CPU-only language models may gain accuracy from generating a typed intermediate calculation graph rather than a final answer. A verifier may reject some malformed calculations before they become answers. The effect on actual task correctness, CPU latency and memory is unknown until measured.

## Four paired arms

- `baseline`: model emits a numeric answer. One model call.
- `hybrid`: model emits one expression, evaluated by exact arithmetic. One model call.
- `adaptive`: model emits an expression; numeric provenance and up to one repair. At most two model calls.
- `proof`: model emits a typed multi-step plan; deterministic step claims, references and numeric provenance are checked. At most two model calls.

Only `proof` checks **intermediate** claims. Neither `proof` nor any other mode verifies semantics. A wrong plan can be internally valid. Each arm is compared to independent ground-truth answers.

## Benchmark procedure

1. Confirm local `llama.cpp` runs with GPU offloading disabled (`-ngl 0`), and log its command line and engine commit/version.
2. Record CPU model, thread count, RAM, model GGUF identifier/hash, quantization level, context length, operating system, and whether any accelerator is present.
3. Start the local model and local API, then run `python -m cpu_platform.experiment3 --dataset data/smoke.jsonl --limit 5 --output results/experiment3_smoke.jsonl`.
4. Treat the smoke run as a program test, not a scientific result. For meaningful testing use an unseen, held-out collection of at least 100 examples, with a separate development set and reported source licenses.
5. Use the **same CPU, model, model revision, quantization, decoding settings, and dataset** across all four modes. The evaluation rotates call order by task to partially reduce order bias.
6. Report every abstention, invalid plan, service error and sample. Never filter away failed examples. The program refuses partial comparisons or missing arms.
7. Report accuracy, valid-output rate, model calls, prompt/completion tokens, wall latency, step-proof pass rate, p95 latency, and exploratory paired accuracy differences. For real CPU-seconds, peak RAM and energy, instrument independently. The program does not invent these quantities.
8. Analyze the cases where the model has a valid plan but an incorrect final answer, and cases where valid numeric operations cannot be expressed within the schema.

## Critical controls

- A two-call proof route has a larger inference budget than one-call baselines. Its higher accuracy, if any, is not attributable to the proof format alone without cost-matched controls.
- Prompt formats vary between arms. Differences may be caused by prompts, model capabilities, or decoding, not architecture. Separate prompt and model ablations are required.
- The proof prompt contains a small example. Exclude exact duplicate problems and assess for contamination.
- This implementation evaluates GPU-free **inference** only; open weights were likely trained using GPU or other accelerators.
- No real-model experiment has been executed by the automated fake-model suite. A running CPU model is required before any claim about reasoning quality or speed.

## Next problems

- Build sounder domain-grounded constraints from a structured specification or formalized problem statement.
- Use selective computation and calibrated confidence, not blind repetition.
- Add cost-matched CPU baselines and memory/CPU-seconds instrumentation.
- Benchmark medium-sized models and larger multistep arithmetic, not just GSM8K.
