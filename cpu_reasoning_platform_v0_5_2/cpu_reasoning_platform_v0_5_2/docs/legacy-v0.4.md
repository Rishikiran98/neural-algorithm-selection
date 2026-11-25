# CPU Reasoning Platform v0.4.0

**Cumulative CPU-only inference research prototype, November 2, 2026.** Version 0.4 includes every original v0.1/v0.2/v0.3 module and test, and adds bounded **restricted-template semantic verification** plus a reproducibility-focused, real-model benchmark harness.

**What it is:** A local Python/HTTP research platform for comparing a small CPU model to deterministic arithmetic, adaptive inference, step-graph verification and a restricted semantic graph checker. **What it is not:** a frontier AI system, proof of arbitrary English comprehension, a GPU-free training pipeline, or an internet-ready SaaS.

## 1. Run on Windows (approximately 8 GB RAM)

Install Python 3.11+ and unzip the project. In PowerShell, from the project root:

```powershell
py -m unittest discover -s tests -v
py -m examples.verify_semantic
py -m cpu_platform.server --port 8070
```

Open **http://127.0.0.1:8070** in your browser. You can now use:

- `calculate: 12*(7+3)` with **zero neural calls**.
- The original **Audit a reasoning graph** arithmetic verifier, which intentionally does *not* check the meaning of the word problem.
- The new **Template semantic check** button, which checks both exact arithmetic and correspondence to one of the few fully recognized English templates. Unsupported wording must abstain.

These tools work without downloading a model. All bundled offline tests use scripted fixture responses, not a measured LLM.

## 2. Optional local model on CPU

Install [llama.cpp](https://github.com/ggml-org/llama.cpp) (for example, `winget install llama.cpp`) and launch the existing CPU-only Windows script in a second PowerShell window:

```powershell
powershell -NoProfile -File .\scripts\serve_windows_cpu.ps1
```

The script requests `Qwen/Qwen3-0.6B-GGUF:Q8_0` and explicitly disables GPU offload via `-ngl 0`. The GGUF model weights are downloaded separately and are not bundled. Commands can differ by installed llama.cpp version, so consult `llama serve --help` when necessary.

In the browser, select **Template-checked graph** and try:

> Maya has 12 apples, buys 8 more, and puts them equally into 5 boxes.

The result is only accepted if the neural model emits a structurally valid graph with correct arithmetic **and** matches the supported operation template. A small model may fail to generate a valid graph. That is measured as failure or abstention, not a success.

## 3. New v0.4 semantic verification API

**Model-free strict verifier:**

```http
POST /v1/verify-semantic-plan
Content-Type: application/json

{"question":"Maya has 12 apples, buys 8 more, and puts them equally into 5 boxes.","plan":{"steps":[{"id":"s1","op":"add","inputs":[{"literal":"12"},{"literal":"8"}],"claim":"20"},{"id":"s2","op":"divide","inputs":[{"ref":"s1"},{"literal":"5"}],"claim":"4"}],"result":"s2"}}
```

Returns 200 with a checked result if supported and correct, otherwise 422. No user-supplied executable code is run. The original `POST /v1/verify-plan` API is unchanged and may accept arithmetically valid but semantically wrong operation graphs.

**CPU-model-assisted strict reasoning:**

```http
POST /v1/reason
Content-Type: application/json

{"question":"Maya has 12 apples, buys 8 more, and puts them equally into 5 boxes.","mode":"semantic","max_calls":2}
```

For an unsupported word problem, `semantic` mode returns an invalid result with `stop_reason=unsupported_template` without contacting a model. This does not mean the problem is intrinsically unsolvable.

**SDK:**

```python
from cpu_platform.client import CPUReasoningClient
client = CPUReasoningClient()
result = client.verify_plan(question, plan, semantic=True)
```

See [docs/semantic-scope.md](docs/semantic-scope.md) for the exact supported templates, structural comparison and limitations.

## 4. CPU-only benchmark (real model required)

Start the model and API as above, then in a third PowerShell window:

```powershell
powershell -NoProfile -File .\scripts\run_windows_experiment4.ps1
```

Outputs:

- `results/experiment4_smoke.jsonl`: one result for each problem and execution mode.
- `results/experiment4_smoke.summary.json`: performance metrics, device metadata, reproducibility manifest, dataset and source hashes.

Benchmark modes: `baseline`, `hybrid`, `adaptive`, `proof` and `semantic`. The five bundled smoke problems are **not enough to establish a reasoning-performance improvement**. The optional `--model-file` flag hashes model weights if you know the location; `--model-pid` can sample model memory with optional `psutil`. `--gpu-layers 0` is a declared requirement, **not hardware attestation**.

For remote Linux CPU hosts:

```bash
bash scripts/build_cpu_llama.sh
bash scripts/serve_cpu.sh
python3 -m cpu_platform.server --port 8070
# In another terminal:
bash scripts/run_linux_experiment4.sh
```

All services bind to loopback. Use SSH forwarding instead of opening ports to the internet. See [docs/benchmark4.md](docs/benchmark4.md) for larger paired runs, limitations and interpretation.

## 5. Preserved versions

| Version | Included features |
| --- | --- |
| v0.1 | local HTTP API, Python SDK, exact arithmetic, web playground, local CPU model adapter, basic metrics |
| v0.2 | adaptive inference, bounded retries, grounding checks, three-mode evaluation |
| v0.3 | exact multi-step arithmetic graph verifier, proof inspection API/UI, four-mode evaluation |
| **v0.4** | narrow semantic plan correspondence, strict offline and model routes, five-mode reproducible benchmark, extra tests |

The old endpoints and experiment scripts remain available. `cpu_platform/experiment2.py` and `cpu_platform/experiment3.py` are retained; v0.4 introduces `experiment4.py` without removing them.

## 6. Limitations and security

- **No frontier-quality, GPU-free AI reasoning has been demonstrated.** The small language model might have been trained using GPUs elsewhere.
- Recognized semantic templates are intentionally limited. Most language problems are unsupported, and even a positive match is not a general semantic proof.
- The checker validates a graph against deterministic schemas; it does not verify general-world facts, multi-step natural language entailment or arbitrary programs.
- The prototype API is loopback-only and is not a hardened public service. Do not directly expose it to the internet. Local bearer authentication is not a substitute for production identity, TLS and isolation.
- Offline tests are software regression tests. Real-model speed, energy efficiency, accuracy and cost advantages **remain unmeasured** in this release.
- Benchmark modes have unequal computation budgets. Performance claims require independently verified CPU-only execution, held-out datasets, adequate sample sizes and comparable cost metrics.

## Repository map

```text
cpu_platform/engine.py         v0.1-v0.3 reasoning routes plus v0.4 semantic mode
cpu_platform/proof.py          arithmetic-only graph verifier (unchanged interface)
cpu_platform/semantic.py       v0.4 supported-template verifier
cpu_platform/experiment4.py    real-model paired benchmark and reproducibility metadata
cpu_platform/server.py         local API including strict semantic route
cpu_platform/static/          local browser playground
cpu_platform/client.py         Python SDK
examples/verify_semantic.py   zero-model-call demonstration
scripts/                      Windows/Linux CPU launch and experiment scripts
tests/                        cumulative tests for v0.1-v0.4
.github/workflows/ci.yml     Windows/Linux offline testing, no model inference
```

Source code is MIT licensed. Model and dataset licensing should be reviewed independently before production redistribution.
