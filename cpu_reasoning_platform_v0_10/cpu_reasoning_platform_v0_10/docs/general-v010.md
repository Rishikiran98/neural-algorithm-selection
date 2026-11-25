# v0.10.0: Multi-domain reasoning research preview

## What changed

The legacy `/v1/reason` arithmetic service remains available. `/v1/ask` is a new local-model endpoint that answers broader questions in general explanation, coding, propositional reasoning, mathematics, science and factual Q&A. It does not turn the 0.6B model into a frontier model. It **does not verify general answer correctness**.

`POST /v1/ask` supports:

```json
{
  "question": "Explain why binary search takes O(log n) time.",
  "domain": "auto",
  "strategy": "check_and_revise",
  "thinking": false,
  "think_budget": 128,
  "max_calls": 2
}
```

- `domain`: `auto`, `general`, `math`, `logic`, `code`, `science`, `factual`.
- `strategy`: `direct` or `check_and_revise`. The second strategy makes another model call **only when a detectable arithmetic equality is wrong**. No mistake means no review call. This is not a general critic.
- `thinking`: opt-in native model thinking; depending on the model/llama.cpp version, it may exhaust the output budget and return no final answer.
- `think_budget`: 0 to 512; an inference budget, **not a guarantee** of improved correctness.
- Exact, unambiguous arithmetic expressions are computed locally with zero model calls.
- For model answers `verified` remains false. Explicit arithmetic equalities in generated text are recomputed and an optional fenced Python snippet is **parsed** but never executed. Arithmetic validity is not an endorsement of the model's interpretation or final claim. Source retrieval is not yet implemented.

`POST /v1/verify-logic` checks formal *propositional* entailment with a bounded complete truth table:

```json
{"premises": ["p -> q", "p"], "conclusion": "q"}
```

Supported operators: `!` not, `&` and, `|` or, `->` implies, plus parentheses, named Boolean atoms and true/false. Maximum 12 premises and 12 distinct variables, 512 characters per formula. It returns a countermodel for invalid entailments and explicitly flags vacuous entailment from inconsistent premises. It does **not** establish that the variables accurately translate a natural-language problem.

## Getting started on Windows

1. Unzip the full v0.10 directory into its own folder. Do not overwrite your v0.9.1 frozen evaluation data.
2. In PowerShell run `py -m unittest discover -s tests -v`.
3. Launch your existing CPU-only Qwen3 GGUF server on `127.0.0.1:8080` using the **known working CPU executable** on your machine. The model has to be running for general questions.
4. Run `py -m cpu_platform.server --port 8070` from the new project folder.
5. Browse to `http://127.0.0.1:8070`. The new general Q&A and formal-logic panels are below the original arithmetic playground.
6. Run `py -m cpu_platform.general_eval --limit 12` to collect a local cross-domain pilot. It writes `results/general_smoke_v010.jsonl` and `results/general_smoke_v010.summary.json`.

Run this example in PowerShell:

```powershell
$body = @{question="Explain binary search complexity"; domain="general"; strategy="direct"} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8070/v1/ask -ContentType application/json -Body $body
```

Python SDK:

```python
from cpu_platform.client import CPUReasoningClient
client = CPUReasoningClient()
print(client.ask("Explain binary search complexity", domain="general"))
print(client.verify_logic(["p -> q", "p"], "q"))
```

## Evaluation rules

`general_eval` rejects frozen evaluation material and renamed copies before calling a model. It uses an intentionally small varied development set. Only unambiguous exact numeric answers can be machine-scored; coding, science, logical explanation, and facts receive `unscored` unless evaluated independently using predeclared task-specific tests. Missing/unparseable answers are **not** counted as correct. The local pilot is NOT a frontier or general-intelligence benchmark.

Before making claims, add frozen held-out task suites: executable coding tasks in a process-isolated sandbox, formal logic with correct translations, mathematical tasks with verified answers, factual QA with timestamped evidence, multistep planning tasks with environments, and calibration tests. Test inference time, memory, throughput, verifier cost, and failure handling across multiple CPU models. Maintain v0.9.1's frozen split constraints.

## Safety and scope

The preview is bound to localhost. There is no internet retrieval, agent automation, safe code-execution sandbox, long-term causal world model, or guarantee of factual accuracy. No arbitrary model-generated code is executed. Jobs in the legacy local queue store prompts and outputs in plaintext; avoid sensitive data. Public internet deployment requires substantial additional controls.
