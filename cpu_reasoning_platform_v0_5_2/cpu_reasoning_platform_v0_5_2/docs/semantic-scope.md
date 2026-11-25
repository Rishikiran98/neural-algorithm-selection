# v0.4 semantic verification: narrowly scoped

`semantic.py` is a fully anchored grammar parser for a few numeric word-problem templates. It is **not** an LLM-based judge, a theorem prover, or general semantic entailment.

## Accepted forms

The currently supported, parameterized templates are:

1. `Maya has 12 apples, buys 8 more, and puts them equally into 5 boxes.`
2. `Maya has 12 apples and buys 8 more. She splits them equally into 5 boxes. How many apples are in each box?`
3. `A bus has 25 passengers. At the first stop, 7 leave and 9 board. How many passengers now?`
4. `A store charges $12 per notebook. You buy 3 notebooks and pay with $50. How much change?`
5. `A factory produces 15 toys per hour for 4 hours. It discards 8 defective toys. How many remain?`

Numerical values and constrained item words are variable. The bus, change, hourly-output and apple examples map to fixed *operation schemas*, not memorized answers. Sentences outside these templates are **unsupported**, even if they look obviously equivalent to a human.

## Algorithm

1. Run the original v0.3 strict arithmetic replay and numeric grounding check.
2. Match the **entire** question to one recognized template. Do not infer meaning if extra clauses or negations are present.
3. Instantiate the expected operation graph and operand roles from captured numeric fields. 
4. Require the model-produced graph to have the same number of steps, operations and operand roles (with commutative operand reordering allowed).
5. Independently compare the final result using exact fractions.
6. Only then set `semantically_verified=true`. This field means **consistent with the matched restricted template**, not general correctness.

An incorrect plan can pass `POST /v1/verify-plan` because that is arithmetic-only. The same plan must fail `POST /v1/verify-semantic-plan`. The stricter endpoint returns **422**, no certified numeric answer, and a `semantic_error` explaining the mismatch or unsupported wording.

`POST /v1/reason` with `"mode":"semantic"` requests a graph from the CPU-hosted model, checks its steps and the template, and optionally requests one repair. Unsupported wording immediately returns an invalid response with **zero** model calls. The original `"mode":"proof"` retains its v0.3 meaning, so there is no silent change to its verification contract.

## Known limitations

- Recognition is brittle by design and covers a tiny English grammar. Paraphrases and genuine mathematical alternatives may be rejected.
- Matching a strict, predefined template is a form of structured schema verification, **not** proof of natural-language intelligence.
- Monetary units, distributive reasoning and complex multi-hop problems outside these patterns are not covered.
- This does not verify all mathematical propositions, the correctness of world facts, or that a model understands the problem.
- The model may reject or fail to generate JSON, particularly with small quantized models.
- Proof repair does not guarantee eventual success, and it can be slower than a single direct answer.
