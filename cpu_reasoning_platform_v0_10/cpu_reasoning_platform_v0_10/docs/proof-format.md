# Version 0.3: Typed arithmetic proof format

This is a deliberately limited arithmetic checker. It checks claims about individual arithmetic steps using exact fractions. It does **not** check whether the selected operations solve the natural-language problem.

## Format

```json
{
  "steps": [
    {"id":"s1","op":"add","inputs":[{"literal":"12"},{"literal":"8"}],"claim":"20"},
    {"id":"s2","op":"divide","inputs":[{"ref":"s1"},{"literal":"5"}],"claim":"4"}
  ],
  "result":"s2"
}
```

Call `verify_plan(question, plan)` in Python or `POST /v1/verify-plan` with JSON `{ "question": "...", "plan": { ... } }`. The API route and the direct Python function do not call a language model.

- 1 to 12 ordered steps, exactly two inputs per step, and unique step IDs.
- Only `add`, `subtract`, `multiply`, `divide` are permitted.
- Every input must be either `{"literal":"12"}` or `{"ref":"s1"}`. References must point to previous steps.
- Literals must be visible as numbers in the original question. Exceptions: 1 and 100 when percentages are mentioned, and 2 when half/double/twice is mentioned.
- Every step must declare an exact `claim` in decimal or rational string form. For example `"1/3"`. The checker independently recomputes it.
- Final result must name the last step; no orphan steps are accepted.
- Small hard limits guard step count, input bytes at HTTP, number lengths, computed magnitudes, and invalid references.
- No Python execution or evaluation of generated code takes place.

## What passing means

A passing proof establishes **only** (1) the arithmetic calculation was performed correctly for the asserted operations; (2) each declared intermediate answer matches; (3) reference order and graph structure are valid; and (4) input literals pass a lexical provenance screen.

A passing proof does **not** show that adding, subtracting, multiplying or dividing is appropriate for the question. For example, if the question asks to divide 20 apples among five boxes, a plan that adds 20 and 5 instead can be accepted when its arithmetic claims are internally consistent and both numbers occur in the question. Therefore `semantically_verified` is always `false`.

The service uses a second model call only if the first plan fails structural, arithmetic or literal checks. It cannot detect and repair a plan that is arithmetically consistent but semantically wrong.

## New API route

`POST /v1/reason` supports `"mode":"proof"` and `"max_calls":1` or `2`. It returns `proof_trace`, `proof_verified`, `arithmetic_verified`, `numeric_grounding_passed`, and `semantically_verified`. A model must be running for the `proof` route.

`POST /v1/verify-plan` uses the same verification logic without any model call and returns the structured audit plus zero model usage. For unauthorized access it returns 401, invalid request fields 400, failed proof 422.

Treat the `proof_trace` text as evidence of arithmetic replay only. Do not label it as verified general reasoning.
