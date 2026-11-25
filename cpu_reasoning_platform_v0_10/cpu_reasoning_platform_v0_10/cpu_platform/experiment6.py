"""Experiment 006: does a small CPU model + verified search beat a bigger model
at the SAME compute?

This is the first experiment that directly tests the project thesis:
"frontier-level reasoning without repeatedly processing billions of parameters".
Token generation on a CPU streams every weight per token, so a 0.6B model is
several times cheaper per token than a 4B/8B one. The question is whether
spending that saving on many cheap, exactly-checked attempts buys more accuracy
than spending it on a bigger model once.

Arms (all on the same questions, paired):
  small_greedy       small model, 1 answer, temperature 0
  small_cot          small model, reasons in prose then answers, temperature 0
  large_greedy       large model, 1 answer, temperature 0            (reference)
  large_cot          large model, reasons in prose then answers      (STRONG reference)
  small_vote         small model, N sampled answers, majority vote   (control: no verifier)
  small_constrained  small model, N grammar-constrained plans computed exactly, vote (v0.6)
  small_verified     small model, N free-form step plans, verified (v0.5.2; not default:
                     0/8 valid plans in the first real run)
  small_cot_vote     small model, N sampled prose reasoning chains, majority vote (v0.7)
  small_cot_checked  the SAME chains; chains with a wrong equation may not vote (v0.7/v0.8)
  small_think        small model, native thinking capped at --think-budget tokens (v0.8)
  small_think_vote   N thinking samples, majority vote (v0.8)
  small_think_checked the SAME thinking samples, checker-filtered (v0.8)
  large_think        large model with thinking (optional; expensive on CPU)

v0.8 default arms: small_cot, small_think, large_cot, small_cot_vote, small_cot_checked,
small_think_vote, small_think_checked. Default data: data/gsm8k_test.jsonl (100 questions).

For sampled arms every prefix k = 1..N is scored from the same samples, giving an
accuracy-vs-cost curve. Cost is model-server CPU-seconds when the servers' PIDs
are metered (psutil), otherwise model wall-clock seconds (stated in the output).

Typical run (two launcher windows first, see docs/benchmark6.md):
    python -m cpu_platform.experiment6 --dataset data/gsm8k_test.jsonl --limit 100
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import random
import statistics
import sys
from urllib.parse import urlparse

from cpu_first.benchmark import load_questions
from cpu_first.core import LocalLlama, gsm8k_gold
from . import __version__
from .experiment4 import sha256_file, source_sha256
from .hardware import detect
from . import datasets as registry
from .provenance import describe_server, evidence_table
from .cot_check import DEFAULT_POLICY, POLICIES
from .search import DEFAULT_THINK_BUDGET, make_cpu_meter, prefix_votes, sample, check_cot_samples

ARM_SPEC = {  # arm -> (model role, candidate kind, sampled?, verifier required?)
    "small_greedy": ("small", "answer", False, False),
    "small_cot": ("small", "cot", False, False),
    "small_cot_vote": ("small", "cot", True, False),
    "small_cot_checked": ("small", "cot", True, True),
    "large_greedy": ("large", "answer", False, False),
    "large_cot": ("large", "cot", False, False),
    "small_vote": ("small", "answer", True, False),
    "small_constrained": ("small", "cplan", True, True),
    "small_verified": ("small", "plan", True, True),
    # v0.8: native thinking mode with a token budget
    "small_think": ("small", "think", False, False),
    "small_think_vote": ("small", "think", True, False),
    "small_think_checked": ("small", "think", True, True),
    "large_think": ("large", "think", False, False),
}
# v0.8 default: the arms that test the thesis. The v0.6 arms (small_greedy, large_greedy,
# small_vote, small_constrained) remain available with --arms.
ARMS = ("small_cot", "small_think", "large_cot", "small_cot_vote", "small_cot_checked",
        "small_think_vote", "small_think_checked")
REFERENCES = ("large_greedy", "large_cot", "large_think")
KIND_MAX_TOKENS = {"answer": 64, "cot": 512, "plan": 512, "cplan": 256, "think": 0}  # think: budget + 256
# Arms that share one set of samples per question: the *_checked arm re-scores exactly
# the same outputs as its *_vote twin, so their difference isolates the checker.
SHARED = {"small_cot_vote": ("small", "cot"), "small_cot_checked": ("small", "cot"),
          "small_think_vote": ("small", "think"), "small_think_checked": ("small", "think")}
CHECKED_PAIRS = (("small_cot_checked", "small_cot_vote", "paired_cot_arithmetic_filter_effect"),
                 ("small_think_checked", "small_think_vote", "paired_think_arithmetic_filter_effect"))
# Errors that a correctly applied grammar makes impossible; any of them means the
# server ignored the grammar (see preflight and grammar_violation_rate).
GRAMMAR_VIOLATIONS = ("No JSON object", "Expected exactly", "operands must be", "bad number",
                      "ungrounded number", "not an earlier step", "expected op, a, b",
                      "Plan must have")


def is_correct(answer: str | None, gold: Fraction) -> bool:
    if answer is None:
        return False
    try:
        return Fraction(answer) == gold
    except (ValueError, ZeroDivisionError):
        return False


def run_experiment(models: dict, questions: list[dict], out: Path, *, arms: tuple[str, ...] = ARMS,
                   samples: int = 8, temperature: float = 0.7, seed: int = 606,
                   max_tokens: int | None = None, meters: dict | None = None,
                   warmup: bool = True, think_samples: int = 4,
                   think_budget: int = DEFAULT_THINK_BUDGET,
                   check_policy: str = DEFAULT_POLICY) -> list[dict]:
    """Run every arm on every question; write one JSON line per question."""
    meters = meters or {}
    unknown = [a for a in arms if a not in ARM_SPEC]
    if unknown or not arms or len(set(arms)) != len(arms):
        raise ValueError(f"Invalid arms: {arms}")
    for arm in arms:
        if ARM_SPEC[arm][0] not in models:
            raise ValueError(f"Arm {arm} needs the {ARM_SPEC[arm][0]} model")
    if not 1 <= samples <= 64 or not 1 <= think_samples <= 64 or not 0 < temperature <= 2:
        raise ValueError("samples and think_samples must be 1..64 and temperature in (0, 2]")
    if not 16 <= think_budget <= 32768:
        raise ValueError("think_budget must be 16..32768 tokens")
    if check_policy not in POLICIES:
        raise ValueError(f"check_policy must be one of {POLICIES}")
    if warmup:
        for model in models.values():
            model.chat("Reply with JSON {\"answer\":\"1\"}.", "1", max_tokens=16)
    order = list(range(len(questions)))
    random.Random(seed).shuffle(order)
    out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with out.open("w", encoding="utf-8") as stream:
        for position, qi in enumerate(order):
            item = questions[qi]
            gold = gsm8k_gold(item["answer"])
            record = {"id": str(item.get("id", qi)), "gold": str(gold), "arms": {}}
            # Rotate arm order so warm caches / thermal drift do not favour one arm.
            rotated = arms[position % len(arms):] + arms[:position % len(arms)]
            shared: dict = {}
            for arm in rotated:
                role, kind, sampled, _ = ARM_SPEC[arm]
                n = (think_samples if kind == "think" else samples) if sampled else 1
                limit = KIND_MAX_TOKENS[kind] if max_tokens is None or kind in ("answer", "cplan") \
                    else max_tokens
                options = dict(kind=kind, temperature=temperature if sampled else 0.0,
                               seed=seed * 1000 + qi * 100, max_tokens=limit,
                               think_budget=think_budget, cpu_meter=meters.get(role))
                if arm in SHARED:
                    key = SHARED[arm]
                    if key not in shared:
                        shared[key] = sample(models[role], item["question"], n, **options)
                    drawn = (check_cot_samples(shared[key], check_policy, item["question"]) if arm.endswith("_checked")
                             else shared[key])
                else:
                    drawn = sample(models[role], item["question"], n, **options)
                record["arms"][arm] = [s.to_dict() for s in drawn]
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            stream.flush()
            records.append(record)
            brief = " ".join(f"{a}={'ok' if is_correct(prefix_votes_from(record, a)[-1], gold) else 'x'}"
                             for a in arms)
            print(f"[{position + 1}/{len(questions)}] {record['id']}: {brief}", flush=True)
    return records


def _samples(record: dict, arm: str):
    from .search import Sample
    return [Sample.from_dict(s) for s in record["arms"][arm]]


def prefix_votes_from(record: dict, arm: str) -> list[str | None]:
    return prefix_votes(_samples(record, arm), require_verified=ARM_SPEC[arm][3])


def _bootstrap(diffs: list[int], seed: int = 6006) -> list[float]:
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(2000))
    return [round(boots[49], 4), round(boots[1949], 4)]


def summarize(records: list[dict], arms: tuple[str, ...], cost_unit: str) -> dict:
    if not records:
        raise ValueError("No records")
    for r in records:
        if set(r["arms"]) != set(arms):
            raise ValueError(f"Record {r['id']} is missing arms; refusing a partial comparison")
    cost_key = "cpu_seconds" if cost_unit == "cpu_seconds" else "latency_s"
    n = len(records)
    curves, correct_at = {}, {}
    for arm in arms:
        lengths = {len(r["arms"][arm]) for r in records}
        if len(lengths) != 1:
            raise ValueError(f"Arm {arm} has inconsistent sample counts")
        k_max = lengths.pop()
        curve, per_k = [], []
        for k in range(1, k_max + 1):
            hits, costs, tokens, abstain = [], [], [], 0
            for r in records:
                gold = Fraction(r["gold"])
                answer = prefix_votes_from(r, arm)[k - 1]
                hits.append(int(is_correct(answer, gold)))
                abstain += answer is None
                first_k = r["arms"][arm][:k]
                if any(s[cost_key] is None for s in first_k):
                    raise ValueError(f"Missing {cost_key} in arm {arm}; cost unit must be uniform")
                costs.append(sum(s[cost_key] for s in first_k))
                tokens.append(sum(s["completion_tokens"] for s in first_k))
            per_k.append(hits)
            ordered = sorted(costs)
            curve.append({"k": k, "accuracy": round(sum(hits) / n, 4),
                          "abstain_rate": round(abstain / n, 4),
                          "mean_cost": round(statistics.mean(costs), 4),
                          "p50_cost": round(ordered[(n - 1) // 2], 4),
                          "p95_cost": round(ordered[min(n - 1, int(0.95 * (n - 1) + 0.5))], 4),
                          "max_cost": round(ordered[-1], 4),
                          "cost_per_correct": round(sum(costs) / sum(hits), 4) if sum(hits) else None,
                          "mean_completion_tokens": round(statistics.mean(tokens), 1),
                          "verified_sample_rate": (round(sum(s["verified"] for r in records
                                                             for s in r["arms"][arm][:k]) / (n * k), 4)
                                                   if ARM_SPEC[arm][3] else None),
                          "unparseable_rate": round(sum(s["answer"] is None and not ARM_SPEC[arm][3]
                                                        for r in records for s in r["arms"][arm][:k])
                                                    / (n * k), 4)})
        curves[arm] = curve
        correct_at[arm] = per_k

    comparisons = {}
    for reference in (r for r in REFERENCES if r in arms):
        ref = curves[reference][0]
        ref_hits = correct_at[reference][0]
        for arm in arms:
            if not ARM_SPEC[arm][2]:
                continue
            within = [p for p in curves[arm] if p["mean_cost"] <= ref["mean_cost"]]
            entry = {"reference_accuracy": ref["accuracy"], "reference_mean_cost": ref["mean_cost"]}
            if within:
                k = within[-1]["k"]
                diffs = [a - b for a, b in zip(correct_at[arm][k - 1], ref_hits)]
                entry.update({
                    "matched_k": k, "matched_accuracy": within[-1]["accuracy"],
                    "matched_mean_cost": within[-1]["mean_cost"],
                    "accuracy_minus_reference": round(statistics.mean(diffs), 4),
                    "exploratory_paired_bootstrap_95pct": _bootstrap(diffs),
                    "wins_only_this_arm": sum(d == 1 for d in diffs),
                    "wins_only_reference": sum(d == -1 for d in diffs),
                })
            else:
                entry.update({"matched_k": None,
                              "note": "Even one sample costs more than the reference; no matched point"})
            entry["per_question_matched"] = per_question_matched(records, arm, reference, cost_key)
            reach = next((p for p in curves[arm] if p["accuracy"] >= ref["accuracy"]), None)
            entry["first_k_reaching_reference_accuracy"] = reach["k"] if reach else None
            entry["cost_ratio_at_that_k"] = (round(reach["mean_cost"] / ref["mean_cost"], 3)
                                             if reach and ref["mean_cost"] else None)
            comparisons[f"{arm}_vs_{reference}"] = entry
    for checked in ("small_constrained", "small_verified"):
        if checked not in arms or "small_vote" not in arms:
            continue
        k = min(len(curves[checked]), len(curves["small_vote"]))
        diffs = [a - b for a, b in zip(correct_at[checked][k - 1], correct_at["small_vote"][k - 1])]
        comparisons[f"verifier_effect_{checked}_vs_small_vote"] = {
            "k": k, "accuracy_difference": round(statistics.mean(diffs), 4),
            "exploratory_paired_bootstrap_95pct": _bootstrap(diffs),
            "note": f"{checked} minus small_vote with the same number of samples; isolates "
                    "structured computation + checking from extra sampling. Costs can differ.",
        }
    # Paired checker effect: each *_checked arm re-scores IDENTICAL outputs (same text,
    # tokens, seeds and model calls) as its *_vote twin; only exact post-processing differs.
    for checked_arm, vote_arm, key in CHECKED_PAIRS:
        if checked_arm not in arms or vote_arm not in arms:
            continue
        before = curves[vote_arm][-1]
        after = curves[checked_arm][-1]
        k = min(before["k"], after["k"])
        diffs = [a - b for a, b in zip(correct_at[checked_arm][k-1], correct_at[vote_arm][k-1])]
        all_checked = [s for r in records for s in r["arms"][checked_arm]]
        n_c = len(all_checked)
        comparisons[key] = {
            "same_model_outputs": True,
            "k": k,
            "accuracy_before": before["accuracy"],
            "accuracy_after": after["accuracy"],
            "accuracy_difference": round(statistics.mean(diffs), 4),
            "exploratory_paired_bootstrap_95pct": _bootstrap(diffs),
            "sample_arithmetic_error_rate": round(sum(s["verification_errors"] > 0 for s in all_checked)/n_c, 4),
            "sample_checkable_rate": round(sum(s["verification_equations"] > 0 for s in all_checked)/n_c, 4),
            "sample_vote_eligible_rate": round(sum(s["verified"] for s in all_checked)/n_c, 4),
            "mean_checker_cpu_seconds_per_question": round(sum(s["verification_cpu_seconds"] for s in all_checked)/n, 6),
            "policy_table": policy_table(records, checked_arm, k),
            "scope": "Visible explicit arithmetic equations only; neither semantics nor chain entailment is verified.",
        }
    if "small_constrained" in arms:
        drawn = [s for r in records for s in r["arms"]["small_constrained"]]
        violations = sum(any(v in (s["error"] or "") for v in GRAMMAR_VIOLATIONS) for s in drawn)
        comparisons["grammar_violation_rate"] = round(violations / len(drawn), 4) if drawn else None
    return {"paired_questions": n, "cost_unit": cost_unit, "curves": curves, "comparisons": comparisons}


def per_question_matched(records: list[dict], arm: str, reference: str, cost_key: str) -> dict:
    """Stricter fairness (v0.9): on EACH question the arm may use only as many samples as fit
    within the reference's cost on that same question. If even one sample costs more, the arm
    abstains on that question (counted wrong). Mean-matched results can hide expensive outliers."""
    hits, ks, abstain, diffs = 0, [], 0, []
    for r in records:
        budget = sum(s[cost_key] for s in r["arms"][reference])
        spent, k = 0.0, 0
        for s in r["arms"][arm]:
            if spent + s[cost_key] > budget:
                break
            spent += s[cost_key]
            k += 1
        ks.append(k)
        answer = prefix_votes_from(r, arm)[k - 1] if k else None
        abstain += answer is None
        gold = Fraction(r["gold"])
        mine, theirs = int(is_correct(answer, gold)), int(is_correct(prefix_votes_from(r, reference)[-1], gold))
        hits += mine
        diffs.append(mine - theirs)
    n = len(records)
    return {"accuracy": round(hits / n, 4), "mean_samples_affordable": round(statistics.mean(ks), 3),
            "questions_with_zero_affordable_samples": sum(k == 0 for k in ks),
            "abstain_rate": round(abstain / n, 4),
            "accuracy_minus_reference": round(statistics.mean(diffs), 4),
            "exploratory_paired_bootstrap_95pct": _bootstrap(diffs)}


def policy_table(records: list[dict], arm: str, k: int) -> dict:
    """Accuracy and abstention at k samples under every checker policy, computed from the
    stored per-sample eligibility. Choose the policy on a development set, then freeze it."""
    from .search import Sample, vote
    table = {}
    for policy in POLICIES:
        hits = abstain = eligible = total = 0
        for r in records:
            drawn = []
            for d in r["arms"][arm][:k]:
                ok = bool((d.get("verification_policies") or {}).get(policy))
                eligible += ok
                total += 1
                drawn.append(replace(Sample.from_dict(d), verified=ok))
            answer = vote(drawn, require_verified=True)
            abstain += answer is None
            hits += is_correct(answer, Fraction(r["gold"]))
        table[policy] = {"accuracy": round(hits / len(records), 4),
                         "abstain_rate": round(abstain / len(records), 4),
                         "sample_eligible_rate": round(eligible / total, 4) if total else None}
    return table


def preflight_thinking(model, budget: int) -> str | None:
    """Check that per-request thinking switches work and the budget is enforced.
    Uses its own question so it never overlaps (or warms the cache for) a dataset item."""
    question = "PREFLIGHT: A baker makes 6 trays of 12 rolls and sells 50 rolls. How many rolls are left?"
    try:
        on = model.chat("Solve it. Reply with ONLY JSON {\"answer\":\"NUMBER\"}.", question,
                        max_tokens=48 + 128, thinking=True, reasoning_budget=48)
        off = model.chat("Solve it. Reply with ONLY JSON {\"answer\":\"NUMBER\"}.", question,
                         max_tokens=64, thinking=False)
    except RuntimeError as exc:
        return f"the server rejected a thinking request ({exc})."
    if not (getattr(on, "reasoning", None) or "").strip():
        return ("thinking could not be switched ON per request (no reasoning text came back). "
                "Restart the model server with the v0.8 launcher, which no longer passes "
                "'--reasoning off':  py -m cpu_platform.launch --model qwen3-0.6b --port 8080")
    if (getattr(off, "reasoning", None) or "").strip():
        return "thinking could not be switched OFF per request; non-thinking arms would be contaminated."
    if on.completion_tokens and on.completion_tokens > 48 + 160:
        return (f"the reasoning budget is not enforced (asked for 48 thinking tokens, got "
                f"{on.completion_tokens} tokens). Update llama.cpp or lower --think-budget.")
    return None


def preflight_grammar(model) -> str | None:
    """One constrained request; returns a problem description or None if grammars work."""
    from .constrained import CONSTRAINED_PLAN_PROMPT, build_grammar, question_literals, to_checked_plan
    from cpu_first.core import json_object
    question = "A bus has 25 passengers. At the first stop, 7 leave and 9 board. How many passengers now?"
    try:
        reply = model.chat(CONSTRAINED_PLAN_PROMPT, question, max_tokens=256,
                           grammar=build_grammar(question_literals(question)))
    except RuntimeError as exc:
        return (f"the server rejected a grammar-constrained request ({exc}). Update llama.cpp, or run "
                f"without the small_constrained arm (--arms ...).")
    try:
        to_checked_plan(question, json_object(reply.content))
    except Exception as exc:  # noqa: BLE001 - any failure here means the grammar was not applied
        return (f"the server accepted the grammar but its reply does not conform ({exc}): "
                f"reply was {reply.content[:200]!r}. The chat template may be overriding custom grammars.")
    return None


def _port(url: str) -> int | None:
    return urlparse(url).port


def _launch_manifest(path: str | None, url: str, results: Path) -> dict | None:
    candidate = Path(path) if path else results / f"launch_{_port(url)}.json"
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Small+verified-search vs large model at matched CPU compute")
    p.add_argument("--dataset", default="data/gsm8k_test.jsonl",
                   help="real benchmark questions (fetch: py scripts/fetch_gsm8k.py --limit 200). "
                        "data/smoke.jsonl is a 5-question software check only.")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--small-url", default="http://127.0.0.1:8080")
    p.add_argument("--large-url", default="http://127.0.0.1:8081", help="'none' to skip the large arm")
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--samples", type=int, default=8, help="samples for sampled non-thinking arms")
    p.add_argument("--think-samples", type=int, default=4, help="samples for small_think_vote/_checked")
    p.add_argument("--think-budget", type=int, default=DEFAULT_THINK_BUDGET,
                   help="max thinking tokens per think sample (llama.cpp then forces an answer)")
    p.add_argument("--check-policy", choices=POLICIES, default=DEFAULT_POLICY,
                   help="checker rule for *_checked arms; ALL rules are also reported. Pick it on the "
                        "pilot, then freeze it for the main run.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=None,
                   help="cap for reasoning (cot) and free-form plan replies; default 512")
    p.add_argument("--seed", type=int, default=606)
    p.add_argument("--small-pid", type=int, help="llama.cpp PID (default: from launch manifest)")
    p.add_argument("--large-pid", type=int)
    p.add_argument("--small-launch-manifest")
    p.add_argument("--large-launch-manifest")
    p.add_argument("--output", default="results/experiment6.jsonl")
    p.add_argument("--confirm-frozen-eval", action="store_true",
                   help="required to run on the frozen eval split; the run is logged as one 'look'")
    p.add_argument("--no-report", action="store_true", help="skip the Markdown report")
    args = p.parse_args(argv)

    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    if args.large_url.lower() == "none":
        arms = tuple(a for a in arms if ARM_SPEC.get(a, ("",))[0] != "large")
    if not arms or any(a not in ARM_SPEC for a in arms) or len(set(arms)) != len(arms):
        p.error(f"--arms must be distinct values from {', '.join(ARMS)}")
    if args.limit < 1:
        p.error("--limit must be positive")
    out = Path(args.output)
    results = out.parent
    dataset = Path(args.dataset)
    if not dataset.is_file():
        print(f"ERROR: dataset {dataset} not found. Fetch real benchmark questions first:\n"
              f"  py -m pip install datasets\n  py scripts/fetch_gsm8k.py --limit 200\n"
              f"(data/smoke.jsonl is only a 5-question software check: pass --dataset data/smoke.jsonl)",
              file=sys.stderr)
        return 4
    try:
        split, split_manifest = registry.split_of(dataset)
    except registry.FrozenRegistryError as exc:
        print(f"ERROR: frozen evaluation guard cannot establish dataset safety: {exc}",
              file=sys.stderr)
        return 6
    eval_look = None
    if split == "eval":
        if not args.confirm_frozen_eval:
            print("ERROR: this is the FROZEN evaluation split. Run on it only with settings fixed in "
                  "advance on the dev split (data/splits/dev.jsonl). If that is the case, add "
                  "--confirm-frozen-eval; the run will be logged in data/splits/eval_runs.jsonl.",
                  file=sys.stderr)
            return 6
        problems = registry.validate(Path(split_manifest["splits"]["eval"]["file"]).parent)
        if registry.sha256_text(dataset) != split_manifest["splits"]["eval"]["sha256"]:
            problems.append(f"{dataset} contains frozen eval questions but is not the frozen eval file "
                            f"({split_manifest['splits']['eval']['file']}); use that file unchanged")
        if problems:
            print("ERROR: frozen splits fail validation: " + "; ".join(problems), file=sys.stderr)
            return 6
    questions = load_questions(dataset, args.limit)
    if len(questions) < 30:
        print("WARNING: fewer than 30 questions - a software check, not evidence.", file=sys.stderr)

    roles = {ARM_SPEC[a][0] for a in arms}
    urls = {"small": args.small_url, "large": args.large_url}
    manifests = {role: _launch_manifest(getattr(args, f"{role}_launch_manifest"), urls[role], results)
                 for role in roles}
    models = {}
    for role in sorted(roles):
        try:
            models[role] = LocalLlama(urls[role], timeout=600)
        except (OSError, RuntimeError, ValueError) as exc:
            port = _port(urls[role])
            hint = "qwen3-0.6b" if role == "small" else "qwen3-8b (or qwen3-4b)"
            print(f"ERROR: no llama.cpp server answering at {urls[role]} ({role} model): {exc}\n"
                  f"Start it in a SEPARATE window and leave that window alone:\n"
                  f"  py -m cpu_platform.launch --model {hint} --port {port}\n"
                  + ("Or skip the large-model arms with --large-url none\n" if role == "large" else ""),
                  file=sys.stderr)
            return 2
    if "small_constrained" in arms:
        problem = preflight_grammar(models["small"])
        if problem:
            print("ERROR: " + problem, file=sys.stderr)
            return 3
    think_roles = {ARM_SPEC[a][0] for a in arms if ARM_SPEC[a][1] == "think"}
    for role in sorted(think_roles):
        problem = preflight_thinking(models[role], args.think_budget)
        if problem:
            print(f"ERROR ({role} model): " + problem, file=sys.stderr)
            return 5
    if len(roles) == 2 and models["small"].model == models["large"].model:
        print("WARNING: small and large servers report the same model id.", file=sys.stderr)

    meters, meter_notes = {}, []
    for role in roles:
        explicit = getattr(args, f"{role}_pid")
        pid = explicit or ((manifests[role] or {}).get("llama_cpp") or {}).get("pid")
        if pid is None:
            continue
        try:
            meters[role] = make_cpu_meter(pid)
        except RuntimeError as exc:
            if explicit:
                p.error(str(exc))
            meter_notes.append(f"{role}: {exc}")
    cost_unit = "cpu_seconds" if set(meters) == roles else "model_wall_seconds"
    if cost_unit != "cpu_seconds":
        meters = {}  # one unit for every arm, never mixed
        print("NOTE: costs use model wall-clock seconds. For CPU-seconds, install psutil and start the "
              "servers with cpu_platform.launch (or pass --small-pid/--large-pid).", file=sys.stderr)

    servers = {role: describe_server(urls[role], results / "model_hashes.json") for role in roles}
    settings = {"arms": arms, "samples": args.samples, "temperature": args.temperature,
                "max_tokens": args.max_tokens, "seed": args.seed,
                "think_samples": args.think_samples, "think_budget": args.think_budget,
                "check_policy": args.check_policy}
    if split == "eval":
        # Logged BEFORE running, so aborted runs also count as looks at the held-out data.
        eval_look = registry.log_eval_run({
            "utc": datetime.now(timezone.utc).isoformat(), "output": str(out), "limit": args.limit,
            "settings": settings, "models": {r: servers[r].get("model_sha256") for r in roles},
            "release": __version__}, Path(split_manifest["splits"]["eval"]["file"]).parent)
        print(f"Frozen eval run #{eval_look} logged.", file=sys.stderr)

    from cpu_first.benchmark import RSSSampler
    pids = {role: ((manifests[role] or {}).get("llama_cpp") or {}).get("pid") for role in roles}
    samplers = {role: RSSSampler(getattr(args, f"{role}_pid") or pids[role]) for role in roles}
    for sampler in samplers.values():
        sampler.__enter__()
    try:
        records = run_experiment(models, questions, out, arms=arms, samples=args.samples,
                                 temperature=args.temperature, seed=args.seed,
                                 max_tokens=args.max_tokens, meters=meters,
                                 think_samples=args.think_samples, think_budget=args.think_budget,
                                 check_policy=args.check_policy)
    finally:
        for sampler in samplers.values():
            sampler.__exit__(None, None, None)
    peak_rss = {role: samplers[role].peak_mb for role in roles}
    summary = summarize(records, arms, cost_unit)
    report = {
        "experiment": "006 matched-compute small+search vs large", "release": __version__,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {"file": str(dataset), "sha256": sha256_file(dataset), "questions": len(questions)},
        "source_sha256": source_sha256(),
        "models": {role: {"url": urls[role], "server_model_id": models[role].model,
                          "launch_manifest": manifests[role]} for role in roles},
        "settings": settings,
        "split": {"name": split or "unregistered", "eval_run_number": eval_look,
                  "manifest_created_utc": (split_manifest or {}).get("created_utc")},
        "servers": servers,
        "peak_model_rss_mb": peak_rss,
        "evidence": evidence_table(cost_unit=cost_unit, servers=servers, manifests=manifests,
                                   split=split, peak_rss_mb=peak_rss),
        "hardware": detect(measure_bandwidth=False),
        "metering_notes": meter_notes,
        "results": summary,
        "limitations": [
            "CPU-only execution is evidenced by launch manifests and logs, not attested hardware.",
            "Both servers should run on the same machine, one request at a time; check the manifests.",
            "Wall-clock cost includes scheduling noise; CPU-seconds exclude memory-stall energy.",
            "Bootstrap intervals are exploratory; use 100+ held-out questions before drawing conclusions.",
            "The verifier checks arithmetic and literal provenance only, not the interpretation of the question.",
        ],
        "warning": "SMOKE TEST ONLY" if len(questions) < 30 else "Exploratory result",
    }
    path = out.with_suffix(".summary.json")
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.no_report:
        from .report import write_bundle
        bundle = write_bundle(out, dataset)
        print("Report:", bundle["report"])
    print(json.dumps(summary["comparisons"], indent=2))
    print("Summary:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
