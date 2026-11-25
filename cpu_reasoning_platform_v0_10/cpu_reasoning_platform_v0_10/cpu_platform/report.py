"""v0.9 result bundle: failure labels, failures.jsonl and a readable Markdown report.

    python -m cpu_platform.report results/pilot_v08.jsonl [--dataset data/gsm8k_test.jsonl]

Works on any experiment-6 output, including v0.8 runs and interrupted runs (the summary
is recomputed from the per-question lines if the .summary.json is missing).

Failure labels are AUTOMATIC HEURISTICS from visible text, meant to direct human review:
  unparseable_output               no usable answer in any sample
  no_answer_after_thinking         thinking ran but no final answer was produced
  abstained_checker_filtered_all   samples had answers, the checker excluded all of them
  arithmetic_error                 a sample behind the wrong answer wrote a wrong equation
  ungrounded_numbers               ...used numbers found neither in the question nor earlier steps
  wrong_operation_or_interpretation arithmetic checked and correct, answer still wrong
  wrong_answer_unexplained         wrong answer with no checkable working
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import json
from pathlib import Path
import sys

from .cot_check import audit_reasoning

LABELS = ("unparseable_output", "no_answer_after_thinking", "abstained_checker_filtered_all",
          "arithmetic_error", "ungrounded_numbers", "wrong_operation_or_interpretation",
          "wrong_answer_unexplained")
EXAMPLES_PER_LABEL = 3
TRACE_CHARS = 700


def _correct(answer, gold) -> bool:
    if answer is None:
        return False
    try:
        return Fraction(answer) == Fraction(gold)
    except (ValueError, ZeroDivisionError):
        return False


def label_failure(arm: str, samples: list[dict], final: str | None, question: str | None) -> tuple[str, dict]:
    """(label, detail) for one wrong or missing final answer."""
    if final is None:
        if any(s.get("answer") is not None for s in samples):
            return "abstained_checker_filtered_all", {}
        if any(s.get("thinking_chars") for s in samples):
            return "no_answer_after_thinking", {}
        return "unparseable_output", {"errors": sorted({(s.get("error") or "")[:80] for s in samples})[:3]}
    backing = [s for s in samples if s.get("answer") == final]
    audits = [audit_reasoning(s["raw_content"], question) for s in backing if s.get("raw_content")]
    if not audits:
        return "wrong_answer_unexplained", {}
    if any(a.error_count for a in audits):
        bad = [f"{e.expression} = {e.claimed} (actual {e.actual})"
               for a in audits for e in a.equations if e.correct is False]
        return "arithmetic_error", {"wrong_equations": bad[:3]}
    if question is not None and any(a.ungrounded_count for a in audits):
        found = sorted({x for a in audits for e in a.equations for x in e.ungrounded})
        return "ungrounded_numbers", {"ungrounded": found[:5]}
    if any(a.checked_count for a in audits):
        return "wrong_operation_or_interpretation", {}
    return "wrong_answer_unexplained", {}


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _questions(dataset: Path | None) -> dict[str, str]:
    if dataset is None or not dataset.is_file():
        return {}
    out = {}
    for i, row in enumerate(_load(dataset)):
        out[str(row.get("id", i))] = row["question"]
    return out


def build_failures(records: list[dict], questions: dict[str, str]) -> list[dict]:
    from .experiment6 import prefix_votes_from
    rows = []
    for r in records:
        for arm, samples in r["arms"].items():
            final = prefix_votes_from(r, arm)[-1]
            if _correct(final, r["gold"]):
                continue
            label, detail = label_failure(arm, samples, final, questions.get(r["id"]))
            rows.append({"id": r["id"], "arm": arm, "gold": r["gold"], "answer": final,
                         "label": label, "detail": detail})
    return rows


def _fmt(x, pct=False):
    if x is None:
        return "–"
    return f"{100 * x:.1f}%" if pct else (f"{x:.2f}" if isinstance(x, float) else str(x))


def _interval(ci):
    return f"[{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]" if ci else "–"


def render(summary: dict, records: list[dict], failures: list[dict], questions: dict[str, str]) -> str:
    res = summary.get("results") or summary
    lines = ["# Experiment report", ""]
    split = (summary.get("split") or {}).get("name", "unregistered")
    lines += [f"- Release: {summary.get('release', '?')}  |  questions: {res['paired_questions']}  |  "
              f"split: **{split}**  |  cost unit: {res['cost_unit']}",
              f"- Dataset: {(summary.get('dataset') or {}).get('file', '?')}  "
              f"(sha256 {str((summary.get('dataset') or {}).get('sha256', '?'))[:12]})"]
    settings = summary.get("settings") or {}
    if settings:
        lines.append("- Settings: " + ", ".join(f"{k}={v}" for k, v in settings.items() if k != "arms"))
    for role, info in (summary.get("servers") or {}).items():
        lines.append(f"- {role} model: {info.get('model_path', '?')} sha256 {str(info.get('model_sha256'))[:12]} "
                     f"| llama.cpp {str(info.get('build_info'))[:40]}")
    if res["paired_questions"] < 30:
        lines += ["", "> **Fewer than 30 questions: a software check, not evidence.**"]
    evidence = summary.get("evidence")
    lines += ["", "## Evidence", ""]
    if evidence:
        lines += ["| Item | Status |", "|---|---|"] + [f"| {k} | {v} |" for k, v in evidence.items()]
    else:
        lines.append("Evidence table: NOT RUN (pre-v0.9 output).")

    lines += ["", "## Arms (at the full sample count)", "",
              "| Arm | Accuracy | k=1 accuracy | Abstain | Mean cost | p95 cost | Cost per correct | Tokens |",
              "|---|---|---|---|---|---|---|---|"]
    for arm, curve in res["curves"].items():
        last, first = curve[-1], curve[0]
        lines.append(f"| {arm} (k={last['k']}) | **{_fmt(last['accuracy'], True)}** | "
                     f"{_fmt(first['accuracy'], True) if len(curve) > 1 else '–'} | "
                     f"{_fmt(last['abstain_rate'], True)} | {_fmt(last['mean_cost'])} | "
                     f"{_fmt(last.get('p95_cost'))} | {_fmt(last.get('cost_per_correct'))} | "
                     f"{_fmt(last.get('mean_completion_tokens'))} |")

    comps = res.get("comparisons") or {}
    vs = {k: v for k, v in comps.items() if "_vs_large" in k}
    if vs:
        lines += ["", "## Against the large model at matched cost", "",
                  "| Comparison | Reference acc. | Matched k | Acc. difference (95% CI) | Per-question matched diff (95% CI) | k to reach reference |",
                  "|---|---|---|---|---|---|"]
        for name, c in vs.items():
            pq = c.get("per_question_matched") or {}
            lines.append(
                f"| {name} | {_fmt(c.get('reference_accuracy'), True)} | {_fmt(c.get('matched_k'))} | "
                + (f"{100 * c['accuracy_minus_reference']:+.1f} pts {_interval(c.get('exploratory_paired_bootstrap_95pct'))}"
                   if c.get("matched_k") else "no matched point")
                + " | " + (f"{100 * pq['accuracy_minus_reference']:+.1f} pts {_interval(pq.get('exploratory_paired_bootstrap_95pct'))}"
                           if pq else "–")
                + f" | {_fmt(c.get('first_k_reaching_reference_accuracy'))} |")
    checker = {k: v for k, v in comps.items() if k.startswith("paired_") and "filter_effect" in k}
    if checker:
        lines += ["", "## What the checker adds (same samples, checked vs unchecked)", ""]
        for name, c in checker.items():
            lines.append(f"- **{name}**: {_fmt(c['accuracy_before'], True)} → {_fmt(c['accuracy_after'], True)} "
                         f"({100 * c['accuracy_difference']:+.1f} pts, 95% CI {_interval(c.get('exploratory_paired_bootstrap_95pct'))}); "
                         f"samples with a wrong equation {_fmt(c.get('sample_arithmetic_error_rate'), True)}, "
                         f"checkable {_fmt(c.get('sample_checkable_rate'), True)}")
            table = c.get("policy_table") or {}
            if table:
                lines += ["", "  | Policy | Accuracy | Abstain | Samples eligible |", "  |---|---|---|---|"]
                lines += [f"  | {p} | {_fmt(t['accuracy'], True)} | {_fmt(t['abstain_rate'], True)} | "
                          f"{_fmt(t.get('sample_eligible_rate'), True)} |" for p, t in table.items()]
                lines.append("")

    counts = defaultdict(Counter)
    for f in failures:
        counts[f["arm"]][f["label"]] += 1
    used = [l for l in LABELS if any(c[l] for c in counts.values())]
    lines += ["", "## Why answers were wrong (automatic heuristic labels)", ""]
    if used:
        lines += ["| Arm | " + " | ".join(used) + " |", "|---|" + "---|" * len(used)]
        for arm in res["curves"]:
            lines.append(f"| {arm} | " + " | ".join(str(counts[arm][l]) for l in used) + " |")
    else:
        lines.append("No failures.")

    by_label = defaultdict(list)
    preference = ("small_think_checked", "small_cot_checked", "small_think", "small_cot",
                  "small_think_vote", "small_cot_vote")
    focus = [a for a in preference if a in res["curves"] and counts[a]][:2]
    record_by_id = {r["id"]: r for r in records}
    for f in failures:
        if f["arm"] in focus:
            by_label[f["label"]].append(f)
    if by_label:
        lines += ["", f"## Example traces ({', '.join(focus)})", ""]
    for label in LABELS:
        for f in by_label.get(label, [])[:EXAMPLES_PER_LABEL]:
            r = record_by_id[f["id"]]
            sample = next((s for s in r["arms"][f["arm"]] if s.get("answer") == f["answer"]),
                          r["arms"][f["arm"]][0])
            q = questions.get(f["id"], "(question text not available: pass --dataset)")
            lines += [f"### {label}: question {f['id']} ({f['arm']})", "",
                      f"**Question:** {q}", "", f"**Gold:** {f['gold']}  |  **Answer:** {f['answer']}", ""]
            if f["detail"]:
                lines.append("**Checker:** " + "; ".join(f"{k}: {v}" for k, v in f["detail"].items()))
                lines.append("")
            trace = (sample.get("raw_content") or "").strip()
            if trace:
                cut = trace if len(trace) <= TRACE_CHARS else trace[:TRACE_CHARS // 2] + "\n…\n" + trace[-TRACE_CHARS // 2:]
                lines += ["```text", cut.replace("```", "'''"), "```", ""]
    lines += ["", "## Caveats", "",
              "- Failure labels are heuristics from visible text; review examples before drawing conclusions.",
              "- The checker verifies arithmetic and number provenance only, never the reading of the question.",
              "- Bootstrap intervals are exploratory; GSM8K is public and may be in training data.",
              "- Results on the dev split guide the next change; only the frozen eval split supports a claim."]
    return "\n".join(lines) + "\n"


def write_bundle(out_jsonl: Path, dataset: Path | None = None) -> dict:
    """Write <out>.failures.jsonl and <out>.report.md next to the experiment output."""
    from .experiment6 import summarize
    records = _load(out_jsonl)
    if not records:
        raise ValueError(f"{out_jsonl} has no records")
    summary_path = out_jsonl.with_suffix(".summary.json")
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if dataset is None and summary.get("dataset", {}).get("file"):
            dataset = Path(summary["dataset"]["file"])
    else:  # interrupted run: recompute from the lines that were written
        arms = tuple(records[0]["arms"])
        records = [r for r in records if set(r["arms"]) == set(arms)]
        metered = all(s.get("cpu_seconds") is not None for r in records for ss in r["arms"].values() for s in ss)
        summary = {"results": summarize(records, arms, "cpu_seconds" if metered else "model_wall_seconds"),
                   "note": "recomputed from an incomplete run (no .summary.json)"}
    questions = _questions(dataset)
    failures = build_failures(records, questions)
    failures_path = out_jsonl.with_suffix(".failures.jsonl")
    failures_path.write_text("".join(json.dumps(f) + "\n" for f in failures), encoding="utf-8")
    report_path = out_jsonl.with_suffix(".report.md")
    report_path.write_text(render(summary, records, failures, questions), encoding="utf-8")
    return {"failures": failures_path, "report": report_path, "failure_count": len(failures)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Failure labels and Markdown report for an experiment-6 run")
    p.add_argument("results", help="experiment output .jsonl")
    p.add_argument("--dataset", help="question file (default: the one recorded in the summary)")
    args = p.parse_args(argv)
    out = write_bundle(Path(args.results), Path(args.dataset) if args.dataset else None)
    print(f"Report: {out['report']}\nFailures: {out['failures']} ({out['failure_count']} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
