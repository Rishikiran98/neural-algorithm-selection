"""v0.9 tests: frozen splits (EVAL-01), provenance and per-question cost (EVAL-02),
failure report (EVAL-03), verifier grounding and adversarial fixtures (VERIFY-01).
Offline; the fake server is not a model and proves nothing about accuracy."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest

from cpu_platform import datasets as registry
from cpu_platform import experiment6, report
from cpu_platform.cot_check import POLICIES, audit_reasoning
from cpu_platform.provenance import NOT_RUN, describe_server, evidence_table
from cpu_platform.search import Sample, check_cot_samples

try:
    from fake_llama import FakeLlama
except ImportError:
    from tests.fake_llama import FakeLlama

ROOT = Path(__file__).resolve().parent.parent
STORE = "A store charges $12 per notebook. You buy 3 notebooks and pay with $50. How much change?"


def synthetic_source(directory: Path, n: int = 300) -> tuple[Path, Path]:
    rng = random.Random(1)
    rows = []
    for i in range(n):
        a, b, c = rng.randint(2, 90), rng.randint(2, 40), rng.randint(2, 9)
        q = (f"Person {i} has {a} stamps and finds {b} more on day {i}. Then {c} are given to "
             f"each of {rng.randint(2, 5)} friends in town {i * 7}. How many are left?")
        rows.append({"id": str(i), "question": q, "answer": f"work #### {a + b}"})
    rows[10]["question"] = rows[5]["question"]                                       # exact duplicate
    rows[11]["question"] = rows[6]["question"].replace("How many are left?", "How many remain?")  # near
    rows[12]["question"] = ("Maya has 12 apples and buys 8 more. She splits them equally into 5 boxes. "
                            "How many apples are in each box?")                       # prompt echo
    rows[13]["answer"] = "no marker"                                                  # invalid gold
    rows[14]["question"] = rows[20]["question"].replace("stamps", "coins")            # near dup of DEV item
    source, pilot = directory / "full.jsonl", directory / "pilot.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    pilot.write_text("".join(json.dumps(rows[i]) + "\n" for i in (6, 20, 21, 22, 23)))
    return source, pilot


class TestFrozenSplits(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.source, self.pilot = synthetic_source(self.dir)
        self.out = self.dir / "splits"
        self.manifest = registry.freeze(self.source, self.pilot, self.out, eval_size=100, dev_size=50, seed=9)

    def tearDown(self):
        self.tmp.cleanup()

    def test_cleaning_and_disjoint_splits(self):
        removed = self.manifest["cleaning"]["removed"]
        self.assertEqual(removed["exact_duplicates"], ["10"])
        self.assertIn("11", removed["near_duplicates"])
        self.assertEqual(removed["prompt_echo"], ["12"])
        self.assertEqual(removed["invalid_gold"], ["13"])
        dev, ev = self.manifest["splits"]["dev"]["ids"], self.manifest["splits"]["eval"]["ids"]
        self.assertEqual(dev, ["6", "20", "21", "22", "23"])          # pilot questions become dev
        self.assertEqual(len(ev), 100)
        self.assertFalse(set(dev) & set(ev))
        self.assertNotIn("14", ev)                                    # near duplicate of a dev item
        self.assertEqual(registry.validate(self.out), [])

    def test_frozen_means_frozen(self):
        with self.assertRaises(SystemExit):
            registry.freeze(self.source, self.pilot, self.out, eval_size=100, dev_size=50, seed=9)
        again = registry.freeze(self.source, self.pilot, self.out, eval_size=100, dev_size=50, seed=9, force=True)
        self.assertEqual(again["splits"]["eval"]["ids"], self.manifest["splits"]["eval"]["ids"])  # deterministic

    def test_validate_detects_tampering(self):
        path = self.out / "eval.jsonl"
        path.write_text(path.read_text() + "not json\n")
        problems = registry.validate(self.out)
        self.assertTrue(any("modified after freezing" in p for p in problems))
        self.assertTrue(any("unreadable" in p for p in problems))

    def test_split_detection_and_eval_log(self):
        self.assertEqual(registry.split_of(self.out / "eval.jsonl")[0], "eval")
        self.assertEqual(registry.split_of(self.out / "dev.jsonl")[0], "dev")
        self.assertEqual(registry.split_of(self.source)[0], "eval")    # contains eval questions
        self.assertEqual(registry.split_of(self.pilot)[0], "dev")      # the pilot file itself
        other = self.dir / "other.jsonl"
        other.write_text(json.dumps({"id": "nope", "question": "q", "answer": "#### 1"}) + "\n")
        self.assertIsNone(registry.split_of(other)[0])
        self.assertEqual(registry.log_eval_run({"x": 1}, self.out), 1)
        self.assertEqual(registry.log_eval_run({"x": 2}, self.out), 2)

    def test_manifest_records_per_question_fingerprints(self):
        self.assertEqual(self.manifest["schema"], "cpu-reasoning-splits/2")
        for name in ("dev", "eval"):
            rows = registry.read_jsonl(self.out / f"{name}.jsonl")
            self.assertEqual(
                self.manifest["splits"][name]["question_sha256"],
                [registry.question_sha256(r["question"]) for r in rows])
        self.assertEqual(registry.validate(self.out), [])

    def test_copied_eval_question_new_id_is_detected(self):
        original = registry.read_jsonl(self.out / "eval.jsonl")[0]
        changed = {**original, "id": "totally-new-id"}
        path = self.dir / "renamed_eval.jsonl"
        registry.write_jsonl(path, [changed])
        self.assertEqual(registry.split_of(path)[0], "eval")

    def test_copied_eval_normalized_text_and_mixed_file_are_detected(self):
        original = registry.read_jsonl(self.out / "eval.jsonl")[0]
        changed = {**original, "id": "new-id",
                   "question": "  " + original["question"].upper() + "  "}
        self.assertEqual(registry.question_sha256(changed["question"]),
                         registry.question_sha256(original["question"]))
        unrelated = {"id": "unknown", "question": "A solitary violet umbrella was left outside.",
                     "answer": "#### 2"}
        path = self.dir / "mixed.jsonl"
        registry.write_jsonl(path, [unrelated, changed])
        self.assertEqual(registry.split_of(path)[0], "eval")

    def test_dev_copy_with_changed_id_still_recognized(self):
        dev = registry.read_jsonl(self.out / "dev.jsonl")[0]
        path = self.dir / "renamed_dev.jsonl"
        registry.write_jsonl(path, [{**dev, "id": "different"}])
        self.assertEqual(registry.split_of(path)[0], "dev")

    def test_id_collision_without_question_match_is_not_eval(self):
        ev = registry.read_jsonl(self.out / "eval.jsonl")[0]
        path = self.dir / "unrelated_reused_id.jsonl"
        registry.write_jsonl(path, [{**ev, "question": "An unrelated question about a violet umbrella."}])
        self.assertIsNone(registry.split_of(path)[0])

    def test_v1_manifest_still_guards_copied_questions_without_refreeze(self):
        # v0.9 users already have frozen evaluation questions. Never require --force.
        old_manifest = json.loads((self.out / registry.MANIFEST).read_text())
        old_manifest["schema"] = "cpu-reasoning-splits/1"
        for split in old_manifest["splits"].values():
            del split["question_sha256"]
        (self.out / registry.MANIFEST).write_text(json.dumps(old_manifest))
        ev = registry.read_jsonl(self.out / "eval.jsonl")[0]
        copy = self.dir / "legacy_relabelled.jsonl"
        registry.write_jsonl(copy, [{**ev, "id": "legacy-different"}])
        self.assertEqual(registry.validate(self.out), [])
        self.assertEqual(registry.split_of(copy)[0], "eval")

    def test_modified_frozen_eval_fails_closed(self):
        path = self.out / "eval.jsonl"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        other = self.dir / "unrelated.jsonl"
        registry.write_jsonl(other, [{"id": "z", "question": "A question about green ladders.",
                                      "answer": "#### 2"}])
        with self.assertRaisesRegex(registry.FrozenRegistryError, "frozen eval split modified"):
            registry.split_of(other)

    def test_manifest_fingerprint_tampering_is_detected(self):
        m = json.loads((self.out / registry.MANIFEST).read_text())
        m["splits"]["eval"]["question_sha256"][0] = "0" * 64
        (self.out / registry.MANIFEST).write_text(json.dumps(m))
        self.assertTrue(any("question fingerprints differ" in p for p in registry.validate(self.out)))
        with self.assertRaises(registry.FrozenRegistryError):
            registry.split_of(self.out / "dev.jsonl")

    def test_cli_status(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(registry.main(["status", "--dir", str(self.out)]), 0)
        self.assertIn("eval: 100 questions", out.getvalue())


def run_cli(args):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
        code = experiment6.main(args)
    return code, err.getvalue()


class TestEvalGuardAndProvenance(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        # Freeze splits made of the five smoke questions (which the fake server knows).
        smoke = [json.loads(l) for l in (ROOT / "data" / "smoke.jsonl").read_text().splitlines()]
        rows = [{"id": str(i), "question": r["question"], "answer": r["answer"]} for i, r in enumerate(smoke)]
        rows[0]["question"] = rows[0]["question"].replace("Maya", "Nina")  # avoid the prompt-echo rule
        src = self.dir / "src.jsonl"
        src.write_text("".join(json.dumps(r) + "\n" for r in rows))
        pilot = self.dir / "pilot.jsonl"
        pilot.write_text(json.dumps(rows[4]) + "\n")
        self.splits = self.dir / "splits"
        registry.freeze(src, pilot, self.splits, eval_size=3, dev_size=1, seed=1)
        self.weights = self.dir / "model.gguf"
        self.weights.write_bytes(b"fake weights " * 1000)

    def tearDown(self):
        self.tmp.cleanup()

    def test_eval_split_needs_confirmation_and_is_logged(self):
        small = FakeLlama("Qwen3-0.6B", accuracy=0.7, model_path=str(self.weights))
        try:
            base = ["--dataset", str(self.splits / "eval.jsonl"), "--small-url", small.url,
                    "--large-url", "none", "--arms", "small_cot,small_cot_vote,small_cot_checked",
                    "--samples", "2", "--output", str(self.dir / "r" / "ev.jsonl")]
            code, err = run_cli(base)
            self.assertEqual(code, 6)
            self.assertIn("FROZEN", err)
            code, err = run_cli(base + ["--confirm-frozen-eval"])
            self.assertEqual(code, 0, err)
            # A copy containing eval questions is still treated as eval, and refused as not-the-frozen-file.
            leak = self.dir / "leak.jsonl"
            leak.write_text((self.splits / "eval.jsonl").read_text() + json.dumps(
                {"id": "x", "question": "Nina has 1 apple. How many?", "answer": "#### 1"}) + "\n")
            code, err = run_cli([a if a != str(self.splits / "eval.jsonl") else str(leak) for a in base]
                                + ["--confirm-frozen-eval"])
            self.assertEqual(code, 6)
            self.assertIn("not the frozen eval file", err)
        finally:
            small.close()
        summary = json.loads((self.dir / "r" / "ev.summary.json").read_text())
        self.assertEqual(summary["split"]["name"], "eval")
        self.assertEqual(summary["split"]["eval_run_number"], 1)
        self.assertEqual(summary["evidence"]["held_out_evaluation"], "RUN on frozen eval split")
        self.assertEqual(summary["servers"]["small"]["model_sha256"],
                         hashlib.sha256(self.weights.read_bytes()).hexdigest())
        self.assertEqual(summary["evidence"]["model_weights_sha256"], "RECORDED")
        self.assertEqual(summary["servers"]["small"]["build_info"], "b9999-fake")
        self.assertTrue((self.dir / "r" / "ev.report.md").is_file())
        self.assertEqual(len(registry.eval_runs(self.splits)), 1)

    def test_eval_copy_with_changed_id_is_blocked_before_model_calls(self):
        ev = registry.read_jsonl(self.splits / "eval.jsonl")[0]
        relabelled = self.dir / "renamed_eval.jsonl"
        registry.write_jsonl(relabelled, [{**ev, "id": "external-row"}])
        code, err = run_cli(["--dataset", str(relabelled), "--small-url", "http://127.0.0.1:1",
                             "--large-url", "none", "--arms", "small_cot",
                             "--confirm-frozen-eval", "--output", str(self.dir / "r" / "copy.jsonl")])
        self.assertEqual(code, 6)
        self.assertIn("not the frozen eval file", err)
        self.assertEqual(registry.eval_runs(self.splits), [])

    def test_corrupted_frozen_file_blocks_even_unrelated_runs(self):
        (self.splits / "eval.jsonl").write_text("tampered", encoding="utf-8")
        code, err = run_cli(["--dataset", str(self.splits / "dev.jsonl"),
                             "--small-url", "http://127.0.0.1:1",
                             "--large-url", "none", "--arms", "small_cot",
                             "--output", str(self.dir / "bad.jsonl")])
        self.assertEqual(code, 6)
        self.assertIn("cannot establish dataset safety", err)

    def test_model_hash_is_cached(self):
        small = FakeLlama("Qwen3-0.6B", accuracy=1, model_path=str(self.weights))
        try:
            cache = self.dir / "hashes.json"
            first = describe_server(small.url, cache)
            second = describe_server(small.url, cache)
        finally:
            small.close()
        self.assertEqual(first["model_sha256_note"], "computed")
        self.assertEqual(second["model_sha256_note"], "cached")
        self.assertEqual(first["model_sha256"], second["model_sha256"])

    def test_missing_props_is_not_run(self):
        small = FakeLlama("Qwen3-0.6B", accuracy=1)  # no model_path -> /props 404
        try:
            info = describe_server(small.url, self.dir / "h.json")
        finally:
            small.close()
        self.assertEqual(info["model_sha256"], NOT_RUN)

    def test_evidence_never_passes_silently(self):
        table = evidence_table(cost_unit="model_wall_seconds", servers={"small": {"model_sha256": NOT_RUN}},
                               manifests={"small": {"cpu_only": {"log_scan": {"verdict": "gpu_visible_or_used"}}}},
                               split="dev", peak_rss_mb={"small": None})
        self.assertTrue(table["cpu_only_log_check"].startswith("FAILED"))
        self.assertIn("NOT RUN", table["cost_metering"])
        for key in ("model_weights_sha256", "peak_model_rss", "energy_joules", "thermal_throttling"):
            self.assertEqual(table[key], NOT_RUN)
        self.assertIn(NOT_RUN, table["held_out_evaluation"])


def S(i, answer, cost, verified=False):
    return Sample(i, answer, verified, None, 0, 10, cost, None).to_dict()


class TestPerQuestionMatching(unittest.TestCase):
    def test_expensive_outlier_is_not_hidden_by_the_mean(self):
        # Reference costs 10 on each question. The arm is cheap on q1 and very expensive on q2:
        # mean-matching affords k=2, but on q2 not even one sample fits the per-question budget.
        records = [
            {"id": "1", "gold": "4", "arms": {"large_cot": [S(0, "4", 10.0)],
                                              "small_cot_vote": [S(0, "4", 1.0), S(1, "4", 1.0)]}},
            {"id": "2", "gold": "5", "arms": {"large_cot": [S(0, "5", 10.0)],
                                              "small_cot_vote": [S(0, "5", 11.0), S(1, "5", 1.0)]}},
        ]
        res = experiment6.summarize(records, ("large_cot", "small_cot_vote"), "model_wall_seconds")
        cmp = res["comparisons"]["small_cot_vote_vs_large_cot"]
        self.assertEqual(cmp["matched_k"], 2)
        self.assertEqual(cmp["matched_accuracy"], 1.0)
        pq = cmp["per_question_matched"]
        self.assertEqual(pq["questions_with_zero_affordable_samples"], 1)
        self.assertEqual(pq["accuracy"], 0.5)
        curve = res["curves"]["small_cot_vote"][-1]
        self.assertEqual((curve["max_cost"], curve["cost_per_correct"]), (12.0, 7.0))


class TestAdversarialVerifier(unittest.TestCase):
    """VERIFY-01: no fixture may come out labelled as semantically verified."""

    FIXTURES = {
        # name: (question, reasoning, final answer, expected status, grounded?, supported?)
        "honest": (STORE, "12 * 3 = 36. 50 - 36 = 14.", "14", "checked_pass", True, True),
        "correct_but_irrelevant": (STORE, "2 + 2 = 4. So the change is 14.", "14", "checked_pass", False, False),
        "wrong_operation_correct_arithmetic": (STORE, "12 + 3 = 15. 50 - 15 = 35.", "35", "checked_pass", True, True),
        "invented_number": (STORE, "12 * 3 = 36. 70 - 36 = 34.", "34", "checked_pass", False, True),
        "bad_unit_conversion": ("He works 2 hours. How many minutes is that?", "2 * 100 = 200.", "200",
                                "checked_pass", False, True),
        "inconsistent_rounding": ("Split 10 dollars among 3 people, then add it back up.",
                                  "10 / 3 = 3.3. 3.3 * 3 = 10.", "10", "arithmetic_error", False, False),
        "missing_steps": (STORE, "The total cost is 36. 50 - 36 = 14.", "14", "checked_pass", False, True),
        "no_working": (STORE, "The change is 14 dollars.", "14", "no_checkable_equations", False, False),
    }

    def test_fixtures(self):
        for name, (q, text, answer, status, grounded, supported) in self.FIXTURES.items():
            with self.subTest(name):
                audit = audit_reasoning(text, q)
                self.assertEqual(audit.status, status)
                self.assertEqual(audit.eligible(answer, "grounded"), grounded)
                self.assertEqual(audit.supports(answer), supported)
                evidence = audit.evidence(answer)
                self.assertEqual(evidence["operation_semantics"], "NOT CHECKED")
                self.assertEqual(evidence["problem_interpretation"], "NOT CHECKED")
                self.assertFalse(any("SEMANTIC" in str(v).upper() and "NOT" not in str(v).upper()
                                     for v in evidence.values()))

    def test_known_limitation_is_explicit(self):
        # Right arithmetic on the wrong operation passes every arithmetic rule: documented, not hidden.
        q, text, answer, *_ = self.FIXTURES["wrong_operation_correct_arithmetic"]
        audit = audit_reasoning(text, q)
        self.assertTrue(all(audit.eligible(answer, p) for p in POLICIES))
        self.assertEqual(audit.evidence(answer)["operation_semantics"], "NOT CHECKED")

    def test_question_enables_grounding_in_sample_checks(self):
        s = [Sample(0, "34", False, None, 0, 10, 1.0, None, raw_content="12 * 3 = 36. 70 - 36 = 34.")]
        with_q = check_cot_samples(s, "grounded", STORE)[0]
        without_q = check_cot_samples(s, "grounded")[0]
        self.assertFalse(with_q.verified)
        self.assertEqual(with_q.verification_evidence["input_grounding"], "1 UNGROUNDED OPERAND(S)")
        self.assertIn("NOT CHECKED", without_q.verification_evidence["input_grounding"])
        self.assertTrue(check_cot_samples(s, "exclude_errors", STORE)[0].verified)


class TestReport(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.out = Path(self.tmp.name) / "run.jsonl"
        small, large = FakeLlama("Qwen3-0.6B", accuracy=0.5), FakeLlama("Qwen3-8B", accuracy=0.8)
        try:
            code, err = run_cli(["--dataset", str(ROOT / "data" / "smoke.jsonl"), "--limit", "5",
                                 "--small-url", small.url, "--large-url", large.url, "--samples", "3",
                                 "--think-samples", "2", "--think-budget", "64", "--output", str(self.out)])
            assert code == 0, err
        finally:
            small.close()
            large.close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_bundle_files_and_labels(self):
        text = self.out.with_suffix(".report.md").read_text()
        for heading in ("## Evidence", "## Arms", "## Against the large model", "## What the checker adds",
                        "## Why answers were wrong"):
            self.assertIn(heading, text)
        failures = [json.loads(l) for l in self.out.with_suffix(".failures.jsonl").read_text().splitlines()]
        self.assertTrue(failures)
        self.assertTrue(all(f["label"] in report.LABELS for f in failures))
        # The fake's wrong answers always contain an arithmetic slip.
        self.assertIn("arithmetic_error", {f["label"] for f in failures})

    def test_interrupted_run_still_reports(self):
        self.out.with_suffix(".summary.json").unlink()
        bundle = report.write_bundle(self.out, ROOT / "data" / "smoke.jsonl")
        text = bundle["report"].read_text()
        self.assertIn("NOT RUN (pre-v0.9 output)", text)
        self.assertIn("## Arms", text)

    def test_label_rules(self):
        self.assertEqual(report.label_failure("a", [{"answer": None, "error": "No JSON"}], None, None)[0],
                         "unparseable_output")
        self.assertEqual(report.label_failure("a", [{"answer": None, "thinking_chars": 500}], None, None)[0],
                         "no_answer_after_thinking")
        self.assertEqual(report.label_failure("a", [{"answer": "3"}], None, None)[0],
                         "abstained_checker_filtered_all")
        self.assertEqual(report.label_failure("a", [{"answer": "35", "raw_content": "12 + 3 = 15. 50 - 15 = 35."}],
                                              "35", STORE)[0], "wrong_operation_or_interpretation")
        self.assertEqual(report.label_failure("a", [{"answer": "34", "raw_content": "12 * 3 = 36. 70 - 36 = 34."}],
                                              "34", STORE)[0], "ungrounded_numbers")
        self.assertEqual(report.label_failure("a", [{"answer": "9"}], "9", STORE)[0], "wrong_answer_unexplained")


if __name__ == "__main__":
    unittest.main()
