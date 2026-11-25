"""Regression tests for the three v0.5.1 fixes. No LLM involved."""
import http.client
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from threading import Event, Thread
import time
import unittest

from cpu_platform.engine import ReasoningEngine
from cpu_platform.grounding import allowed_numbers, check_numeric_grounding
from cpu_platform.jobs import JobStore, JobWorkers
from cpu_platform.semantic import classify, verify_semantic_plan
from cpu_platform.server import create_server

ROOT = Path(__file__).resolve().parent.parent


def plan_from_template(question: str) -> dict:
    """Build the canonical correct plan for a supported template question."""
    spec = classify(question)
    steps, values = [], []
    for index, (op, operands) in enumerate(spec.expected_steps):
        inputs, nums = [], []
        for kind, value in operands:
            if kind == "literal":
                text = str(value.numerator) if value.denominator == 1 else str(float(value))
                inputs.append({"literal": text})
                nums.append(value)
            else:
                inputs.append({"ref": f"s{value + 1}"})
                nums.append(values[value])
        a, b = nums
        out = {"add": a + b, "subtract": a - b, "multiply": a * b, "divide": a / b}[op]
        values.append(out)
        claim = str(out.numerator) if out.denominator == 1 else f"{out.numerator}/{out.denominator}"
        steps.append({"id": f"s{index + 1}", "op": op, "inputs": inputs, "claim": claim})
    return {"steps": steps, "result": steps[-1]["id"]}


class TestSentenceFinalNumbers(unittest.TestCase):
    """Bug 1: numbers followed by sentence punctuation were silently ignored."""

    def test_sentence_final_numbers_are_grounded(self):
        cases = {
            "You pay with $50. How much change?": {"50"},
            "Tom had 12. He got 8 more.": {"12", "8"},
            "The price is 3.5. Then": {"7/2"},
            "Count: 7, 8; 9! 10? (11)": {"7", "8", "9", "10", "11"},
            "Total 1,000. Done": {"1000"},
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                self.assertEqual({str(x) for x in allowed_numbers(question)}, expected)

    def test_still_rejects_non_number_tokens(self):
        # Version strings, ordinals and identifiers must not become literals.
        self.assertEqual(allowed_numbers("Use v1.2.3 on the 5th day of x7."), set())
        self.assertEqual(allowed_numbers("Section 1.2.3 applies."), set())

    def test_adaptive_grounding_accepts_sentence_final_literal(self):
        q = "A store charges $12 per notebook. You buy 3 notebooks and pay with $50. How much change?"
        self.assertTrue(check_numeric_grounding(q, "50 - 12*3").passed)

    def test_every_smoke_template_question_verifies_end_to_end(self):
        # The v0.5 tests only checked classify(); this checks full verification,
        # which is what failed for the purchase_change template.
        rows = [json.loads(line) for line in (ROOT / "data" / "smoke.jsonl").read_text().splitlines() if line.strip()]
        supported = [r for r in rows if classify(r["question"]) is not None]
        self.assertGreaterEqual(len(supported), 4)
        for row in supported:
            with self.subTest(id=row["id"]):
                checked = verify_semantic_plan(row["question"], plan_from_template(row["question"]))
                self.assertTrue(checked["valid"], checked["error"])
                self.assertTrue(checked["semantically_verified"], checked["semantic_error"])
                gold = row["answer"].rsplit("####", 1)[1].strip()
                self.assertEqual(checked["answer"], gold)


class EchoEngine(ReasoningEngine):
    """Real engine for arithmetic; questions containing 'block' wait on an event."""

    def __init__(self):
        super().__init__(model=None)
        self.release = Event()
        self.entered = Event()

    def reason(self, question, mode="auto", max_calls=None):
        if "block" in question:
            self.entered.set()
            self.release.wait(10)
            question = "calculate: 1+1"
        return super().reason(question, mode, max_calls)


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class TestWorkerSurvivesStorageErrors(unittest.TestCase):
    """Bug 2: an exception in finish()/claim() killed the worker and stranded the job."""

    def setUp(self):
        self.dir = TemporaryDirectory()
        self.store = JobStore(str(Path(self.dir.name) / "jobs.db"))
        JobWorkers.FINISH_RETRY_DELAYS = (0.01, 0.01, 0.01)
        JobWorkers.CLAIM_ERROR_BACKOFF = 0.01

    def tearDown(self):
        JobWorkers.FINISH_RETRY_DELAYS = (0.2, 0.5, 1.0)
        JobWorkers.CLAIM_ERROR_BACKOFF = 0.5
        self.workers.close()
        self.store.close()
        self.dir.cleanup()

    def test_transient_finish_error_is_retried(self):
        original, failures = self.store.finish, {"left": 2}

        def flaky(*args, **kwargs):
            if failures["left"]:
                failures["left"] -= 1
                raise sqlite3.OperationalError("disk I/O error")
            return original(*args, **kwargs)
        self.store.finish = flaky
        self.workers = JobWorkers(self.store, EchoEngine(), worker_count=1)
        job, _ = self.store.submit("t", {"question": "calculate: 2+3", "mode": "auto", "max_calls": None})
        self.assertTrue(wait_for(lambda: self.store.get("t", job["id"])["status"] == "succeeded"))
        self.assertEqual(self.store.get("t", job["id"])["result"]["answer"], "5")
        self.assertTrue(all(t.is_alive() for t in self.workers.threads))

    def test_unwritable_result_falls_back_to_failed_and_frees_quota(self):
        original = self.store.finish

        def rejects_results(job_id, result=None, **kwargs):
            if result is not None:
                raise sqlite3.OperationalError("database or disk is full")
            return original(job_id, result, **kwargs)
        self.store.finish = rejects_results
        self.workers = JobWorkers(self.store, EchoEngine(), worker_count=1)
        job, _ = self.store.submit("t", {"question": "calculate: 2+3", "mode": "auto", "max_calls": None})
        self.assertTrue(wait_for(lambda: self.store.get("t", job["id"])["status"] == "failed"))
        self.assertEqual(self.store.get("t", job["id"])["error"], "result_storage_failed")
        self.assertNotIn("running", self.store.snapshot()["counts"])
        # The same worker keeps serving later jobs.
        self.store.finish = original
        later, _ = self.store.submit("t", {"question": "calculate: 4+4", "mode": "auto", "max_calls": None})
        self.assertTrue(wait_for(lambda: self.store.get("t", later["id"])["status"] == "succeeded"))

    def test_claim_error_does_not_kill_worker(self):
        original, failures = self.store.claim, {"left": 3}

        def flaky_claim():
            if failures["left"]:
                failures["left"] -= 1
                raise sqlite3.OperationalError("database is locked")
            return original()
        self.store.claim = flaky_claim
        self.workers = JobWorkers(self.store, EchoEngine(), worker_count=1)
        job, _ = self.store.submit("t", {"question": "calculate: 6*7", "mode": "auto", "max_calls": None})
        self.assertTrue(wait_for(lambda: self.store.get("t", job["id"])["status"] == "succeeded"))
        self.assertTrue(all(t.is_alive() for t in self.workers.threads))


class TestSharedExecutionBudget(unittest.TestCase):
    """Bug 3: sync requests and job workers had separate, additive concurrency limits."""

    def setUp(self):
        self.dir = TemporaryDirectory()
        self.engine = EchoEngine()
        self.server = create_server(0, engine=self.engine, max_inflight=1, job_workers=2,
                                    jobs_db=str(Path(self.dir.name) / "jobs.db"))
        Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.engine.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.dir.cleanup()

    def post(self, path, body):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=10)
        payload = json.dumps(body)
        conn.request("POST", path, payload, {"Content-Type": "application/json"})
        response = conn.getresponse()
        data = json.loads(response.read())
        conn.close()
        return response.status, data

    def test_running_job_consumes_the_shared_slot(self):
        status, job = self.post("/v1/jobs", {"question": "block please"})
        self.assertEqual(status, 202)
        self.assertTrue(self.engine.entered.wait(5))
        status, body = self.post("/v1/reason", {"question": "calculate: 2+2"})
        self.assertEqual(status, 429)
        self.assertEqual(body["error"], "capacity_exceeded")
        self.engine.release.set()
        self.assertTrue(wait_for(lambda: self.server.jobs.get("local", job["id"])["status"] == "succeeded"))
        self.assertTrue(wait_for(lambda: self.post("/v1/reason", {"question": "calculate: 2+2"})[0] == 200))

    def test_two_workers_never_exceed_one_slot(self):
        _, first = self.post("/v1/jobs", {"question": "block one"})
        _, second = self.post("/v1/jobs", {"question": "calculate: 9+9"})
        self.assertTrue(self.engine.entered.wait(5))
        time.sleep(0.4)
        # The second worker must wait for the slot, leaving its job queued (not running).
        self.assertEqual(self.server.jobs.get("local", second["id"])["status"], "queued")
        self.engine.release.set()
        self.assertTrue(wait_for(lambda: self.server.jobs.get("local", second["id"])["status"] == "succeeded"))


if __name__ == "__main__":
    unittest.main()
