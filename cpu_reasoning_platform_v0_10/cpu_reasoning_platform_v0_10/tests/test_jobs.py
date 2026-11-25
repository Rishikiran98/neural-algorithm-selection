"""Offline and real-HTTP tests for the opt-in persistent queue. No LLM involved."""
import http.client
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
import time
import unittest

from cpu_platform.client import CPUReasoningClient
from cpu_platform.engine import ReasoningEngine
from cpu_platform.jobs import JobFault, JobStore
from cpu_platform.server import create_server


class FakeModel:
    def ready(self):
        return False


class TestJobStore(unittest.TestCase):
    def setUp(self):
        self.dir = TemporaryDirectory()
        self.path = str(Path(self.dir.name) / "jobs.db")
        self.jobs = JobStore(self.path, max_active=3, max_per_tenant=2)

    def tearDown(self):
        self.jobs.close()
        self.dir.cleanup()

    def test_new_job_persistent_and_tenant_isolated(self):
        job, created = self.jobs.submit("alice", {"question": "calculate: 2+2"})
        self.assertTrue(created)
        self.assertEqual(job["status"], "queued")
        self.assertNotIn("question", job)
        self.assertEqual(self.jobs.get("alice", job["id"])["id"], job["id"])
        self.assertIsNone(self.jobs.get("bob", job["id"]))
        self.jobs.close()
        self.jobs = JobStore(self.path, max_active=3, max_per_tenant=2)
        self.assertEqual(self.jobs.get("alice", job["id"])["status"], "queued")

    def test_same_key_returns_same_job(self):
        req = {"question": "calculate: 2+2"}
        first, _ = self.jobs.submit("alice", req, "key.1")
        second, created = self.jobs.submit("alice", req, "key.1")
        self.assertFalse(created)
        self.assertEqual(first["id"], second["id"])
        third, _ = self.jobs.submit("bob", req, "key.1")
        self.assertNotEqual(first["id"], third["id"])

    def test_conflicting_key_fails(self):
        self.jobs.submit("alice", {"question": "1"}, "key1")
        with self.assertRaises(JobFault) as ctx:
            self.jobs.submit("alice", {"question": "2"}, "key1")
        self.assertEqual(ctx.exception.http_status, 409)

    def test_queue_bounded(self):
        self.jobs.submit("alice", {"question": "1"})
        self.jobs.submit("alice", {"question": "2"})
        with self.assertRaises(JobFault) as ctx:
            self.jobs.submit("alice", {"question": "3"})
        self.assertEqual(ctx.exception.code, "queue_full")
        self.jobs.submit("bob", {"question": "4"})
        with self.assertRaises(JobFault):
            self.jobs.submit("charlie", {"question": "5"})

    def test_exclusive_database_owner(self):
        with self.assertRaises(RuntimeError):
            JobStore(self.path)

    def test_claim_and_complete(self):
        job, _ = self.jobs.submit("alice", {"question": "calculate: 2+3"})
        claimed = self.jobs.claim()
        self.assertEqual(claimed[0], job["id"])
        self.assertEqual(self.jobs.get("alice", job["id"])["status"], "running")
        self.jobs.finish(job["id"], {"answer": "5"})
        result = self.jobs.get("alice", job["id"])
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["result"]["answer"], "5")
        self.assertIsNone(self.jobs.claim())

    def test_cancel_pending_and_running_conflict(self):
        queued, _ = self.jobs.submit("alice", {"question": "1"})
        self.assertEqual(self.jobs.cancel("alice", queued["id"])["status"], "cancelled")
        self.assertIsNone(self.jobs.claim())
        running, _ = self.jobs.submit("alice", {"question": "2"})
        self.jobs.claim()
        with self.assertRaises(JobFault) as ctx:
            self.jobs.cancel("alice", running["id"])
        self.assertEqual(ctx.exception.http_status, 409)
        with self.assertRaises(JobFault):
            self.jobs.cancel("other", running["id"])

    def test_restart_marks_running_interrupted_not_retried(self):
        job, _ = self.jobs.submit("alice", {"question": "calculate: 7*7"})
        self.jobs.claim()
        self.jobs.close()
        self.jobs = JobStore(self.path, max_active=3, max_per_tenant=2)
        self.assertEqual(self.jobs.get("alice", job["id"])["status"], "interrupted")
        self.assertEqual(self.jobs.get("alice", job["id"])["error"], "worker_interrupted")
        self.assertIsNone(self.jobs.claim())

    def test_expire_only_terminal_jobs(self):
        finished, _ = self.jobs.submit("alice", {"question": "1"})
        self.jobs.claim()
        self.jobs.finish(finished["id"], {"answer": "1"})
        active, _ = self.jobs.submit("alice", {"question": "2"})
        self.jobs.db.execute("UPDATE jobs SET updated_at=?", (time.time()-90000,))
        self.jobs.db.commit()
        self.jobs._prune()
        self.assertIsNone(self.jobs.get("alice", finished["id"]))
        self.assertIsNotNone(self.jobs.get("alice", active["id"]))


class TestJobHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = TemporaryDirectory()
        cls.server = create_server(0, engine=ReasoningEngine(FakeModel()),
                                   api_keys={"alice": "alice-secret", "bob": "bob-secret"},
                                   jobs_db=str(Path(cls.dir.name)/"queue.sqlite"),
                                   max_queued=16, max_per_tenant=8)
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_port
        cls.a = CPUReasoningClient(f"http://127.0.0.1:{cls.port}", "alice-secret")
        cls.b = CPUReasoningClient(f"http://127.0.0.1:{cls.port}", "bob-secret")

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)
        cls.dir.cleanup()

    def raw(self, method, path, data=None, *, key="alice-secret", headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        hdrs = {"Content-Type": "application/json", **(headers or {})}
        if key is not None:
            hdrs["Authorization"] = "Bearer " + key
        conn.request(method, path, body=json.dumps(data) if data is not None else None,
                     headers=hdrs)
        resp = conn.getresponse()
        code, body = resp.status, json.loads(resp.read())
        conn.close()
        return code, body

    def test_submit_poll_result_and_metrics(self):
        result = self.a.submit_job("calculate: (10+2)*3", idempotency_key="first")
        self.assertIn(result["http_status"], (200, 202))
        data = self.a.wait_for_job(result["id"], timeout_seconds=4, poll_seconds=.02)
        self.assertEqual(data["status"], "succeeded")
        self.assertEqual(data["result"]["answer"], "36")
        self.assertEqual(data["result"]["usage"]["model_calls"], 0)
        code, metrics = self.raw("GET", "/v1/metrics")
        self.assertEqual(code, 200)
        self.assertIn("jobs", metrics)

    def test_idempotency_and_cross_tenant_access(self):
        req = {"question": "calculate: 8*9", "mode": "auto", "max_calls": 2}
        code, a1 = self.raw("POST", "/v1/jobs", req, headers={"Idempotency-Key": "dup-key"})
        self.assertEqual(code, 202)
        code, a2 = self.raw("POST", "/v1/jobs", req, headers={"Idempotency-Key": "dup-key"})
        self.assertEqual(code, 200)
        self.assertEqual(a1["id"], a2["id"])
        self.assertEqual(self.b.get_job(a1["id"])["http_status"], 404)
        self.assertEqual(self.b.cancel_job(a1["id"])["http_status"], 404)
        code, conflict = self.raw("POST", "/v1/jobs", {**req, "question": "calculate: 1+2"},
                                  headers={"Idempotency-Key": "dup-key"})
        self.assertEqual((code, conflict["error"]), (409, "idempotency_conflict"))

    def test_authentication_and_invalid_payloads(self):
        self.assertEqual(self.raw("POST", "/v1/jobs", {}, key=None)[0], 401)
        self.assertEqual(self.raw("GET", "/v1/jobs/" + "a"*32, key="bad")[0], 401)
        for payload in ({}, {"question": ""}, {"question": "2+2", "mode": "nope"},
                        {"question": "2+2", "max_calls": True},
                        {"question": "2+2", "max_calls": 999},
                        {"question": "2+2", "extra": 1}):
            with self.subTest(payload=payload):
                self.assertEqual(self.raw("POST", "/v1/jobs", payload)[0], 400)
        self.assertEqual(self.raw("POST", "/v1/jobs", {"question":"hi"},
                                  headers={"Idempotency-Key": "spaces not allowed"})[0], 400)
        self.assertEqual(self.raw("GET", "/v1/jobs/bad-id")[0], 404)

    def test_rejection_without_model(self):
        submission = self.a.submit_job("hello", mode="semantic")
        status = self.a.wait_for_job(submission["id"], timeout_seconds=4, poll_seconds=.02)
        self.assertEqual(status["status"], "rejected")
        self.assertEqual(status["result"]["stop_reason"], "unsupported_template")

    def test_worker_upstream_unavailable(self):
        submission = self.a.submit_job("What is the meaning of life?", mode="baseline")
        status = self.a.wait_for_job(submission["id"], timeout_seconds=4, poll_seconds=.02)
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["error"], "local_model_unavailable")
        self.assertNotIn("meaning of life", json.dumps(status))


class BlockingEngine:
    def __init__(self):
        self.started = Event()
        self.release = Event()

    def reason(self, **kwargs):
        self.started.set()
        self.release.wait(timeout=2)
        return {"status":"ok", "answer":"1"}


class TestBusyQueue(unittest.TestCase):
    def test_backpressure_and_cancel_before_execution(self):
        with TemporaryDirectory() as d:
            from cpu_platform.jobs import JobWorkers
            engine = BlockingEngine()
            store = JobStore(str(Path(d)/"queue.db"), max_active=2, max_per_tenant=2)
            workers = JobWorkers(store, engine)
            try:
                first,_ = store.submit("alice", {"question":"1"})
                self.assertTrue(engine.started.wait(timeout=1))
                second,_ = store.submit("alice", {"question":"2"})
                with self.assertRaises(JobFault):
                    store.submit("alice", {"question":"3"})
                cancelled = store.cancel("alice", second["id"])
                self.assertEqual(cancelled["status"], "cancelled")
                third,_ = store.submit("alice", {"question":"3"})
                self.assertEqual(third["status"], "queued")
            finally:
                engine.release.set()
                workers.close()
                store.close()


if __name__ == "__main__":
    unittest.main()
