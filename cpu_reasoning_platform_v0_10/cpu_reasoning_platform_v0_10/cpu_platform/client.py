"""Minimal standard-library Python SDK for the local research API."""
import json
from urllib.request import Request, urlopen
from urllib.error import HTTPError


class CPUReasoningClient:
    def __init__(self, url="http://127.0.0.1:8070", api_key="", timeout=180):
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("Preview SDK only supports local HTTP origins")
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def verify_plan(self, question, plan, semantic=False):
        """Verify a user-supplied step graph locally, without a model call."""
        payload = json.dumps({"question": question, "plan": plan}).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        endpoint = "/v1/verify-semantic-plan" if semantic else "/v1/verify-plan"
        req = Request(self.url + endpoint, data=payload, headers=headers, method="POST")
        try:
            with urlopen(req, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            try:
                detail = json.loads(exc.read())
            except Exception:
                detail = {"error": str(exc)}
            return {"http_status": exc.code, **detail}

    def reason(self, question, mode="auto", max_calls=2):
        payload = json.dumps({"question": question, "mode": mode, "max_calls": max_calls}).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        req = Request(self.url + "/v1/reason", data=payload, headers=headers, method="POST")
        try:
            with urlopen(req, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            # Surface a structured response for invalid input, no model, etc.
            try:
                detail = json.loads(exc.read())
            except Exception:
                detail = {"error": str(exc)}
            return {"http_status": exc.code, **detail}

    def ask(self, question, *, domain="auto", strategy="direct", max_calls=2,
            thinking=False, think_budget=128):
        """General local Q&A. Model claims are NOT independently verified."""
        return self._post_json("/v1/ask", {
            "question": question, "domain": domain, "strategy": strategy,
            "max_calls": max_calls, "thinking": thinking, "think_budget": think_budget,
        })

    def verify_logic(self, premises, conclusion):
        """Bounded propositional entailment. No model calls, no natural-language semantics."""
        return self._post_json("/v1/verify-logic", {
            "premises": premises, "conclusion": conclusion,
        })

    def _post_json(self, path, payload):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        req = Request(self.url + path, data=json.dumps(payload).encode(),
                      headers=headers, method="POST")
        try:
            with urlopen(req, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            try:
                detail = json.loads(exc.read())
            except (ValueError, OSError):
                detail = {"error": str(exc)}
            return {"http_status": exc.code, **detail}

    def _job_request(self, method, path, payload=None, idempotency_key=None):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        if idempotency_key is not None:
            headers["Idempotency-Key"] = idempotency_key
        req = Request(self.url + path,
                      data=json.dumps(payload).encode() if payload is not None else None,
                      headers=headers, method=method)
        try:
            with urlopen(req, timeout=self.timeout) as response:
                return {"http_status": response.status, **json.load(response)}
        except HTTPError as exc:
            try:
                detail = json.loads(exc.read())
            except (ValueError, OSError):
                detail = {"error": str(exc)}
            return {"http_status": exc.code, **detail}

    def submit_job(self, question, mode="auto", max_calls=2, *, idempotency_key=None):
        """Enqueue a durable job; requires server --jobs-db and stores text on disk."""
        return self._job_request("POST", "/v1/jobs",
                                 {"question": question, "mode": mode, "max_calls": max_calls},
                                 idempotency_key=idempotency_key)

    def get_job(self, job_id):
        return self._job_request("GET", "/v1/jobs/" + job_id)

    def cancel_job(self, job_id):
        return self._job_request("DELETE", "/v1/jobs/" + job_id)

    def wait_for_job(self, job_id, *, timeout_seconds=120, poll_seconds=0.25):
        """Poll with a finite wall-clock timeout; returns the last seen state."""
        import time
        if timeout_seconds <= 0 or poll_seconds <= 0:
            raise ValueError("Timeout and polling interval must be positive")
        deadline = time.monotonic() + timeout_seconds
        while True:
            response = self.get_job(job_id)
            if response.get("status") in ("succeeded", "rejected", "failed", "interrupted", "cancelled"):
                return response
            if response.get("http_status") != 200:
                return response
            if time.monotonic() >= deadline:
                return {"http_status": 408, "error": "client_poll_timeout", "last_status": response.get("status")}
            time.sleep(min(poll_seconds, max(0, deadline - time.monotonic())))
