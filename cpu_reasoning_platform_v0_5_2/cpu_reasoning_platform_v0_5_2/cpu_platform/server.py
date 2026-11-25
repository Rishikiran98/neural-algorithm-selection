"""Local CPU reasoning API, with opt-in durable single-node jobs.

Not an internet-facing or production-hardened multi-tenant service.
"""
from __future__ import annotations

import argparse
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import os
import re
from secrets import token_hex
from threading import BoundedSemaphore
from time import perf_counter
from typing import Any

from .engine import LazyLocalModel, ReasoningEngine, UpstreamUnavailable
from .proof import verify_plan
from .semantic import verify_semantic_plan
from .metrics import Metrics
from .jobs import JobStore, JobWorkers, JobFault
from . import __version__

MAX_BODY_BYTES = 16 * 1024


class App(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], engine: ReasoningEngine, *,
                 api_key: str = "", max_inflight: int = 2,
                 api_keys: dict[str, str] | None = None,
                 jobs_db: str | None = None, job_workers: int = 1,
                 max_queued: int = 32, max_per_tenant: int = 8,
                 job_ttl_hours: int = 24, max_submissions_per_hour: int = 120):
        host = address[0]
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("Preview API binds to loopback only; use a TLS reverse proxy for deployment")
        super().__init__(address, Handler)
        self.engine = engine
        if api_key and api_keys:
            raise ValueError("Use one API key mechanism, not both")
        if api_keys is not None and (not isinstance(api_keys, dict) or
            not api_keys or any(not isinstance(k, str) or not k or
            not isinstance(v, str) or not v for k, v in api_keys.items())):
            raise ValueError("api_keys must contain nonempty tenant/key strings")
        if api_keys and len(set(api_keys.values())) != len(api_keys):
            raise ValueError("API keys must be unique per tenant")
        self.api_key = api_key
        self.api_keys = api_keys or ({"default": api_key} if api_key else {})
        self.slots = BoundedSemaphore(max_inflight)
        self.metrics = Metrics()
        self.jobs = JobStore(jobs_db, max_active=max_queued, max_per_tenant=max_per_tenant,
                             ttl_hours=job_ttl_hours,
                             max_submissions_per_hour=max_submissions_per_hour) if jobs_db is not None else None
        # v0.5.1: queued jobs and synchronous requests share ONE execution budget
        # (max_inflight), so they cannot jointly overload the single local model.
        self.workers = (JobWorkers(self.jobs, engine, worker_count=job_workers, model_slots=self.slots)
                        if self.jobs else None)

    def server_close(self):
        if self.workers:
            self.workers.close()
            # An in-flight model call can still be active. Do not close its DB handle.
            if all(not t.is_alive() for t in self.workers.threads):
                self.jobs.close()
        super().server_close()


class Handler(BaseHTTPRequestHandler):
    server: App
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        # Do not log request bodies, headers, or question texts.
        pass

    def _send(self, code: int, data: dict):
        blob = json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(blob)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.close_connection = True
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(blob)

    def _tenant(self):
        if not self.server.api_keys:
            return "local"  # loopback development only
        supplied = self.headers.get("Authorization", "")
        match = None
        for tenant, secret in self.server.api_keys.items():
            if hmac.compare_digest(supplied, "Bearer " + secret):
                match = tenant
        return match

    def _auth(self):
        if self._tenant() is None:
            self._send(401, {"error": "unauthorized"})
            return False
        return True

    def _json_body(self):
        if self.headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json":
            raise JobFault("content_type_must_be_json", 415)
        n_str = self.headers.get("Content-Length")
        if not n_str or not n_str.isdigit():
            raise JobFault("content_length_required", 411)
        n = int(n_str)
        if n > MAX_BODY_BYTES:
            self.close_connection = True
            raise JobFault("body_too_large", 413)
        if n <= 0:
            raise JobFault("empty_body", 400)
        try:
            body = json.loads(self.rfile.read(n))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise JobFault("invalid_json", 400) from None
        if not isinstance(body, dict):
            raise JobFault("invalid_request_fields", 400)
        return body

    def _validate_job_request(self, body: dict):
        if set(body) - {"question", "mode", "max_calls"}:
            raise JobFault("invalid_request_fields", 400)
        question = body.get("question")
        mode = body.get("mode", "auto")
        calls = body.get("max_calls", None)
        settings = self.server.engine.settings
        if not isinstance(question, str) or not question.strip() or len(question) > settings.max_chars:
            raise JobFault("invalid_question", 400)
        if not isinstance(mode, str) or mode not in ("auto", "baseline", "hybrid", "adaptive", "proof", "semantic"):
            raise JobFault("invalid_mode", 400)
        if calls is not None and (type(calls) is not int or not 1 <= calls <= settings.max_model_calls):
            raise JobFault("invalid_max_calls", 400)
        return {"question": question, "mode": mode, "max_calls": calls}

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html_path = Path(__file__).parent / "static" / "index.html"
            blob = html_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(blob)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; form-action 'none'")
            self.send_header("Connection", "close")
            self.close_connection = True
            self.end_headers()
            self.wfile.write(blob)
            return
        if self.path == "/healthz":
            return self._send(200, {"status": "ok", "version": __version__, "scope": "local-preview"})
        if not self._auth():
            return
        if self.path == "/readyz":
            ready = bool(self.server.engine.model.ready()) if hasattr(self.server.engine.model, "ready") else True
            return self._send(200 if ready else 503,
                              {"model_ready": ready, "async_jobs_enabled": self.server.jobs is not None})
        if self.path == "/v1/metrics":
            data = self.server.metrics.snapshot()
            if self.server.jobs:
                data["jobs"] = self.server.jobs.snapshot()
            return self._send(200, data)
        if self.path.startswith("/v1/jobs/"):
            job_id = self.path[len("/v1/jobs/"):]
            if not re.fullmatch(r"[0-9a-f]{32}", job_id):
                return self._send(404, {"error": "job_not_found"})
            if self.server.jobs is None:
                return self._send(503, {"error": "jobs_not_configured"})
            job = self.server.jobs.get(self._tenant(), job_id)
            return self._send(200, job) if job else self._send(404, {"error": "job_not_found"})
        return self._send(404, {"error": "not_found"})

    def do_DELETE(self):
        if not self._auth():
            return
        if not self.path.startswith("/v1/jobs/"):
            return self._send(404, {"error": "not_found"})
        job_id = self.path[len("/v1/jobs/"):]
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            return self._send(404, {"error": "job_not_found"})
        if not self.server.jobs:
            return self._send(503, {"error": "jobs_not_configured"})
        try:
            return self._send(200, self.server.jobs.cancel(self._tenant(), job_id))
        except JobFault as exc:
            return self._send(exc.http_status, {"error": exc.code})

    def do_POST(self):
        if not self._auth():
            return
        if self.path not in ("/v1/reason", "/v1/verify-plan", "/v1/verify-semantic-plan", "/v1/jobs"):
            return self._send(404, {"error": "not_found"})
        try:
            body = self._json_body()
            if self.path == "/v1/jobs":
                if not self.server.jobs:
                    raise JobFault("jobs_not_configured", 503)
                request = self._validate_job_request(body)
                key = self.headers.get("Idempotency-Key")
                if key is not None and not re.fullmatch(r"[A-Za-z0-9._:-]{1,96}", key):
                    raise JobFault("invalid_idempotency_key", 400)
                job, created = self.server.jobs.submit(self._tenant(), request, key)
                return self._send(202 if created else 200, job)
            if self.path in ("/v1/verify-plan", "/v1/verify-semantic-plan"):
                if set(body) != {"question", "plan"}:
                    raise JobFault("invalid_request_fields", 400)
            elif set(body) - {"question", "mode", "max_calls"}:
                raise JobFault("invalid_request_fields", 400)
        except JobFault as exc:
            return self._send(exc.http_status, {"error": exc.code})
        if not self.server.slots.acquire(blocking=False):
            return self._send(429, {"error": "capacity_exceeded", "retryable": True})
        self.server.metrics.begin()
        status, route, duration, calls = "error", "unknown", 0, 0
        started = perf_counter()
        try:
            try:
                if self.path in ("/v1/verify-plan", "/v1/verify-semantic-plan"):
                    checked = (verify_semantic_plan(body.get("question"), body.get("plan"))
                               if self.path == "/v1/verify-semantic-plan"
                               else verify_plan(body.get("question"), body.get("plan")))
                    semantic_route = self.path == "/v1/verify-semantic-plan"
                    accepted = checked["valid"] and (not semantic_route or checked["semantically_verified"])
                    result = {
                        **checked, "answer": checked["answer"] if accepted else None,
                        "status": "ok" if accepted else "invalid",
                        "route": "offline_semantic_verification" if semantic_route else "offline_proof_verification", "expression": None,
                        "stop_reason": "checked_plan" if accepted else "rejected_plan",
                        "proof_verified": checked["valid"], "proof_trace": checked["steps"],
                        "attempts": [],
                        "error": checked.get("semantic_error") if semantic_route and not accepted else checked.get("error"),
                        "usage": {"model_calls": 0, "prompt_tokens": 0,
                                  "completion_tokens": 0, "total_model_latency_ms": 0,
                                  "total_latency_ms": round((perf_counter() - started) * 1000, 2)},
                    }
                else:
                    result = self.server.engine.reason(
                        body.get("question"), body.get("mode", "auto"), body.get("max_calls"),
                    )
            except ValueError as exc:
                status = "bad_request"
                return self._send(400, {"error": "invalid_request", "detail": str(exc)})
            except UpstreamUnavailable:
                status = "upstream_unavailable"
                return self._send(503, {"error": "local_model_unavailable", "retryable": True})
            result["request_id"] = token_hex(8)
            result["version"] = __version__
            status, route = result["status"], result["route"] or "unknown"
            duration = result["usage"]["total_latency_ms"]
            calls = result["usage"]["model_calls"]
            return self._send(200 if status == "ok" else 422, result)
        finally:
            self.server.metrics.end(status, route, duration, calls)
            self.server.slots.release()


def create_server(port: int = 8070, *, engine: ReasoningEngine | None = None,
                  api_key: str = "", max_inflight: int = 2, model_url: str = "http://127.0.0.1:8080",
                  api_keys: dict[str, str] | None = None,
                  jobs_db: str | None = None, job_workers: int = 1,
                  max_queued: int = 32, max_per_tenant: int = 8,
                  job_ttl_hours: int = 24, max_submissions_per_hour: int = 120) -> App:
    if max_inflight < 1 or max_inflight > 128:
        raise ValueError("max_inflight must be between 1 and 128")
    engine = engine or ReasoningEngine(LazyLocalModel(model_url))
    return App(("127.0.0.1", port), engine, api_key=api_key, max_inflight=max_inflight,
               api_keys=api_keys, jobs_db=jobs_db, job_workers=job_workers,
               max_queued=max_queued, max_per_tenant=max_per_tenant, job_ttl_hours=job_ttl_hours,
               max_submissions_per_hour=max_submissions_per_hour)


def main():
    parser = argparse.ArgumentParser(description="Local CPU-first reasoning API preview")
    parser.add_argument("--port", type=int, default=8070)
    parser.add_argument("--model-url", default="http://127.0.0.1:8080")
    parser.add_argument("--max-inflight", type=int, default=2)
    parser.add_argument("--jobs-db", default=None, help="Opt-in durable SQLite queue; stores prompts on disk")
    parser.add_argument("--job-workers", type=int, default=1)
    parser.add_argument("--max-queued", type=int, default=32)
    parser.add_argument("--max-per-tenant", type=int, default=8)
    parser.add_argument("--job-ttl-hours", type=int, default=24)
    parser.add_argument("--max-submissions-per-hour", type=int, default=120)
    args = parser.parse_args()
    keys_raw = os.environ.get("CPU_REASONING_API_KEYS_JSON", "")
    api_keys = json.loads(keys_raw) if keys_raw else None
    server = create_server(args.port, model_url=args.model_url,
                           api_key=os.environ.get("CPU_REASONING_API_KEY", ""), api_keys=api_keys,
                           max_inflight=args.max_inflight, jobs_db=args.jobs_db,
                           job_workers=args.job_workers, max_queued=args.max_queued,
                           max_per_tenant=args.max_per_tenant, job_ttl_hours=args.job_ttl_hours,
                           max_submissions_per_hour=args.max_submissions_per_hour)
    print(f"CPU reasoning preview http://127.0.0.1:{server.server_port}")
    print("Use GET /healthz and POST /v1/reason; model must be started separately for word problems")
    if not server.api_keys:
        print("No API key configured. Loopback only. Do not publicly expose this service.")
    if server.jobs:
        print("Durable jobs enabled: prompts and results stored as plaintext on local disk.")
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
