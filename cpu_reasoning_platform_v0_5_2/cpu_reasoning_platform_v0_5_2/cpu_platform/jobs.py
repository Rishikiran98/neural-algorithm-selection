"""Bounded persistent single-node CPU job execution.

SQLite is a local development queue, not a distributed task broker. Prompts are stored
in plaintext on the host disk when async jobs are explicitly enabled.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import os
from threading import Condition, RLock, Thread
import time
from uuid import uuid4

from .engine import UpstreamUnavailable


class JobFault(Exception):
    def __init__(self, code: str, http_status: int):
        super().__init__(code)
        self.code = code
        self.http_status = http_status


class JobStore:
    def __init__(self, path: str, *, max_active: int = 32, max_per_tenant: int = 8,
                 ttl_hours: int = 24, max_submissions_per_hour: int = 120):
        if not 1 <= max_active <= 10000 or not 1 <= max_per_tenant <= max_active:
            raise ValueError("Invalid queue limits")
        if not 1 <= ttl_hours <= 720:
            raise ValueError("ttl_hours must be 1..720")
        if not 1 <= max_submissions_per_hour <= 100000:
            raise ValueError("Invalid submissions-per-hour limit")
        self.max_submissions_per_hour = max_submissions_per_hour
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lockfile = open(str(self.path) + ".lock", "a+b")
        try:
            # A second server must not interrupt the first server's running jobs.
            if os.name == "nt":
                import msvcrt
                self.lockfile.seek(0)
                if not self.lockfile.read(1):
                    self.lockfile.write(b"1")
                    self.lockfile.flush()
                self.lockfile.seek(0)
                msvcrt.locking(self.lockfile.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lockfile.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, IOError) as exc:
            self.lockfile.close()
            raise RuntimeError("Job database is already open by another process") from exc
        self.max_active = max_active
        self.max_per_tenant = max_per_tenant
        self.ttl_seconds = ttl_hours * 3600
        self.lock = RLock()
        self.changed = Condition(self.lock)
        self.db = sqlite3.connect(str(self.path), check_same_thread=False, timeout=10)
        self.db.row_factory = sqlite3.Row
        with self.lock:
            self.db.execute("PRAGMA busy_timeout=10000")
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    tenant TEXT NOT NULL,
                    idempotency_key TEXT,
                    request_hash TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    output_json TEXT,
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    started_at REAL,
                    completed_at REAL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_tenant_idempotency
                    ON jobs(tenant, idempotency_key)
                    WHERE idempotency_key IS NOT NULL;
                CREATE INDEX IF NOT EXISTS idx_jobs_queue ON jobs(status, created_at);
                CREATE INDEX IF NOT EXISTS idx_jobs_tenant ON jobs(tenant, status);
                CREATE INDEX IF NOT EXISTS idx_jobs_submission_limit ON jobs(tenant, created_at);
            """)
            # Forward-compatible migration for a v0.5 pre-release schema.
            columns = {r["name"] for r in self.db.execute("PRAGMA table_info(jobs)")}
            for field in ("started_at", "completed_at"):
                if field not in columns:
                    self.db.execute(f"ALTER TABLE jobs ADD COLUMN {field} REAL")
            # An interrupted running model call has unknown completion. Never report success
            # and never automatically repeat a potentially expensive operation.
            self.db.execute("UPDATE jobs SET status='interrupted', error='worker_interrupted', "
                            "updated_at=?, completed_at=? WHERE status='running'", (time.time(), time.time()))
            self.db.commit()
            self._prune()

    def _prune(self):
        cutoff = time.time() - self.ttl_seconds
        self.db.execute("DELETE FROM jobs WHERE status NOT IN ('queued','running') "
                        "AND updated_at < ?", (cutoff,))
        self.db.commit()

    @staticmethod
    def _public(row):
        if row is None:
            return None
        obj = {"id": row["id"], "status": row["status"],
               "created_at": row["created_at"], "updated_at": row["updated_at"],
               "error": row["error"]}
        if row["started_at"] is not None:
            obj["started_at"] = row["started_at"]
            obj["queue_wait_ms"] = round((row["started_at"] - row["created_at"]) * 1000, 2)
        if row["completed_at"] is not None:
            obj["completed_at"] = row["completed_at"]
            if row["started_at"] is not None:
                obj["execution_ms"] = round((row["completed_at"] - row["started_at"]) * 1000, 2)
        if row["output_json"] is not None:
            obj["result"] = json.loads(row["output_json"])
        return obj

    def submit(self, tenant: str, request: dict, idempotency_key: str | None = None):
        raw = json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest = sha256(raw.encode("utf-8")).hexdigest()
        now = time.time()
        with self.changed:
            self._prune()
            # IMMEDIATE serializes admissions across processes sharing this SQLite file.
            self.db.execute("BEGIN IMMEDIATE")
            try:
                if idempotency_key:
                    existing = self.db.execute(
                        "SELECT * FROM jobs WHERE tenant=? AND idempotency_key=?",
                        (tenant, idempotency_key)).fetchone()
                    if existing:
                        if existing["request_hash"] != digest:
                            raise JobFault("idempotency_conflict", 409)
                        self.db.commit()
                        return self._public(existing), False
                active = self.db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
                tenant_active = self.db.execute(
                    "SELECT COUNT(*) FROM jobs WHERE tenant=? AND status IN ('queued','running')",
                    (tenant,)).fetchone()[0]
                if active >= self.max_active or tenant_active >= self.max_per_tenant:
                    raise JobFault("queue_full", 429)
                hourly = self.db.execute(
                    "SELECT COUNT(*) FROM jobs WHERE tenant=? AND created_at >= ?",
                    (tenant, now - 3600)).fetchone()[0]
                if hourly >= self.max_submissions_per_hour:
                    raise JobFault("hourly_submission_limit", 429)
                ident = uuid4().hex
                self.db.execute("INSERT INTO jobs (id,tenant,idempotency_key,request_hash,input_json,"
                                "status,created_at,updated_at) VALUES (?,?,?,?,?,'queued',?,?)",
                                (ident, tenant, idempotency_key, digest, raw, now, now))
                self.db.commit()
            except BaseException:
                self.db.rollback()
                raise
            self.changed.notify_all()
            return self._public(self.db.execute("SELECT * FROM jobs WHERE id=?", (ident,)).fetchone()), True

    def get(self, tenant: str, job_id: str):
        with self.lock:
            return self._public(self.db.execute(
                "SELECT * FROM jobs WHERE id=? AND tenant=?", (job_id, tenant)).fetchone())

    def cancel(self, tenant: str, job_id: str):
        with self.changed:
            existing = self.db.execute("SELECT * FROM jobs WHERE id=? AND tenant=?", (job_id, tenant)).fetchone()
            if not existing:
                raise JobFault("job_not_found", 404)
            if existing["status"] == "queued":
                now = time.time()
                self.db.execute("UPDATE jobs SET status='cancelled', updated_at=?, completed_at=? WHERE id=?",
                                (now, now, job_id))
                self.db.commit()
                self.changed.notify_all()
            elif existing["status"] == "running":
                raise JobFault("job_already_running", 409)
            return self.get(tenant, job_id)

    def claim(self):
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                row = self.db.execute("SELECT id, input_json FROM jobs WHERE status='queued' "
                                      "ORDER BY created_at ASC, rowid ASC LIMIT 1").fetchone()
                if row is None:
                    self.db.commit()
                    return None
                now = time.time()
                changed = self.db.execute("UPDATE jobs SET status='running', started_at=?, updated_at=? "
                                          "WHERE id=? AND status='queued'", (now, now, row["id"]))
                if changed.rowcount != 1:
                    raise RuntimeError("Lost atomic claim")
                self.db.commit()
                return row["id"], json.loads(row["input_json"])
            except BaseException:
                self.db.rollback()
                raise

    def finish(self, job_id: str, result=None, *, state: str = "succeeded", error=None):
        if state not in ("succeeded", "rejected", "failed", "interrupted"):
            raise ValueError("Invalid terminal state")
        with self.changed:
            now = time.time()
            self.db.execute("UPDATE jobs SET status=?, output_json=?, error=?, updated_at=?, completed_at=? "
                            "WHERE id=? AND status='running'",
                            (state, json.dumps(result, ensure_ascii=False) if result is not None else None,
                             error, now, now, job_id))
            self.db.commit()
            self.changed.notify_all()

    def snapshot(self):
        with self.lock:
            counts = {row["status"]: row["n"] for row in self.db.execute(
                "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status")}
            return {"counts": counts, "max_active": self.max_active,
                    "max_per_tenant": self.max_per_tenant, "retention_hours": self.ttl_seconds // 3600,
                    "max_submissions_per_hour": self.max_submissions_per_hour}

    def close(self):
        with self.lock:
            self.db.close()
            if not self.lockfile.closed:
                if os.name == "nt":
                    import msvcrt
                    self.lockfile.seek(0)
                    msvcrt.locking(self.lockfile.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.lockfile.fileno(), fcntl.LOCK_UN)
                self.lockfile.close()


class JobWorkers:
    """Worker threads that never die on a storage or engine error (v0.5.1).

    `model_slots` is an optional semaphore shared with the synchronous API, so the
    total number of concurrent reasoning executions stays within one budget.
    """
    FINISH_RETRY_DELAYS = (0.2, 0.5, 1.0)
    CLAIM_ERROR_BACKOFF = 0.5

    def __init__(self, store: JobStore, engine, *, worker_count: int = 1, model_slots=None):
        if worker_count < 1 or worker_count > 16:
            raise ValueError("worker_count must be 1..16")
        self.store = store
        self.engine = engine
        self.model_slots = model_slots
        self.stopping = False
        self.storage_errors = 0
        self.threads = [Thread(target=self._loop, name=f"cpu-job-{i}", daemon=True)
                        for i in range(worker_count)]
        for thread in self.threads:
            thread.start()

    def _sleep(self, seconds: float):
        with self.store.changed:
            if not self.stopping:
                self.store.changed.wait(timeout=seconds)

    def _acquire_slot(self) -> bool:
        if self.model_slots is None:
            return True
        while not self.stopping:
            if self.model_slots.acquire(timeout=0.25):
                return True
        return False

    def _release_slot(self):
        if self.model_slots is not None:
            self.model_slots.release()

    def _loop(self):
        while not self.stopping:
            # Take an execution slot BEFORE claiming, so a claimed job never sits
            # in 'running' while waiting for capacity.
            if not self._acquire_slot():
                return
            idle_delay = 0.0
            try:
                task = self.store.claim()
                if task is None:
                    idle_delay = 0.5
                else:
                    self._execute(*task)
            except Exception:
                # Transient DB failure (locked, I/O) or unexpected bug: keep the
                # worker alive instead of silently shrinking the pool.
                self.storage_errors += 1
                idle_delay = self.CLAIM_ERROR_BACKOFF
            finally:
                self._release_slot()
            if idle_delay:
                self._sleep(idle_delay)

    def _execute(self, job_id: str, request: dict):
        try:
            result = self.engine.reason(**request)
            state = "succeeded" if result.get("status") == "ok" else "rejected"
            self._finish(job_id, result, state=state)
        except UpstreamUnavailable:
            self._finish(job_id, state="failed", error="local_model_unavailable")
        except ValueError:
            self._finish(job_id, state="rejected", error="invalid_request")
        except Exception:
            # Do not save prompts, stack traces, or raw upstream errors to the DB.
            self._finish(job_id, state="failed", error="internal_worker_error")

    def _finish(self, job_id: str, result=None, *, state: str, error=None):
        """Persist a terminal state, retrying transient storage errors.

        If the full result cannot be written, fall back to a bare 'failed' record so
        the job stops counting against quotas. If even that fails, the job stays
        'running' until restart recovery marks it 'interrupted'; the worker survives.
        """
        attempts = [(result, state, error)] * (len(self.FINISH_RETRY_DELAYS) + 1)
        attempts.append((None, "failed", "result_storage_failed"))
        for index, (res, st, err) in enumerate(attempts):
            try:
                self.store.finish(job_id, res, state=st, error=err)
                return
            except Exception:
                self.storage_errors += 1
                if index < len(self.FINISH_RETRY_DELAYS):
                    self._sleep(self.FINISH_RETRY_DELAYS[index])

    def close(self):
        with self.store.changed:
            self.stopping = True
            self.store.changed.notify_all()
        for thread in self.threads:
            thread.join(timeout=0.5)
