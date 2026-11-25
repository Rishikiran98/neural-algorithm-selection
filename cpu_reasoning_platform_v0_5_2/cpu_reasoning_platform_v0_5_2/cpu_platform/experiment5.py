"""Single-node job-system workload experiment, not a frontier AI benchmark."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import statistics
import time

from .client import CPUReasoningClient


def percentile(values, pct):
    if not values:
        return None
    values = sorted(values)
    return round(values[min(len(values) - 1, int((len(values) - 1) * pct))], 2)


def generate_case(i, workload):
    n = i + 11
    if workload == "arithmetic":
        return f"calculate: ({n}+9)*3", str((n+9)*3)
    # This tests model-backed reasoning on a repeated template, not generalization.
    n = 3 * (i + 4) - 9
    return (f"Maya has {n} apples, buys 9 more, and puts them equally into 3 boxes.",
            str(i + 4))


def run_experiment(client, *, n=16, concurrency=4, mode="auto", workload="arithmetic", wait=120):
    if not (1 <= n <= 1000 and 1 <= concurrency <= 32 and 1 <= wait <= 600):
        raise ValueError("Out-of-range experiment configuration")
    if workload not in ("arithmetic", "wordproblems"):
        raise ValueError("Invalid workload")

    def one(i):
        q, expected = generate_case(i, workload)
        start = time.perf_counter()
        try:
            created = client.submit_job(q, mode=mode, idempotency_key=f"experiment5-{int(start*1000000000)}-{i}")
            if created.get("http_status") not in (200, 202):
                return {"case": i, "status": "submission_failed", "error": created.get("error"),
                        "http_status": created.get("http_status"), "wall_ms": round((time.perf_counter()-start)*1000,2)}
            finished = client.wait_for_job(created["id"], timeout_seconds=wait)
            result = finished.get("result") or {}
            observed = result.get("answer")
            return {"case": i, "status": finished.get("status", "poll_failed"),
                    "correct": observed == expected if expected is not None else None,
                    "expected": expected, "observed": observed, "job_id": finished.get("id"),
                    "queue_wait_ms": finished.get("queue_wait_ms"),
                    "execution_ms": finished.get("execution_ms"),
                    "model_calls": result.get("usage",{}).get("model_calls"),
                    "wall_ms": round((time.perf_counter()-start)*1000,2)}
        except Exception as exc:
            return {"case": i, "status": "client_error", "error_type": type(exc).__name__,
                    "wall_ms": round((time.perf_counter()-start)*1000,2)}

    began = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = [future.result() for future in as_completed(pool.submit(one, i) for i in range(n))]
    duration = time.perf_counter() - began
    rows.sort(key=lambda x: x["case"])
    terminal = sum(r["status"] in ("succeeded", "rejected", "failed", "cancelled", "interrupted") for r in rows)
    verified = [r for r in rows if r.get("correct") is not None]
    success = sum(r["status"] == "succeeded" for r in rows)
    report = {"configuration": {"n": n, "concurrency": concurrency, "mode": mode,
                                "workload": workload, "timeout_per_job_s": wait},
              "summary": {"success": success, "terminal": terminal,
                          "failures_or_timeouts": n - success,
                          "accuracy_on_scored_cases": (sum(r["correct"] for r in verified)/len(verified)
                                                       if verified else None),
                          "scored_cases": len(verified),
                          "attempted_jobs_per_s": round(n/duration,2),
                          "elapsed_s": round(duration,2),
                          "wall_p50_ms": percentile([r["wall_ms"] for r in rows], .50),
                          "wall_p95_ms": percentile([r["wall_ms"] for r in rows], .95),
                          "queue_wait_p95_ms": percentile([r["queue_wait_ms"] for r in rows
                                                             if r.get("queue_wait_ms") is not None], .95),
                          "execution_p95_ms": percentile([r["execution_ms"] for r in rows
                                                            if r.get("execution_ms") is not None], .95)},
              "disclaimer": "Arithmetic mode measures job-service overhead without AI. Repeated word-problem templates do not demonstrate generalized reasoning or GPU-free training.",
              "rows": rows}
    return report


def main():
    parser = argparse.ArgumentParser(description="CPU reasoning durable queue workload experiment")
    parser.add_argument("--url", default="http://127.0.0.1:8070")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--mode", choices=["auto","baseline","hybrid","adaptive","proof","semantic"], default="auto")
    parser.add_argument("--workload", choices=["arithmetic", "wordproblems"], default="arithmetic")
    parser.add_argument("--wait", type=int, default=120)
    parser.add_argument("--out", default="results/experiment5.json")
    args = parser.parse_args()
    report = run_experiment(CPUReasoningClient(args.url, args.api_key), n=args.n,
                            concurrency=args.concurrency, mode=args.mode,
                            workload=args.workload, wait=args.wait)
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    if report["summary"]["failures_or_timeouts"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
