"""Small local HTTP load test. It does NOT establish frontier-model quality."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import statistics
import time
from .client import CPUReasoningClient


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8070")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--key", default="")
    args = p.parse_args()
    if not (1 <= args.n <= 10000 and 1 <= args.concurrency <= 128):
        p.error("n must be 1..10000 and concurrency must be 1..128")
    client = CPUReasoningClient(args.url, args.key)
    def one(i):
        begin = time.perf_counter()
        result = client.reason(f"calculate: ({i%13}+5)*3")
        return (time.perf_counter()-begin)*1000, result.get("status") == "ok", result.get("http_status")
    began = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        samples = [f.result() for f in as_completed([pool.submit(one, i) for i in range(args.n)])]
    duration = time.perf_counter()-began
    latencies = sorted(r[0] for r in samples)
    print(json.dumps({
        "requests": args.n, "concurrency": args.concurrency,
        "success": sum(ok for _,ok,_ in samples), "failures": sum(not ok for _,ok,_ in samples),
        "throughput_rps": round(args.n/duration, 2),
        "latency_p50_ms": round(latencies[int((len(latencies)-1)*0.5)],2),
        "latency_p95_ms": round(latencies[int((len(latencies)-1)*0.95)],2),
        "note": "Arithmetic-only API stress test, no neural model inference",
    }, indent=2))


if __name__ == "__main__":
    main()
