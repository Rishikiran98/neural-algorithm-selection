"""Submit and poll a local CPU job with an idempotency key."""
import os
from cpu_platform.client import CPUReasoningClient


def main():
    client = CPUReasoningClient(api_key=os.environ.get("CPU_REASONING_API_KEY", ""))
    job = client.submit_job("calculate: (12+8)/5", idempotency_key="example-local-job")
    print("Submitted:", job)
    if "id" in job:
        print("Completed:", client.wait_for_job(job["id"], timeout_seconds=20))


if __name__ == "__main__":
    main()
