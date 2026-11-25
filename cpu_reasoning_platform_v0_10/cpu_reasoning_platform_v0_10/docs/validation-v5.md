# v0.5 offline validation record

Date: November 2, 2026.

- Python standard-library test discovery: **112 tests passed** on Python 3.13.5 (Linux).
- `python -m compileall` succeeded for source, tests and examples.
- Browser JavaScript passed `node --check` (Node 22).
- Standalone live HTTP process started on a loopback port and completed synchronous `2+3` with answer `5`.
- Real HTTP async submission completed `calculate: 12*(7+3)` with answer `120` and status `succeeded`.
- 16 concurrently submitted queued exact-arithmetic smoke cases completed successfully with 16/16 arithmetic answers correct. This is a test of arithmetic operations and queue service only, with **zero actual language model calls**.

These checks were **not** run on the user's Windows laptop or a leased CPU-only server. There is no measured real-LLM reasoning performance, pricing data, energy usage, or hardware attestation. The sample wall-clock results contain polling overhead and should not be compared to accelerator inference throughput.
