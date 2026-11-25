"""v0.5.2 tests: hardware profile, launcher, model client, verified search,
matched-compute experiment. Offline; the fake server is not a model."""
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from cpu_first.core import LocalLlama, strip_thinking
from cpu_platform import experiment6, hardware, launch
from cpu_platform.search import Sample, prefix_votes, sample, vote
try:
    from fake_llama import FakeLlama          # python -m unittest discover -s tests
except ImportError:
    from tests.fake_llama import FakeLlama    # python -m unittest tests.test_v052

GB = hardware.GB
ROOT = Path(__file__).resolve().parent.parent


class TestHardware(unittest.TestCase):
    def test_physical_cores_ignore_hyperthreads(self):
        text = "\n\n".join(f"processor : {i}\nphysical id : 0\ncore id : {i % 6}" for i in range(12))
        self.assertEqual(hardware.parse_proc_cpuinfo_cores(text), 6)
        self.assertIsNone(hardware.parse_proc_cpuinfo_cores("processor : 0\n"))

    def test_profile_scales_with_ram(self):
        laptop32 = hardware.recommend(32 * GB, 6, 12, 20e9)
        self.assertEqual(laptop32["threads"], 6)
        self.assertEqual(laptop32["ctx_size"], 4096)
        self.assertEqual(laptop32["large_model"], "qwen3-8b")
        self.assertIn("qwen3-14b", laptop32["feasible_models"])
        self.assertEqual(hardware.recommend(8 * GB, 4, 8, None)["large_model"], "qwen3-4b")
        tiny = hardware.recommend(4 * GB, 2, 4, None)
        self.assertEqual(tiny["ctx_size"], 2048)
        self.assertEqual(tiny["feasible_models"], ["qwen3-0.6b"])
        self.assertIsNone(tiny["large_model"])

    def test_unknown_cores_fall_back_to_half_logical(self):
        self.assertEqual(hardware.recommend(16 * GB, None, 12, None)["threads"], 6)

    def test_token_estimate_is_bandwidth_over_size(self):
        est = hardware.recommend(32 * GB, 6, 12, 32e9)["token_rate_estimate_per_s"]
        self.assertGreater(est["qwen3-0.6b"], 6 * est["qwen3-8b"])

    def test_detect_runs_on_this_machine(self):
        info = hardware.detect(measure_bandwidth=False)
        self.assertIn("recommended", info)
        self.assertGreaterEqual(info["recommended"]["threads"], 1)


FULL_HELP = ("-t, --threads N\n-tb, --threads-batch N\n-np, --parallel N\n-dev, --device <dev1,dev2,..>\n"
             "--jinja\n-rea, --reasoning [on|off|auto]\n--reasoning-budget N\n")


class TestLauncher(unittest.TestCase):
    def build(self, help_text, **kw):
        args = dict(model_source=("hf", "Qwen/Qwen3-4B-GGUF:Q4_K_M"), port=8081, threads=6,
                    ctx_size=4096, allow_gpu=False, is_qwen3=True)
        args.update(kw)
        return launch.build_command(["llama-server"], help_text, **args)

    def test_full_featured_build(self):
        cmd, notes = self.build(FULL_HELP)
        joined = " ".join(cmd)
        for part in ("--n-gpu-layers 0", "--threads 6", "--threads-batch 6", "--device none",
                     "--host 127.0.0.1", "--port 8081", "--parallel 1", "--reasoning auto", "--jinja"):
            self.assertIn(part, joined)
        self.assertNotIn("--reasoning off", joined)     # v0.8: thinking is per request
        self.assertNotIn("--reasoning-budget", joined)
        self.assertEqual(notes, [])

    def test_older_build_falls_back(self):
        cmd, notes = self.build("-t N\n--reasoning-budget N\n")
        self.assertNotIn("--reasoning-budget", cmd)      # per-request budget instead (v0.8)
        self.assertNotIn("--device", cmd)
        self.assertTrue(any("--device" in n for n in notes))

    def test_reasoning_flag_not_confused_with_budget(self):
        self.assertFalse(launch.supports("--reasoning-budget N", "--reasoning"))
        self.assertTrue(launch.supports("-rea, --reasoning [on|off]", "--reasoning"))

    def test_allow_gpu_and_non_qwen(self):
        cmd, _ = self.build(FULL_HELP, allow_gpu=True, is_qwen3=False)
        self.assertNotIn("--device", cmd)
        self.assertNotIn("--reasoning", cmd)
        self.assertIn("0", cmd[cmd.index("--n-gpu-layers") + 1])

    def test_log_scan(self):
        used = ("ggml_cuda_init: found 1 CUDA devices (Total VRAM: 4096 MiB):\n"
                "llama_model_load_from_file_impl: using device CUDA0 (Quadro P600) - 3900 MiB free\n")
        scan = launch.scan_log_for_gpu(used)
        self.assertEqual(scan["verdict"], "gpu_visible_or_used")
        hidden = ("ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected\n"
                  "load_tensors: offloaded 0/29 layers to GPU\nload_tensors: CPU_Mapped model buffer size = 600 MiB\n")
        self.assertEqual(launch.scan_log_for_gpu(hidden)["verdict"], "gpu_backend_mentioned_not_used")
        self.assertEqual(launch.scan_log_for_gpu("load_tensors: CPU model buffer size = 1 MiB")["verdict"],
                         "no_gpu_backend_seen")

    def test_dry_run_and_ram_guard(self):
        with TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as out, \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(launch.main(["--model", "qwen3-0.6b", "--dry-run", "--results-dir", tmp]), 0)
        self.assertIn("--n-gpu-layers 0", out.getvalue())
        original = launch.detect
        launch.detect = lambda measure_bandwidth=True: {**original(False), "memory": {
            "total_bytes": 4 * GB, "available_bytes": None, "total_gb": 4.0}}
        try:
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()), \
                    contextlib.redirect_stdout(io.StringIO()):
                launch.main(["--model", "qwen3-8b", "--dry-run"])
        finally:
            launch.detect = original


class TestModelClient(unittest.TestCase):
    def test_strip_thinking(self):
        self.assertEqual(strip_thinking('<think>{"x":1}</think>{"answer":"4"}'), '{"answer":"4"}')
        self.assertEqual(strip_thinking("<think>still thinking when the budget ran out"), "")
        self.assertEqual(strip_thinking('{"answer":"4"}'), '{"answer":"4"}')

    def test_qwen3_gets_no_think_seed_and_temperature(self):
        fake = FakeLlama("Qwen3-0.6B-Q8_0.gguf", accuracy=1.0, think=True)
        try:
            client = LocalLlama(fake.url)
            out = client.chat("Solve.", "Maya has 12 apples", temperature=0.7, seed=42)
            self.assertNotIn("<think>", out.content)
            body = fake.requests[-1]
            self.assertTrue(body["messages"][0]["content"].endswith("/no_think"))
            self.assertEqual((body["temperature"], body["seed"]), (0.7, 42))
            with self.assertRaises(ValueError):
                client.chat("Solve.", "x", temperature=3)
        finally:
            fake.close()

    def test_other_models_unchanged(self):
        fake = FakeLlama("llama-3.2-1b", accuracy=1.0)
        try:
            LocalLlama(fake.url).chat("Solve.", "q")
            body = fake.requests[-1]
            self.assertEqual(body["messages"][0]["content"], "Solve.")
            self.assertNotIn("seed", body)
        finally:
            fake.close()


def S(i, answer, verified=True):
    return Sample(i, answer, verified, None, 0, 0, 0.1, None)


class TestSearch(unittest.TestCase):
    def test_vote_rules(self):
        samples = [S(0, "5", False), S(1, "4"), S(2, "6"), S(3, "6"), S(4, "4")]
        self.assertEqual(vote(samples, require_verified=False), "4")  # 4/6 tie, 4 seen first
        self.assertEqual(vote(samples, require_verified=True), "4")  # tie 4/6 -> first seen
        self.assertIsNone(vote([S(0, None), S(1, "3", False)], require_verified=True))
        self.assertEqual(prefix_votes(samples, require_verified=True), [None, "4", "4", "6", "4"])

    def test_sample_verifies_plans(self):
        fake = FakeLlama("small", accuracy=0.5)
        try:
            q = "A store charges $12 per notebook. You buy 3 notebooks and pay with $50. How much change?"
            drawn = sample(LocalLlama(fake.url), q, 12, kind="plan", seed=1)
            self.assertTrue(any(s.verified for s in drawn) and not all(s.verified for s in drawn))
            self.assertTrue(all(s.answer == "14" for s in drawn if s.verified))
            self.assertTrue(all(s.answer is None for s in drawn if not s.verified))
            seeds = [r["seed"] for r in fake.requests]
            self.assertEqual(seeds, list(range(1, 13)))
        finally:
            fake.close()


class TestExperiment6(unittest.TestCase):
    def test_end_to_end_cli(self):
        small = FakeLlama("Qwen3-0.6B", accuracy=0.55)
        large = FakeLlama("Qwen3-4B", accuracy=0.8, delay_s=0.03)
        try:
            with TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                out = Path(tmp) / "exp6.jsonl"
                code = experiment6.main(["--dataset", str(ROOT / "data" / "smoke.jsonl"), "--limit", "5",
                                         "--small-url", small.url, "--large-url", large.url,
                                         "--samples", "6", "--output", str(out),
                                         "--arms", "small_greedy,large_greedy,small_vote,small_verified"])
                self.assertEqual(code, 0)
                report = json.loads(out.with_suffix(".summary.json").read_text())
                rows = [json.loads(l) for l in out.read_text().splitlines()]
            res = report["results"]
            self.assertEqual(res["paired_questions"], 5)
            self.assertEqual(res["cost_unit"], "model_wall_seconds")
            self.assertEqual(len(res["curves"]["small_verified"]), 6)
            self.assertEqual(len(res["curves"]["large_greedy"]), 1)
            costs = [p["mean_cost"] for p in res["curves"]["small_vote"]]
            self.assertEqual(costs, sorted(costs))  # cumulative cost grows with k
            self.assertIn("small_verified_vs_large_greedy", res["comparisons"])
            self.assertIn("verifier_effect_small_verified_vs_small_vote", res["comparisons"])
            self.assertEqual({len(r["arms"]["small_verified"]) for r in rows}, {6})
            for r in rows:  # verified answers are always right; unverified plans never vote
                for s in r["arms"]["small_verified"]:
                    self.assertEqual(s["verified"], s["answer"] is not None)
                    if s["verified"]:
                        self.assertEqual(s["answer"], r["gold"])
            self.assertEqual(report["models"]["large"]["server_model_id"], "Qwen3-4B")
        finally:
            small.close()
            large.close()

    def test_small_only_run(self):
        small = FakeLlama("Qwen3-0.6B", accuracy=0.6)
        try:
            with TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                out = Path(tmp) / "e.jsonl"
                experiment6.main(["--limit", "3", "--small-url", small.url, "--large-url", "none",
                                  "--samples", "3", "--dataset", str(ROOT / "data" / "smoke.jsonl"),
                                  "--output", str(out)])
                res = json.loads(out.with_suffix(".summary.json").read_text())["results"]
            self.assertNotIn("large_greedy", res["curves"])
            self.assertNotIn("small_verified_vs_large_greedy", res["comparisons"])
        finally:
            small.close()

    def test_refuses_partial_records(self):
        rec = {"id": "1", "gold": "4", "arms": {"small_greedy": [S(0, "4", False).to_dict()]}}
        with self.assertRaises(ValueError):
            experiment6.summarize([rec], ("small_greedy", "small_vote"), "model_wall_seconds")

    def test_matched_point_selection(self):
        def rec(i, gold, large_ok, verified_answers):
            ls = Sample(0, gold if large_ok else "0", False, None, 0, 0, 1.0, None).to_dict()
            vs = [Sample(j, a, a is not None, None, 0, 0, 0.3, None).to_dict()
                  for j, a in enumerate(verified_answers)]
            return {"id": str(i), "gold": gold, "arms": {"large_greedy": [ls], "small_verified": vs}}
        records = [rec(0, "4", True, [None, "4", "4", "4"]), rec(1, "5", False, ["5", None, "5", "5"])]
        res = experiment6.summarize(records, ("large_greedy", "small_verified"), "model_wall_seconds")
        cmp = res["comparisons"]["small_verified_vs_large_greedy"]
        self.assertEqual(cmp["matched_k"], 3)  # 3 x 0.3 s <= 1.0 s < 4 x 0.3 s
        self.assertEqual(cmp["matched_accuracy"], 1.0)
        self.assertEqual(cmp["reference_accuracy"], 0.5)
        self.assertEqual(cmp["first_k_reaching_reference_accuracy"], 1)


if __name__ == "__main__":
    unittest.main()
