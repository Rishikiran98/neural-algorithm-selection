from tempfile import TemporaryDirectory
from pathlib import Path
from threading import Thread
import unittest
from cpu_platform.client import CPUReasoningClient
from cpu_platform.engine import ReasoningEngine
from cpu_platform.experiment5 import percentile, generate_case, run_experiment
from cpu_platform.server import create_server


class NoModel:
    def ready(self):
        return False


class TestExperiment5(unittest.TestCase):
    def test_percentile(self):
        self.assertIsNone(percentile([], .95))
        self.assertEqual(percentile([10, 20, 30, 40], .95), 30)

    def test_arithmetic_cases(self):
        self.assertEqual(generate_case(0,"arithmetic"), ("calculate: (11+9)*3", "60"))
        self.assertEqual(generate_case(1,"arithmetic")[1], "63")

    def test_wordproblem_expected(self):
        question, correct = generate_case(0, "wordproblems")
        self.assertIn("has 3 apples", question)
        self.assertEqual(correct, "4")

    def test_small_workload(self):
        with TemporaryDirectory() as tmp:
            server = create_server(0, engine=ReasoningEngine(NoModel()),
                                   jobs_db=str(Path(tmp)/"jobs.sqlite"))
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                client = CPUReasoningClient(f"http://127.0.0.1:{server.server_port}")
                report = run_experiment(client, n=4, concurrency=2)
                self.assertEqual(report["summary"]["success"], 4)
                self.assertEqual(report["summary"]["accuracy_on_scored_cases"], 1.0)
                self.assertEqual(report["summary"]["scored_cases"], 4)
                self.assertEqual(len(report["rows"]), 4)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == "__main__":
    unittest.main()
