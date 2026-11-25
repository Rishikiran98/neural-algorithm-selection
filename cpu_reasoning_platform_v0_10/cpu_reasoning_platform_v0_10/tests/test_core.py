import unittest
from fractions import Fraction

from cpu_first.core import (
    InvalidExpression, InvalidModelOutput, exact_arithmetic,
    gsm8k_gold, json_object, parse_number, formatted_fraction,
)
from cpu_first.benchmark import summarize


class TestCPUFirst(unittest.TestCase):
    def test_arithmetic(self):
        self.assertEqual(exact_arithmetic("(12 + 8) / 5 * 2"), Fraction(8, 1))
        self.assertEqual(exact_arithmetic("0.1+0.2"), Fraction(3, 10))
        self.assertEqual(exact_arithmetic("-3 + (+8)"), Fraction(5, 1))

    def test_forbidden(self):
        invalid = ["__import__('os').system('id')", "2**100", "abs(-5)",
                   "1//2", "1/0", "True", "[1,2]", "1e999", "2%3", "10^4"]
        for expression in invalid:
            with self.subTest(expression=expression):
                with self.assertRaises(InvalidExpression):
                    exact_arithmetic(expression)

    def test_json_output(self):
        self.assertEqual(json_object('extra {"expression":"2*(3+4)"} end')["expression"], "2*(3+4)")
        with self.assertRaises(InvalidModelOutput):
            json_object("no object")

    def test_gold_and_numbers(self):
        self.assertEqual(gsm8k_gold("reasoning #### 1,250"), Fraction(1250, 1))
        self.assertEqual(parse_number("-0.25"), Fraction(-1, 4))
        self.assertEqual(formatted_fraction(Fraction(1, 3)), "1/3")

    def test_summary(self):
        rows = [
            {"id":"a","mode":"baseline","correct":True,"output_valid":True,"latency_s":1,"completion_tokens":4,"prompt_tokens":5},
            {"id":"a","mode":"hybrid","correct":True,"output_valid":True,"latency_s":2,"completion_tokens":6,"prompt_tokens":5},
            {"id":"b","mode":"baseline","correct":False,"output_valid":True,"latency_s":1,"completion_tokens":4,"prompt_tokens":5},
            {"id":"b","mode":"hybrid","correct":True,"output_valid":True,"latency_s":2,"completion_tokens":6,"prompt_tokens":5},
        ]
        out = summarize(rows)
        self.assertEqual(out["hybrid_minus_baseline_accuracy"], 0.5)
        self.assertEqual(out["modes"]["hybrid"]["accuracy"], 1)


if __name__ == "__main__":
    unittest.main()
