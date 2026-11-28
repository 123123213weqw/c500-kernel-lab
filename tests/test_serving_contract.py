import importlib.util
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "experimental/vllm_metax_serving/summarize.py"
MODULE = importlib.util.spec_from_file_location("serving_summarize", SOURCE)
s = importlib.util.module_from_spec(MODULE)
MODULE.loader.exec_module(s)

def fake():
    return {"summary":{"max_tokens":160,"concurrency":16,"fixed":True,"total_output_tokens":160,
                       "wall_s":2.,"mean_ttft_ms":1.,"mean_tpot_ms":1.},
            "rows":[{"id":"x","prompt_tokens":512,"completion_tokens":160,"output_ids":[0]*160}]}

class ServingContract(unittest.TestCase):
    def test_recompute_token_rate(self):
        self.assertEqual(s.checked_sample(fake(),{"x":512})["output_tok_s"],80)

    def test_reject_variable_eos(self):
        x = fake()
        x["summary"]["fixed"] = False
        with self.assertRaises(ValueError):
            s.checked_sample(x,{"x":512})

    def test_reject_missing_ids_under_claimed_count(self):
        x = fake()
        x["rows"][0]["output_ids"].pop()
        with self.assertRaises(ValueError):
            s.checked_sample(x,{"x":512})

    def test_reject_different_prompt_count(self):
        with self.assertRaises(ValueError):
            s.checked_sample(fake(),{"x":1024})

    def test_reject_nonfinite_time(self):
        x = fake()
        x["summary"]["wall_s"] = float("nan")
        with self.assertRaises(ValueError):
            s.checked_sample(x,{"x":512})

if __name__ == "__main__":
    unittest.main()
