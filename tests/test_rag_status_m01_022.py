import unittest

from app.classify_engine import compute_rag_status, error_result


class RagStatusTests(unittest.TestCase):
    def test_compute_rag_status_ok_degraded_failed(self):
        self.assertEqual(compute_rag_status(failed=False, examples_count=3), "ok")
        self.assertEqual(compute_rag_status(failed=False, examples_count=0), "degraded")
        self.assertEqual(compute_rag_status(failed=True, examples_count=0), "failed")
        self.assertEqual(compute_rag_status(failed=True, examples_count=5), "failed")

    def test_error_result_incluye_ragStatus_cuando_se_pasa(self):
        res = error_result(
            "r1",
            {"kind": "purchase", "period": {"isClosed": False}},
            "Embedding error: boom",
            rag_status="failed",
        )
        self.assertEqual(res["json"]["ragStatus"], "failed")
        self.assertEqual(res["json"]["outcome"], "error")


if __name__ == "__main__":
    unittest.main()
