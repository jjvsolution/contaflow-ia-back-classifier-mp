import io
import json
import logging
import unittest
from pathlib import Path

from app.logging_setup import configure_logging, log_event


class LoggingSetupTests(unittest.TestCase):
    def test_log_event_emite_json_con_requestId_y_latency(self):
        configure_logging()
        logger = logging.getLogger("test.m01_018")
        logger.handlers.clear()
        logger.propagate = False
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        log_event(
            logger,
            "classify_done",
            requestId="req-123",
            latencyMs=456,
            llmLatencyMs=400,
        )
        line = stream.getvalue().strip()
        payload = json.loads(line)
        self.assertEqual(payload["event"], "classify_done")
        self.assertEqual(payload["requestId"], "req-123")
        self.assertEqual(payload["latencyMs"], 456)
        self.assertEqual(payload["llmLatencyMs"], 400)

    def test_classify_engine_sin_print_de_debug(self):
        src = Path(__file__).resolve().parents[1] / "app" / "classify_engine.py"
        text = src.read_text(encoding="utf-8")
        self.assertNotIn("print(", text)
        self.assertIn("classify_done", text)
        self.assertIn("requestId", text)
        self.assertIn("latencyMs", text)


if __name__ == "__main__":
    unittest.main()
