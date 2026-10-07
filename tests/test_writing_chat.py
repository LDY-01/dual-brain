from datetime import datetime, timedelta
from http.client import HTTPConnection
import json
import threading
import unittest

from kwon_lab.writing.call_response import plan_call
from kwon_lab.writing.web_server import make_server


class CallResponseTests(unittest.TestCase):
    def test_call_returns_fixed_response_and_motion_denied_geometry(self):
        job = plan_call("  야  ")
        self.assertEqual(job["input"], "야")
        self.assertEqual(job["response"], "네")
        self.assertEqual(job["response_source"], "fixed_call_response")
        self.assertEqual(job["mode"], "geometry_preview_only")
        self.assertEqual(job["plan"]["text"], "네")
        self.assertEqual(job["plan"]["paper"]["width_mm"], 210)
        self.assertEqual(job["plan"]["paper"]["height_mm"], 297)
        self.assertEqual(job["plan"]["paper"]["character_mm"], 28)
        self.assertFalse(job["plan"]["motion_authorized"])
        self.assertTrue(job["plan"]["physical_calibration_required"])
        self.assertEqual(datetime.fromisoformat(job["created_at"]).utcoffset(), timedelta(hours=9))
        self.assertNotEqual(job["job_id"], plan_call("야")["job_id"])

    def test_call_normalizes_decomposed_korean(self):
        import unicodedata
        self.assertEqual(plan_call(unicodedata.normalize("NFD", "야"))["input"], "야")

    def test_random_questions_and_invalid_inputs_are_not_fake_ai_answers(self):
        for text in (None, 3, [], "", "야!", "사과가 무슨 색이야", "hello", "야" * 121):
            with self.subTest(text=text), self.assertRaises(ValueError):
                plan_call(text)


class WritingServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = make_server(0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def request(self, method, path, body=None, headers=None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=3)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_page_and_local_icons_are_served(self):
        for path in ("/", "/app.css", "/app.js", "/icons/arrow-up.svg"):
            with self.subTest(path=path):
                status, headers, body = self.request("GET", path)
                self.assertEqual(status, 200)
                self.assertTrue(body)
                self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])

    def test_valid_call_api(self):
        status, _, body = self.request("POST", "/api/call", json.dumps({"input": "야"}),
                                       {"Content-Type": "application/json"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["response"], "네")

    def test_invalid_payloads_fail_explicitly(self):
        for body in ('{"input":"hello"}', "null", "[]", "{", "{}"):
            with self.subTest(body=body):
                status, _, response = self.request("POST", "/api/call", body,
                                                   {"Content-Type": "application/json"})
                self.assertEqual(status, 400)
                self.assertIn("error", json.loads(response))

    def test_foreign_host_and_origin_are_rejected(self):
        for headers in ({"Host": "foreign.example"}, {"Origin": "https://foreign.example"}):
            with self.subTest(headers=headers):
                status, _, _ = self.request("GET", "/", headers=headers)
                self.assertEqual(status, 403)

    def test_content_type_size_and_non_allowlisted_paths(self):
        self.assertEqual(self.request("POST", "/api/call", "test")[0], 415)
        self.assertEqual(self.request("POST", "/api/call", "a" * 4097,
                                      {"Content-Type": "application/json"})[0], 413)
        self.assertEqual(self.request("GET", "/../../pyproject.toml")[0], 404)
        self.assertEqual(self.request("POST", "/api/unknown", "{}")[0], 404)


if __name__ == "__main__":
    unittest.main()
