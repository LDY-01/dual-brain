"""Loopback-only writing preview server, using Python's standard library."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlsplit

from .call_response import plan_call


STATIC_DIR = Path(__file__).with_name("web")
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    **{f"/icons/{name}.svg": (f"icons/{name}.svg", "image/svg+xml")
       for name in ("arrow-up", "square", "rotate-ccw", "download", "pen-tool")},
}


class WritingHandler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; script-src 'self'; style-src 'self'; "
                         "img-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data: dict) -> None:
        self._send(status, json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _local_request(self) -> bool:
        port = self.server.server_address[1]
        hosts = (f"127.0.0.1:{port}", f"localhost:{port}")
        if self.headers.get("Host") not in hosts:
            self._json(403, {"error": "로컬 주소로만 접속할 수 있습니다."})
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in tuple(f"http://{host}" for host in hosts):
            self._json(403, {"error": "다른 사이트의 요청은 허용하지 않습니다."})
            return False
        return True

    def do_GET(self) -> None:
        if not self._local_request():
            return
        entry = STATIC_FILES.get(urlsplit(self.path).path)
        if not entry:
            self._json(404, {"error": "페이지를 찾을 수 없습니다."})
            return
        filename, content_type = entry
        try:
            body = (STATIC_DIR / filename).read_bytes()
        except OSError:
            self._json(500, {"error": "화면 파일을 불러오지 못했습니다."})
            return
        self._send(200, body, content_type)

    def do_POST(self) -> None:
        if not self._local_request():
            return
        if self.path != "/api/call":
            self._json(404, {"error": "지원하지 않는 요청입니다."})
            return
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip() != "application/json":
            self._json(415, {"error": "JSON 요청만 지원합니다."})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 4096:
                self._json(413, {"error": "요청 크기가 허용 범위를 벗어났습니다."})
                return
            self.connection.settimeout(5)
            payload = json.loads(self.rfile.read(size).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("입력 형식이 올바르지 않습니다.")
            result = plan_call(payload.get("input"))
        except (ValueError, UnicodeDecodeError) as error:
            self._json(400, {"error": str(error)})
            return
        except (OSError, TimeoutError):
            self._json(408, {"error": "요청을 읽는 시간이 초과됐습니다."})
            return
        self._json(200, result)


def make_server(port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), WritingHandler)
