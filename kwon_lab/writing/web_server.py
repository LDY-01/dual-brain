"""Loopback-only writing preview server, using Python's standard library."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import urlsplit

from .call_response import plan_call
from .jobs import DEFAULT_RECORDS, JobBusyError, JobManager, RecordOwnerError
from .paper import PaperSpaceError


STATIC_DIR = Path(__file__).with_name("web")
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/jobs": ("jobs.html", "text/html; charset=utf-8"),
    "/jobs.css": ("jobs.css", "text/css; charset=utf-8"),
    "/jobs.js": ("jobs.js", "text/javascript; charset=utf-8"),
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
        path = urlsplit(self.path).path
        if path == "/api/page" or path == "/api/jobs" or path.startswith("/api/jobs/"):
            try:
                manager = self.server.jobs()
                parts = path.strip("/").split("/")
                if path == "/api/page":
                    self._json(200, manager.page())
                elif path == "/api/jobs":
                    self._json(200, manager.list())
                elif len(parts) == 3:
                    self._json(200, manager.get(parts[2]))
                elif len(parts) == 5 and parts[3] == "artifacts":
                    artifact = manager.artifact(parts[2], parts[4])
                    mime = "image/png" if artifact.suffix == ".png" else "application/json; charset=utf-8"
                    if artifact.suffix == ".jsonl":
                        mime = "application/x-ndjson; charset=utf-8"
                    self._send(200, artifact.read_bytes(), mime)
                else:
                    raise KeyError("Unknown endpoint")
            except KeyError:
                self._json(404, {"error": "작업이나 파일을 찾을 수 없습니다."})
            except (OSError, RecordOwnerError, ValueError):
                self._json(503, {"error": "작업 기록을 열 수 없습니다. 같은 기록 폴더의 다른 서버를 확인하세요."})
            return
        entry = STATIC_FILES.get(path)
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
        parts = self.path.strip("/").split("/")
        job_create = self.path == "/api/jobs"
        job_action = len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] in {"stop", "review"}
        page_action = self.path in {"/api/page/preview", "/api/page/reset"}
        if self.path != "/api/call" and not job_create and not job_action and not page_action:
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
            payload = json.loads(self.rfile.read(size).decode("utf-8"), parse_constant=self._invalid_constant)
            if not isinstance(payload, dict):
                raise ValueError("입력 형식이 올바르지 않습니다.")
            if self.path == "/api/call":
                result = plan_call(payload.get("input"))
            elif job_create:
                if set(payload) - {"text", "character_mm", "page_id"}:
                    raise ValueError("지원하지 않는 작업 설정입니다.")
                result = self.server.jobs().submit(payload.get("text"), payload.get("character_mm", 28), payload.get("page_id"))
            elif page_action:
                if parts[2] == "reset":
                    if set(payload) - {"page_id", "confirmed"}:
                        raise ValueError("지원하지 않는 종이 초기화 설정입니다.")
                    result = self.server.jobs().new_page(payload.get("page_id"), payload.get("confirmed", False))
                else:
                    if set(payload) - {"text", "character_mm", "page_id"}:
                        raise ValueError("지원하지 않는 배치 설정입니다.")
                    result = self.server.jobs().preview(payload.get("text"), payload.get("character_mm", 28), payload.get("page_id"))
            elif parts[3] == "stop":
                if payload:
                    raise ValueError("중지 요청에는 추가 설정을 넣을 수 없습니다.")
                result = self.server.jobs().stop(parts[2])
            else:
                if set(payload) - {"status", "readback"}:
                    raise ValueError("지원하지 않는 검수 설정입니다.")
                result = self.server.jobs().review(parts[2], payload.get("status"), payload.get("readback", ""))
        except PaperSpaceError as error:
            labels = {"insufficient_remaining_space": "종이 공간 부족: 남은 필기 공간이 부족합니다. 새 종이를 준비하세요.",
                      "text_does_not_fit_blank_sheet": "종이 공간 부족: 새 A4 가로 종이에도 들어가지 않습니다. 글자 크기나 문장 길이를 줄이세요.",
                      "paper_layout_or_simulation_profile_changed": "종이 방향 또는 가상 설정이 변경됐습니다. 새 A4 가로 종이 준비를 확인하세요.",
                      "stale_sheet_confirmation": "종이가 이미 변경됐습니다. 최신 종이를 확인하세요.",
                      "stale_sheet_submission": "종이가 변경됐습니다. 최신 배치를 확인하세요.",
                      "stale_sheet_preview": "종이가 변경됐습니다. 최신 배치를 확인하세요."}
            self._json(409, {"error": labels.get(error.reason, "새 종이 준비 확인이 필요합니다."),
                             "reason": error.reason,
                             "category": "paper_space" if error.reason in {"insufficient_remaining_space", "text_does_not_fit_blank_sheet"} else "sheet_confirmation",
                             "page": self.server.jobs().page()})
            return
        except JobBusyError as error:
            self._json(409, {"error": str(error)})
            return
        except KeyError:
            self._json(404, {"error": "작업을 찾을 수 없습니다."})
            return
        except RecordOwnerError:
            self._json(503, {"error": "다른 서버가 같은 작업 기록 폴더를 사용 중입니다."})
            return
        except (ValueError, UnicodeDecodeError) as error:
            self._json(400, {"error": str(error)})
            return
        except (OSError, TimeoutError):
            self._json(408, {"error": "요청을 읽는 시간이 초과됐습니다."})
            return
        except RuntimeError:
            self._json(503, {"error": "작업 관리자가 닫혔거나 기록 저장소를 사용할 수 없습니다."})
            return
        self._json(202 if job_create or (job_action and parts[3] == "stop") else 200, result)

    @staticmethod
    def _invalid_constant(value):
        raise ValueError(f"Invalid JSON constant: {value}")


class WritingServer(ThreadingHTTPServer):
    def __init__(self, port, records, manager, config):
        self._jobs, self.records, self.config = manager, records, config
        self.manager_lock = threading.Lock()
        super().__init__(("127.0.0.1", port), WritingHandler)

    def jobs(self):
        with self.manager_lock:
            if self._jobs is None:
                self._jobs = JobManager(self.records, config=self.config)
            return self._jobs

    def server_close(self):
        super().server_close()
        if self._jobs is not None:
            self._jobs.close()


def make_server(port: int = 8765, records=DEFAULT_RECORDS, manager=None, config=None) -> ThreadingHTTPServer:
    return WritingServer(port, records, manager, config)
