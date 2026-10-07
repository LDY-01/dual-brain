"""Run the local text-call writing preview, without microphone or robot access."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from writing.web_server import make_server


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    try:
        server = make_server(args.port)
    except OSError as error:
        parser.error(f"Cannot start the local server: {error}. Try another --port.")
    print(f"Writing preview: http://127.0.0.1:{server.server_address[1]}", flush=True)
    print("Mode: fixed text-call response; no AI, microphone, or robot commands", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
