"""Run the local text-call writing preview, without microphone or robot access."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kwon_lab.writing.web_server import make_server
from kwon_lab.writing.jobs import DEFAULT_RECORDS, validate_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--config", type=Path, help="Complete uncalibrated simulation profile")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else None
        if config is not None:
            validate_config(config)
        server = make_server(args.port, records=args.records, config=config)
    except (OSError, ValueError, TypeError) as error:
        parser.error(f"Cannot start the local server: {error}. Try another --port.")
    print(f"Writing preview: http://127.0.0.1:{server.server_address[1]}", flush=True)
    print(f"Simulation jobs: http://127.0.0.1:{server.server_address[1]}/jobs", flush=True)
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
