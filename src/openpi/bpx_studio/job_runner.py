from __future__ import annotations

import argparse
from datetime import UTC
from datetime import datetime
import json
import os
import pathlib
import subprocess


def _atomic_json(path: pathlib.Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=pathlib.Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    log_path, exit_path = pathlib.Path(spec["log_path"]), pathlib.Path(spec["exit_path"])
    environment = os.environ.copy()
    environment.update(spec.get("env", {}))
    return_code, error = 127, ""
    try:
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            process = subprocess.Popen(
                spec["argv"], cwd=spec["cwd"], env=environment, stdout=log, stderr=subprocess.STDOUT, text=True
            )
            return_code = process.wait()
    except BaseException as exception:
        error = f"{type(exception).__name__}: {exception}"
        with log_path.open("a", encoding="utf-8") as log:
            log.write(error + "\n")
    _atomic_json(
        exit_path,
        {"return_code": return_code, "error": error, "ended_at": datetime.now(UTC).isoformat()},
    )


if __name__ == "__main__":
    main()
