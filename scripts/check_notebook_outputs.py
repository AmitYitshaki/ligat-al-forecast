"""Verify staged notebooks through nbstripout without modifying working files."""

import json
import subprocess
import sys


def main(paths: list[str]) -> int:
    for path in paths:
        staged = subprocess.run(["git", "show", f":{path}"], capture_output=True, check=True).stdout
        stripped = subprocess.run(
            [sys.executable, "-m", "nbstripout"],
            input=staged,
            capture_output=True,
            check=True,
        ).stdout
        if json.loads(staged) != json.loads(stripped):
            print(f"Staged notebook contains outputs/metadata: {path}; install nbstripout filter.")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
