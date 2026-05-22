# src/deepagent/main.py
import asyncio
import io
import sys

from deepagent.config import Config
from deepagent.cli.app import run_cli

# Force UTF-8 output on Windows to avoid GBK encoding errors with emoji etc.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )


def main():
    try:
        config = Config()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    asyncio.run(run_cli(config))


if __name__ == "__main__":
    main()
