# src/deepagent/main.py
import asyncio
import sys

from deepagent.config import Config
from deepagent.cli.app import run_cli


def main():
    try:
        config = Config()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    asyncio.run(run_cli(config))


if __name__ == "__main__":
    main()
