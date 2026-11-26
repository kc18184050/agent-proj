import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def main():
    root = Path(__file__).resolve().parent
    os.chdir(root)
    load_dotenv(root / ".env")
    os.environ.setdefault("LANGFLOW_CONFIG_DIR", str(root / ".langflow"))
    os.environ.setdefault("LANGFLOW_DO_NOT_TRACK", "true")

    from langflow.__main__ import main as langflow_main

    sys.argv = [
        "langflow", "run", "--host", "127.0.0.1", "--port", "7860",
        "--open-browser", "--workers", "1",
    ]
    langflow_main()


if __name__ == "__main__":
    main()
