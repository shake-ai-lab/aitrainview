"""Entry point: python backend/run.py [--port 8765]"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aitrainview.server import main  # noqa: E402

if __name__ == "__main__":
    main()
