"""
Logging setup: Tee stdout/stderr to experiment.log.
"""

import sys
from pathlib import Path


class _LogFile:
    """Shared log file handle for both stdout and stderr tee."""

    def __init__(self, path: Path):
        self.handle = path.open("w", encoding="utf-8", buffering=1)

    def write(self, data: str):
        self.handle.write(data)

    def flush(self):
        self.handle.flush()


class _Tee:
    """Tee to both console and log file. Survives broken console pipe."""

    def __init__(self, stream, log_file: _LogFile):
        self.stream = stream
        self.log = log_file

    def write(self, data: str):
        self.log.write(data)
        try:
            self.stream.write(data)
        except Exception:
            pass  # console disconnected (nohup), keep writing to log

    def flush(self):
        self.log.flush()
        try:
            self.stream.flush()
        except Exception:
            pass


def setup_logging(exp_dir: Path):
    """Redirect stdout and stderr to tee into experiment.log."""
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_file = _LogFile(exp_dir / "experiment.log")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    return log_file
