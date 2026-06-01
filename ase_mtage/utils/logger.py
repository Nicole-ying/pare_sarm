"""Minimal timestamped logger for ASE-MTAGE nohup runs."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import TextIO


class Logger:
    """Prints timestamped messages to stdout and optionally a log file.

    When ``log_path`` is set, stderr is also tee'd into the same file so
    tracebacks and other stderr output appear in one consolidated log.
    """

    def __init__(self, *, log_path: str | Path | None = None, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self._file: TextIO | None = None
        self._stderr_orig = sys.stderr
        if log_path:
            p = Path(log_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._file = p.open("a", encoding="utf-8")
            sys.stderr = _Tee(sys.stderr, self._file)

    def _emit(self, msg: str) -> None:
        if not self.enabled:
            return
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        if self._file:
            self._file.write(line + "\n")
            self._file.flush()

    def section(self, title: str) -> None:
        self._emit("=" * 60)
        self._emit(f"  {title}")
        self._emit("=" * 60)

    def info(self, msg: str) -> None:
        self._emit(msg)

    def llm_call(self, agent: str, model: str, prompt_len: int, resp_len: int, duration_s: float, success: bool = True) -> None:
        status = "OK" if success else "FAILED"
        self._emit(f"LLM | {agent} | model={model} | prompt={prompt_len:,}B | resp={resp_len:,}B | {duration_s:.1f}s | {status}")

    def training_start(self, round_idx: int, candidate_id: str, timesteps: int, n_envs: int = 1) -> None:
        self._emit(f"TRAIN | round={round_idx} | candidate={candidate_id} | timesteps={timesteps:,} | n_envs={n_envs} | starting...")

    def training_progress(self, round_idx: int, done: int, total: int, fps: float | None = None) -> None:
        pct = done / max(total, 1) * 100
        bar_width = 30
        filled = int(bar_width * done / max(total, 1))
        bar = "█" * filled + "░" * (bar_width - filled)
        fps_str = f" | fps={fps:.0f}" if fps else ""
        self._emit(f"TRAIN | round={round_idx} | [{bar}] {pct:.0f}% | {done:,}/{total:,}{fps_str}")

    def training_done(self, round_idx: int, candidate_id: str, success: bool) -> None:
        status = "OK" if success else "FAILED"
        self._emit(f"TRAIN | round={round_idx} | candidate={candidate_id} | {status}")

    def round_start(self, round_idx: int, total_rounds: int, phase: str) -> None:
        self.section(f"Round {round_idx}/{total_rounds - 1} [{phase}]")

    def round_done(self, round_idx: int, summary: str) -> None:
        self._emit(f"ROUND | round={round_idx} | {summary}")

    def gif_recorded(self, path: str) -> None:
        self._emit(f"GIF | saved {path}")

    def close(self) -> None:
        if self._file:
            sys.stderr = self._stderr_orig
            self._file.close()
            self._file = None


class _Tee:
    """Tee stderr to both the original stream and a log file.
    Survives broken console pipe (nohup).
    """

    def __init__(self, stream, log_handle):
        self.stream = stream
        self.log = log_handle

    def write(self, data: str):
        self.log.write(data)
        self.log.flush()
        try:
            self.stream.write(data)
        except Exception:
            pass

    def flush(self):
        try:
            self.stream.flush()
        except Exception:
            pass


# Module-level singleton for convenience
_logger: Logger | None = None


def get_logger() -> Logger:
    global _logger
    if _logger is None:
        _logger = Logger()
    return _logger


def setup_logger(*, log_path: str | Path | None = None, enabled: bool = True) -> Logger:
    global _logger
    if _logger is not None:
        _logger.close()
    _logger = Logger(log_path=log_path, enabled=enabled)
    return _logger
