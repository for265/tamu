"""
keyboard_logger.py - Personal typing-analytics logger.

Records YOUR OWN keystroke activity on YOUR OWN machine and writes a fresh CSV
every hour. It is meant for self-tracking (typing volume, speed, active hours),
not for monitoring anyone else. It runs openly and prints a banner while active.

Privacy:
    By default it does NOT store the literal characters you type. Each keystroke
    is recorded as a *category* (letter / digit / space / punctuation / ...) plus
    timing. That is enough for volume, WPM and active-hours analytics without
    keeping a plaintext transcript of everything (passwords included).
    Pass --raw-keys only if you specifically want the actual characters stored.

Install the one dependency:
    pip install pynput

Run it:
    python keyboard_logger.py                      # logs to ./keystroke_data/
    python keyboard_logger.py --out ~/typing_data  # pick an output folder
    python keyboard_logger.py --raw-keys           # also store literal keys
    python keyboard_logger.py --flush-secs 10      # how often to flush to disk

Stop it:
    Ctrl+C in the terminal. It flushes and writes a final summary row.

Output (in the chosen folder):
    events_YYYY-MM-DD_HH.csv   one file per clock hour, one row per keystroke
    hourly_summary.csv         one appended row per completed hour (totals)
"""

import argparse
import csv
import os
import signal
import string
import sys
import threading
from datetime import datetime

try:
    from pynput import keyboard
except ImportError:
    sys.exit(
        "Missing dependency 'pynput'. Install it with:\n    pip install pynput"
    )


EVENT_HEADER = [
    "timestamp_iso",
    "epoch_ms",
    "event",          # "press" or "release"
    "key_category",   # letter / digit / space / punctuation / enter / ...
    "key",            # literal key, only populated with --raw-keys
    "inter_key_ms",   # ms since the previous press (blank for the first press)
]

SUMMARY_HEADER = [
    "hour_start_iso",
    "presses",
    "unique_active_seconds",  # seconds in the hour with >=1 keypress
    "approx_wpm",             # presses/5 over active time, a rough typing rate
    "backspaces",
    "enters",
]


def categorize(key) -> str:
    """Map a pynput key to a coarse, non-identifying category."""
    if isinstance(key, keyboard.KeyCode) and key.char is not None:
        ch = key.char
        if ch in string.ascii_letters:
            return "letter"
        if ch in string.digits:
            return "digit"
        if ch in string.punctuation:
            return "punctuation"
        if ch.isspace():
            return "space"
        return "other"
    # Special keys (Key.space, Key.enter, Key.backspace, modifiers, ...)
    name = getattr(key, "name", str(key))
    if name in ("ctrl", "ctrl_l", "ctrl_r", "alt", "alt_l", "alt_r",
                "alt_gr", "shift", "shift_l", "shift_r", "cmd", "cmd_l",
                "cmd_r"):
        return "modifier"
    if name.startswith("f") and name[1:].isdigit():
        return "function"
    if name in ("up", "down", "left", "right", "home", "end",
                "page_up", "page_down"):
        return "navigation"
    if name in ("enter", "backspace", "delete", "tab", "esc", "space"):
        return name
    return "other"


def raw_key_value(key) -> str:
    """Literal key string, used only in --raw-keys mode."""
    if isinstance(key, keyboard.KeyCode) and key.char is not None:
        return key.char
    return getattr(key, "name", str(key))


class HourlyLogger:
    def __init__(self, out_dir: str, raw_keys: bool, flush_secs: float):
        self.out_dir = out_dir
        self.raw_keys = raw_keys
        self.flush_secs = flush_secs

        self.lock = threading.Lock()
        self.buffer = []              # rows waiting to be written
        self.current_hour = None      # datetime truncated to the hour
        self.file = None
        self.writer = None

        self.last_press_ms = None

        # per-hour counters for the summary
        self._reset_hour_counters()

        os.makedirs(self.out_dir, exist_ok=True)
        self._ensure_summary_header()

        self._stop = threading.Event()
        self._flusher = threading.Thread(target=self._flush_loop, daemon=True)

    # --- hour/file management ------------------------------------------------
    def _reset_hour_counters(self):
        self.presses = 0
        self.backspaces = 0
        self.enters = 0
        self.active_seconds = set()   # distinct second-of-hour with a press

    def _hour_floor(self, dt: datetime) -> datetime:
        return dt.replace(minute=0, second=0, microsecond=0)

    def _open_hour_file(self, hour: datetime):
        path = os.path.join(
            self.out_dir, f"events_{hour.strftime('%Y-%m-%d_%H')}.csv"
        )
        is_new = not os.path.exists(path)
        self.file = open(path, "a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        if is_new:
            self.writer.writerow(EVENT_HEADER)
        self.current_hour = hour

    def _roll_if_needed(self, now: datetime):
        hour = self._hour_floor(now)
        if hour != self.current_hour:
            if self.current_hour is not None:
                # Drain buffered rows into the OLD hour's file before closing,
                # otherwise they would be written to the new hour's file.
                self._drain_locked()
                self._write_summary(self.current_hour)
                self._reset_hour_counters()
            if self.file:
                self.file.flush()
                self.file.close()
            self._open_hour_file(hour)

    # --- summary -------------------------------------------------------------
    def _ensure_summary_header(self):
        path = os.path.join(self.out_dir, "hourly_summary.csv")
        if not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(SUMMARY_HEADER)

    def _write_summary(self, hour: datetime):
        active = max(len(self.active_seconds), 1)
        # Classic typing metric: 1 "word" ~= 5 keystrokes.
        approx_wpm = round((self.presses / 5.0) / (active / 60.0), 1)
        path = os.path.join(self.out_dir, "hourly_summary.csv")
        with open(path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                hour.isoformat(timespec="seconds"),
                self.presses,
                len(self.active_seconds),
                approx_wpm,
                self.backspaces,
                self.enters,
            ])

    # --- event intake --------------------------------------------------------
    def on_press(self, key):
        now = datetime.now()
        epoch_ms = int(now.timestamp() * 1000)
        category = categorize(key)

        inter = ""
        if self.last_press_ms is not None:
            inter = epoch_ms - self.last_press_ms
        self.last_press_ms = epoch_ms

        row = [
            now.isoformat(timespec="milliseconds"),
            epoch_ms,
            "press",
            category,
            raw_key_value(key) if self.raw_keys else "",
            inter,
        ]

        with self.lock:
            self._roll_if_needed(now)
            self.buffer.append(row)
            self.presses += 1
            self.active_seconds.add(now.replace(microsecond=0))
            if category == "backspace":
                self.backspaces += 1
            if category == "enter":
                self.enters += 1

    def on_release(self, key):
        now = datetime.now()
        row = [
            now.isoformat(timespec="milliseconds"),
            int(now.timestamp() * 1000),
            "release",
            categorize(key),
            raw_key_value(key) if self.raw_keys else "",
            "",
        ]
        with self.lock:
            self._roll_if_needed(now)
            self.buffer.append(row)

    # --- flushing ------------------------------------------------------------
    def _flush_loop(self):
        while not self._stop.wait(self.flush_secs):
            self.flush()

    def _drain_locked(self):
        """Write buffered rows with the current writer. Caller holds the lock."""
        if self.buffer and self.writer:
            self.writer.writerows(self.buffer)
            self.file.flush()
            self.buffer.clear()

    def flush(self):
        with self.lock:
            self._drain_locked()

    def start(self):
        self._flusher.start()

    def close(self):
        self._stop.set()
        self.flush()
        with self.lock:
            if self.current_hour is not None:
                self._write_summary(self.current_hour)
            if self.file:
                self.file.flush()
                self.file.close()


def main():
    parser = argparse.ArgumentParser(
        description="Personal typing-analytics logger (your own machine)."
    )
    parser.add_argument(
        "--out", default="keystroke_data",
        help="Output folder for the CSV files (default: ./keystroke_data).",
    )
    parser.add_argument(
        "--raw-keys", action="store_true",
        help="Store the literal characters typed, not just categories. "
             "Off by default for privacy.",
    )
    parser.add_argument(
        "--flush-secs", type=float, default=5.0,
        help="How often (seconds) to flush buffered events to disk.",
    )
    args = parser.parse_args()

    out_dir = os.path.abspath(os.path.expanduser(args.out))
    logger = HourlyLogger(out_dir, args.raw_keys, args.flush_secs)

    print("=" * 62)
    print(" TYPING-ANALYTICS LOGGER IS RECORDING YOUR KEYSTROKES")
    print("=" * 62)
    print(f" Output folder : {out_dir}")
    print(f" Mode          : {'RAW KEYS (literal characters)' if args.raw_keys else 'categories only (privacy-preserving)'}")
    print(" New CSV        : every clock hour (events_YYYY-MM-DD_HH.csv)")
    print(" Summary        : hourly_summary.csv")
    print(" Stop           : press Ctrl+C")
    print("=" * 62, flush=True)

    logger.start()

    listener = keyboard.Listener(
        on_press=logger.on_press, on_release=logger.on_release
    )

    def shutdown(*_):
        print("\nStopping, flushing remaining data...", flush=True)
        listener.stop()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    listener.start()
    listener.join()      # blocks until listener.stop()
    logger.close()
    print(f"Done. CSV files are in: {out_dir}")


if __name__ == "__main__":
    main()
