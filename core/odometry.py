"""
odometry.py – Mileage tracking for the rail inspection system.

Two implementations are provided:

  EncoderOdometry  – reads a Modbus-RTU single-turn absolute encoder over TCP,
                     accumulates multi-turn count in software (identical logic
                     to the provided example script, wrapped in a background thread
                     with auto-reconnect).

  VirtualOdometry  – simulates linear movement at a constant speed.
                     Use for local debugging when the encoder is not connected.

Factory
───────
    from odometry import create_odometry
    odo = create_odometry(cfg)   # cfg = full config dict
    odo.start()
    mm  = odo.get_mileage_mm()
    odo.stop()

Set ``odometry.enabled: false`` in config.yaml to switch to virtual mode.
"""

import logging
import socket
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────────────────────

class OdometryBase(ABC):
    """Common interface for all odometry sources."""

    @abstractmethod
    def start(self) -> None:
        """Start the background polling thread."""

    @abstractmethod
    def stop(self) -> None:
        """Stop polling and release resources."""

    @abstractmethod
    def get_mileage_mm(self) -> float:
        """Return accumulated mileage in mm (thread-safe)."""

    def reset(self) -> None:
        """Reset mileage accumulator to zero."""
        raise NotImplementedError(f"{type(self).__name__} does not support reset().")


# ──────────────────────────────────────────────────────────────
# Real encoder
# ──────────────────────────────────────────────────────────────

class EncoderOdometry(OdometryBase):
    """
    Reads a single-turn absolute encoder via Modbus-RTU over TCP.

    Multi-turn mileage is accumulated in software by detecting wrap-arounds
    (the same technique used in the reference encoder.py script).
    The background thread reconnects automatically on network errors.
    """

    def __init__(self, cfg: dict):
        self._ip          = cfg["tcp_ip"]
        self._port        = cfg["tcp_port"]
        self._resolution  = cfg["resolution"]           # pulses / revolution
        self._mm_per_rev  = cfg["travel_per_rev_mm"]    # mm / revolution
        self._slave_id    = cfg.get("slave_id", 0x01)
        self._reg_addr    = cfg.get("reg_addr", 0x0000)
        self._update_hz   = cfg.get("update_rate_hz", 50)

        self._mileage_mm: float = 0.0
        self._lock              = threading.Lock()
        self._running: bool     = False
        self._thread: Optional[threading.Thread] = None
        self._sock:   Optional[socket.socket]    = None

        # Multi-turn state
        self._last_pos:   Optional[int] = None
        self._multi_turn: int           = 0

        self._read_cmd = self._build_cmd()

    # ── Modbus helpers ───────────────────────────────────────

    @staticmethod
    def _crc16(data: bytes) -> bytes:
        crc = 0xFFFF
        for b in data:
            crc ^= b
            for _ in range(8):
                crc = (crc >> 1) ^ 0xA001 if (crc & 1) else (crc >> 1)
        return crc.to_bytes(2, "little")

    def _build_cmd(self) -> bytes:
        frame = bytes([
            self._slave_id,
            0x03,                                # Function: Read Holding Registers
            self._reg_addr >> 8,
            self._reg_addr & 0xFF,
            0x00, 0x01,                          # Quantity: 1 register (2 bytes)
        ])
        return frame + self._crc16(frame)

    @staticmethod
    def _parse(data: bytes) -> Optional[int]:
        """Parse Modbus response; return single-turn position or None on error."""
        if len(data) < 7 or data[1] != 0x03:
            return None
        return (data[3] << 8) | data[4]

    # ── TCP connection ───────────────────────────────────────

    def _connect(self) -> Optional[socket.socket]:
        """Block until connected or _running becomes False."""
        while self._running:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(2.0)
                s.connect((self._ip, self._port))
                logger.info("Encoder connected to %s:%d.", self._ip, self._port)
                return s
            except (OSError, socket.timeout) as exc:
                logger.warning("Encoder connect failed (%s). Retry in 2 s…", exc)
                time.sleep(2.0)
        return None

    # ── Background loop ──────────────────────────────────────

    def _loop(self) -> None:
        interval = 1.0 / self._update_hz
        self._sock = self._connect()
        if self._sock is None:
            return

        while self._running:
            try:
                self._sock.send(self._read_cmd)
                raw = self._sock.recv(32)
                pos = self._parse(raw)
                if pos is None:
                    continue

                # Detect wrap-around (single-turn → multi-turn accumulation)
                if self._last_pos is not None:
                    diff = pos - self._last_pos
                    half = self._resolution // 2
                    if diff < -half:        # forward wrap  (e.g. 1022 → 3)
                        self._multi_turn += 1
                    elif diff > half:       # backward wrap (e.g. 3 → 1022)
                        self._multi_turn -= 1

                self._last_pos   = pos
                total_pulses     = self._multi_turn * self._resolution + pos
                mileage          = total_pulses * (self._mm_per_rev / self._resolution)

                with self._lock:
                    self._mileage_mm = mileage

                time.sleep(interval)

            except (OSError, socket.timeout) as exc:
                logger.warning("Encoder read error (%s). Reconnecting…", exc)
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = self._connect()
                if self._sock is None:
                    break

    # ── Public API ───────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, daemon=True, name="EncoderOdo"
        )
        self._thread.start()
        logger.info("EncoderOdometry started.")

    def stop(self) -> None:
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
        if self._thread:
            self._thread.join(timeout=4.0)
        logger.info("EncoderOdometry stopped. Final mileage: %.2f mm.", self._mileage_mm)

    def get_mileage_mm(self) -> float:
        with self._lock:
            return self._mileage_mm

    def reset(self) -> None:
        with self._lock:
            self._mileage_mm = 0.0
            self._last_pos   = None
            self._multi_turn = 0
        logger.info("EncoderOdometry reset to zero.")


# ──────────────────────────────────────────────────────────────
# Virtual odometry (debug)
# ──────────────────────────────────────────────────────────────

class VirtualOdometry(OdometryBase):
    """
    Simulates linear travel at a constant speed.
    Useful for end-to-end pipeline testing without physical hardware.
    """

    def __init__(self, cfg: dict):
        self._speed_mm_s: float          = cfg.get("speed_mm_per_s", 1000.0)
        self._mileage_mm: float          = 0.0
        self._lock                        = threading.Lock()
        self._running: bool              = False
        self._thread: Optional[threading.Thread] = None
        self._t0:     Optional[float]    = None

    def _loop(self) -> None:
        self._t0 = time.monotonic()
        while self._running:
            elapsed = time.monotonic() - self._t0
            with self._lock:
                self._mileage_mm = elapsed * self._speed_mm_s
            time.sleep(0.02)   # ~50 Hz update

    def start(self) -> None:
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, daemon=True, name="VirtualOdo"
        )
        self._thread.start()
        logger.info("VirtualOdometry started (speed=%.1f mm/s).", self._speed_mm_s)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("VirtualOdometry stopped. Final mileage: %.2f mm.", self._mileage_mm)

    def get_mileage_mm(self) -> float:
        with self._lock:
            return self._mileage_mm

    def reset(self) -> None:
        self._t0 = time.monotonic()
        with self._lock:
            self._mileage_mm = 0.0
        logger.info("VirtualOdometry reset to zero.")


# ──────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────

def create_odometry(cfg: dict) -> OdometryBase:
    """
    Return the appropriate OdometryBase subclass based on config.

    ``odometry.enabled: true``  → EncoderOdometry  (requires hardware)
    ``odometry.enabled: false`` → VirtualOdometry   (debug/testing)
    """
    if cfg["odometry"].get("enabled", True):
        logger.info("Odometry: using real encoder.")
        return EncoderOdometry(cfg["odometry"])
    else:
        logger.info("Odometry: using virtual simulation (debug mode).")
        return VirtualOdometry(cfg.get("virtual_odometry", {}))
