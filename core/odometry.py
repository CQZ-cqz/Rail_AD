"""
odometry.py – Mileage tracking for the rail inspection system.

Two implementations are provided:

    EncoderOdometry  – reads a Modbus-RTU single-turn absolute encoder via
                                         switchable transport (TCP or serial/COM), accumulates
                                         multi-turn count in software, and auto-reconnects.

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

try:
    import serial
    from serial import SerialException
except ImportError:  # serial mode will raise a clearer runtime error on use
    serial = None
    SerialException = OSError

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
    Reads a single-turn absolute encoder via Modbus-RTU over TCP or serial.

    Multi-turn mileage is accumulated in software by detecting wrap-arounds
    (the same technique used in the reference encoder.py/encoder_com.py script).
    The background thread reconnects automatically on communication errors.
    """

    def __init__(self, cfg: dict):
        self._connection_type = str(cfg.get("connection_type", "tcp")).lower()
        if self._connection_type in ("com", "rs485", "serial"):
            self._connection_type = "serial"
        elif self._connection_type in ("tcp", "ethernet", "network"):
            self._connection_type = "tcp"
        else:
            raise ValueError(
                f"Invalid odometry.connection_type '{self._connection_type}'. "
                "Must be 'tcp' or 'serial'."
            )

        # TCP configuration (for connection_type=tcp)
        self._ip          = cfg.get("tcp_ip", "169.254.88.200")
        self._port        = int(cfg.get("tcp_port", 4001))
        self._tcp_timeout = float(cfg.get("tcp_timeout", 2.0))

        # Serial configuration (for connection_type=serial)
        self._serial_port = cfg.get("serial_port", "COM1")
        self._baudrate    = int(cfg.get("baudrate", 9600))
        self._bytesize    = int(cfg.get("bytesize", 8))
        self._parity      = str(cfg.get("parity", "N"))
        self._stopbits    = cfg.get("stopbits", 1)
        self._ser_timeout = float(cfg.get("timeout", 1.0))

        self._resolution  = cfg["resolution"]           # pulses / revolution
        self._mm_per_rev  = cfg["travel_per_rev_mm"]    # mm / revolution
        self._slave_id    = cfg.get("slave_id", 0x01)
        self._reg_addr    = cfg.get("reg_addr", 0x0000)
        self._update_hz   = cfg.get("update_rate_hz", 50)

        self._mileage_mm: float = 0.0
        self._lock              = threading.Lock()
        self._running: bool     = False
        self._thread: Optional[threading.Thread] = None
        self._conn: Optional[object]             = None

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

    def _parse(self, data: bytes) -> Optional[int]:
        """Parse Modbus response; return single-turn position or None on error."""
        if len(data) < 7:
            return None

        # Search for a valid 7-byte RTU frame in the received payload.
        for idx in range(0, len(data) - 6):
            frame = data[idx:idx + 7]
            if frame[0] != self._slave_id or frame[1] != 0x03 or frame[2] != 0x02:
                continue
            if frame[5:7] != self._crc16(frame[:5]):
                continue
            return (frame[3] << 8) | frame[4]
        return None

    # ── Connection helpers ───────────────────────────────────

    def _comm_exceptions(self):
        exc_types = [OSError, socket.timeout]
        if serial is not None:
            exc_types.append(SerialException)
        return tuple(exc_types)

    def _close_connection(self) -> None:
        if not self._conn:
            return
        try:
            self._conn.close()
        except self._comm_exceptions():
            pass
        finally:
            self._conn = None

    def _connect_tcp(self) -> Optional[object]:
        while self._running:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(self._tcp_timeout)
                s.connect((self._ip, self._port))
                logger.info("Encoder connected via TCP %s:%d.", self._ip, self._port)
                return s
            except self._comm_exceptions() as exc:
                logger.warning("Encoder TCP connect failed (%s). Retry in 2 s...", exc)
                time.sleep(2.0)
        return None

    def _connect_serial(self) -> Optional[object]:
        if serial is None:
            raise RuntimeError(
                "pyserial is required for odometry.connection_type=serial. "
                "Install with: pip install pyserial"
            )

        while self._running:
            try:
                s = serial.Serial(
                    port=self._serial_port,
                    baudrate=self._baudrate,
                    bytesize=self._bytesize,
                    parity=self._parity,
                    stopbits=self._stopbits,
                    timeout=self._ser_timeout,
                )
                logger.info(
                    "Encoder connected via serial %s (%d,%d,%s,%s).",
                    self._serial_port,
                    self._baudrate,
                    self._bytesize,
                    self._parity,
                    self._stopbits,
                )
                return s
            except self._comm_exceptions() as exc:
                logger.warning("Encoder serial open failed (%s). Retry in 2 s...", exc)
                time.sleep(2.0)
        return None

    def _connect(self) -> Optional[object]:
        """Block until connected or _running becomes False."""
        if self._connection_type == "serial":
            return self._connect_serial()
        return self._connect_tcp()

    def _read_frame(self) -> bytes:
        """Send one Modbus read request and return raw response bytes."""
        if self._conn is None:
            raise OSError("Encoder connection is not established.")

        if self._connection_type == "serial":
            self._conn.write(self._read_cmd)
            return self._conn.read(8)

        # TCP mode
        self._conn.sendall(self._read_cmd)
        return self._conn.recv(32)

    # ── Background loop ──────────────────────────────────────

    def _loop(self) -> None:
        interval = 1.0 / self._update_hz
        self._conn = self._connect()
        if self._conn is None:
            return

        while self._running:
            try:
                raw = self._read_frame()
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

            except self._comm_exceptions() as exc:
                logger.warning("Encoder read error (%s). Reconnecting…", exc)
                self._close_connection()
                self._conn = self._connect()
                if self._conn is None:
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
        self._close_connection()
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
