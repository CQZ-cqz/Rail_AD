"""
coverage.py – Persistent coverage map for track segment state tracking.

Each track segment has one of four states:

  UNSCANNED  →  not yet scanned
  PENDING    →  scanned, point cloud saved, awaiting detection
  NORMAL     →  detected, no anomaly (point cloud deleted)
  ANOMALY    →  detected, anomaly found (point cloud kept in anomaly/)

The map is stored as a JSON file. All writes are atomic (write to .tmp then
rename) to prevent corruption on unexpected shutdown.

Usage
─────
    from coverage import CoverageMap, SegmentState

    cm = CoverageMap("coverage_map.json", total_length_mm=100_000, segment_length_mm=3906.25)
    targets = cm.get_unscanned(limit=20)
    cm.update("42", state=SegmentState.PENDING, scan_time="2024-01-01T10:00:00")
"""

import json
import logging
import math
import threading
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Segment state
# ──────────────────────────────────────────────────────────────

class SegmentState(str, Enum):
    UNSCANNED = "unscanned"
    PENDING   = "pending"
    NORMAL    = "normal"
    ANOMALY   = "anomaly"


# ──────────────────────────────────────────────────────────────
# Coverage map
# ──────────────────────────────────────────────────────────────

class CoverageMap:
    """
    Thread-safe, disk-backed coverage map.

    Parameters
    ──────────
    map_path         : path to the JSON file (created if it doesn't exist)
    total_length_mm  : full one-way track length in mm
    segment_length_mm: physical length of one acquisition segment in mm
                       (compute via config.compute_segment_length)
    """

    _SCHEMA_VERSION = 1

    def __init__(
        self,
        map_path: str,
        total_length_mm: float,
        segment_length_mm: float,
    ):
        self._path        = Path(map_path)
        self._total_mm    = total_length_mm
        self._seg_len_mm  = segment_length_mm
        self._lock        = threading.Lock()
        self._segments:   Dict[str, dict] = {}

        self._init_or_load()

    # ── Initialisation ───────────────────────────────────────

    def _init_or_load(self) -> None:
        if self._path.exists():
            try:
                self._load()
                logger.info(
                    "Coverage map loaded: %s  (%d segments, %.2f mm each).",
                    self._path, len(self._segments), self._seg_len_mm,
                )
                return
            except Exception as exc:
                logger.warning(
                    "Failed to load coverage map (%s). Reinitialising.", exc
                )

        self._build_fresh()
        self._persist()
        logger.info(
            "Coverage map initialised: %d segments, %.2f mm each.",
            len(self._segments), self._seg_len_mm,
        )

    def _build_fresh(self) -> None:
        n = math.ceil(self._total_mm / self._seg_len_mm)
        self._segments = {}
        for i in range(n):
            seg_id = str(i)
            start  = i * self._seg_len_mm
            end    = min((i + 1) * self._seg_len_mm, self._total_mm)
            self._segments[seg_id] = {
                "state":            SegmentState.UNSCANNED,
                "start_mm":         round(start, 3),
                "end_mm":           round(end,   3),
                "scan_time":        None,
                "detect_time":      None,
                "point_cloud_file": None,
                "meta_file":        None,
            }

    def _load(self) -> None:
        with open(self._path, "r", encoding="utf-8") as f:
            data = json.load(f)

        stored_total = data.get("total_length_mm", 0.0)
        stored_seg   = data.get("segment_length_mm", 0.0)

        # If key geometry changed, reinitialise rather than silently corrupt state
        if abs(stored_total - self._total_mm) > 1.0:
            logger.warning(
                "Track length changed (%.0f → %.0f mm). Reinitialising coverage map.",
                stored_total, self._total_mm,
            )
            self._build_fresh()
            return

        if abs(stored_seg - self._seg_len_mm) > 1.0:
            logger.warning(
                "Segment length changed (%.2f → %.2f mm). Reinitialising coverage map.",
                stored_seg, self._seg_len_mm,
            )
            self._build_fresh()
            return

        self._total_mm   = stored_total
        self._seg_len_mm = stored_seg
        self._segments   = data["segments"]

    # ── Persistence ──────────────────────────────────────────

    def _persist(self) -> None:
        """Write map to disk atomically (temp-file + rename)."""
        payload = {
            "schema_version":   self._SCHEMA_VERSION,
            "total_length_mm":  self._total_mm,
            "segment_length_mm": self._seg_len_mm,
            "segments":         self._segments,
        }
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        tmp.replace(self._path)

    # ── Queries ──────────────────────────────────────────────

    def get_segments_by_state(
        self,
        state: SegmentState,
        limit: Optional[int] = None,
    ) -> List[Tuple[str, dict]]:
        """
        Return [(seg_id, seg_info), …] for segments in `state`,
        sorted by ascending start position.
        """
        with self._lock:
            result = [
                (k, dict(v))
                for k, v in self._segments.items()
                if v["state"] == state
            ]
        result.sort(key=lambda x: x[1]["start_mm"])
        return result[:limit] if limit is not None else result

    def get_unscanned(self, limit: Optional[int] = None) -> List[Tuple[str, dict]]:
        """Return unscanned segments, sorted by position."""
        return self.get_segments_by_state(SegmentState.UNSCANNED, limit)

    def get_pending(self) -> List[Tuple[str, dict]]:
        """Return segments awaiting detection, sorted by position."""
        return self.get_segments_by_state(SegmentState.PENDING)

    def find_by_mileage(
        self, mileage_mm: float
    ) -> Tuple[Optional[str], Optional[dict]]:
        """Return (seg_id, seg_info) for the segment containing `mileage_mm`."""
        with self._lock:
            for seg_id, seg in self._segments.items():
                if seg["start_mm"] <= mileage_mm < seg["end_mm"]:
                    return seg_id, dict(seg)
        return None, None

    def get(self, seg_id: str) -> Optional[dict]:
        """Return a copy of segment info, or None if not found."""
        with self._lock:
            seg = self._segments.get(seg_id)
            return dict(seg) if seg else None

    # ── Mutation ─────────────────────────────────────────────

    def update(self, seg_id: str, **fields) -> None:
        """
        Update one or more fields of a segment and persist to disk.

        Example
        ───────
            coverage.update("42",
                state=SegmentState.PENDING,
                scan_time="2024-01-01T10:00:00",
                point_cloud_file="/data/pending/seg_000042.ply",
            )
        """
        with self._lock:
            if seg_id not in self._segments:
                raise KeyError(f"Segment '{seg_id}' not found in coverage map.")
            self._segments[seg_id].update(fields)
            self._persist()

    # ── Summary ──────────────────────────────────────────────

    def summary(self) -> Dict[str, int]:
        """Return {state_name: count} for all segments."""
        with self._lock:
            counts: Dict[str, int] = {s.value: 0 for s in SegmentState}
            for seg in self._segments.values():
                counts[seg["state"]] += 1
        return counts

    def total_segments(self) -> int:
        with self._lock:
            return len(self._segments)

    def is_complete(self) -> bool:
        """True if every segment has been detected (normal or anomaly)."""
        with self._lock:
            return all(
                v["state"] in (SegmentState.NORMAL, SegmentState.ANOMALY)
                for v in self._segments.values()
            )
