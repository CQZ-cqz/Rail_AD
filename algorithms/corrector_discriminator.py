"""
corrector_discriminator.py – Assess whether a point cloud segment needs
motion-distortion correction before anomaly detection.

Returns one of three outcomes:
  RESCAN_REQUIRED   – scan quality too low (valid frame ratio below threshold);
                      the segment should be returned to UNSCANNED state.
  APPLY_CORRECTION  – quality ok, lateral deviation exceeds threshold;
                      run RailMotionCorrector before detection.
  SKIP_CORRECTION   – quality ok, deviation within threshold;
                      proceed directly to detection.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class AssessResult(Enum):
    RESCAN_REQUIRED  = "rescan_required"
    APPLY_CORRECTION = "apply_correction"
    SKIP_CORRECTION  = "skip_correction"


class CorrectionDiscriminator:
    """
    Lightweight discriminator that reuses RailMotionCorrector's frame-slicing
    and rail-head extraction without executing the full correction pipeline.
    """

    def __init__(self, cfg: Dict):
        disc_cfg = cfg.get("correction", {}).get("discriminator", {})
        corr_cfg = cfg.get("correction", {}).get("corrector", {})

        self._enabled             = cfg.get("correction", {}).get("enabled", True)
        self._disc_enabled        = disc_cfg.get("enabled", True)
        self._lateral_threshold   = float(disc_cfg.get("lateral_std_threshold_mm", 1.0))
        self._min_valid_ratio     = float(disc_cfg.get("min_valid_frame_ratio", 0.5))

        # Build a RailMotionCorrector solely for its _slice_frames and extractor
        from algorithms.RailMotionCorrector import RailMotionCorrector
        self._corrector = RailMotionCorrector(
            frame_thickness_mm    = float(corr_cfg.get("frame_thickness_mm", 1.0)),
            smooth_window_frames  = int(corr_cfg.get("smooth_window_frames", 301)),
            smooth_polyorder      = int(corr_cfg.get("smooth_polyorder", 3)),
            rail_head_percentile  = float(corr_cfg.get("rail_head_percentile", 92.0)),
            min_points_per_frame  = int(corr_cfg.get("min_points_per_frame", 10)),
            dual_rail_mode        = bool(corr_cfg.get("dual_rail_mode", False)),
            gauge_nominal_mm      = float(corr_cfg.get("gauge_nominal_mm", 1435.0)),
        )

    def assess(self, points: np.ndarray) -> Tuple[AssessResult, Dict]:
        """
        Assess a point cloud segment.

        Args:
            points: (N, 3) float32 numpy array.

        Returns:
            (AssessResult, diag_dict)
            diag_dict contains: valid_frame_ratio, dx_std, n_frames, n_valid_frames
        """
        if not self._enabled:
            return AssessResult.SKIP_CORRECTION, {}

        frames = self._corrector._slice_frames(points)
        n_frames = len(frames)

        if n_frames == 0:
            logger.warning("Discriminator: no frames extracted, treating as low quality.")
            return AssessResult.RESCAN_REQUIRED, {
                "valid_frame_ratio": 0.0, "dx_std": 0.0,
                "n_frames": 0, "n_valid_frames": 0,
            }

        x_vals = []
        valid_count = 0

        for _, idx in frames:
            xz = points[idx][:, [0, 2]]
            if self._corrector.dual_rail:
                result = self._corrector.extractor.extract_dual_rail(
                    xz, self._corrector.gauge_nominal
                )
                if result is not None:
                    x_vals.append(
                        (result["left_rail"]["x_center"] +
                         result["right_rail"]["x_center"]) / 2.0
                    )
                    valid_count += 1
            else:
                result = self._corrector.extractor.extract_single_rail(xz)
                if result is not None:
                    x_vals.append(result["x_center"])
                    valid_count += 1

        valid_frame_ratio = valid_count / n_frames
        dx_std = float(np.std(x_vals)) if len(x_vals) >= 2 else 0.0

        diag = {
            "valid_frame_ratio": round(valid_frame_ratio, 4),
            "dx_std":            round(dx_std, 4),
            "n_frames":          n_frames,
            "n_valid_frames":    valid_count,
        }

        logger.info(
            "Discriminator: valid_frame_ratio=%.3f, dx_std=%.3f mm",
            valid_frame_ratio, dx_std,
        )

        if valid_frame_ratio < self._min_valid_ratio:
            return AssessResult.RESCAN_REQUIRED, diag

        if not self._disc_enabled or dx_std > self._lateral_threshold:
            return AssessResult.APPLY_CORRECTION, diag

        return AssessResult.SKIP_CORRECTION, diag