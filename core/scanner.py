"""
scanner.py – High-level wrapper around the Mech-Eye 3D laser profiler SDK.

Responsibilities
────────────────
  1. Connect to and configure the profiler from config.yaml.
  2. Acquire one batch of profile data using a callback (non-blocking wait).
  3. Convert the ProfileBatch to a NaN-free numpy (N, 3) mm array in the
     custom coordinate system (if a transformation has been set in Mech-Eye
     Viewer; otherwise the camera frame is used with a warning).

The conversion pipeline is:
  ProfileBatch
    → get_untextured_point_cloud()        (SDK, camera frame)
    → transform_point_cloud()             (SDK, custom frame if configured)
    → save to temp PLY via SDK            (guaranteed-compatible format)
    → read_ply() → numpy (N, 3) float32   (our utils.py reader)
    → remove NaN / Inf rows

Design note: saving to a temp PLY and reloading is intentionally chosen over
trying to access the SDK's internal point cloud buffer directly.  It is
simpler, guaranteed to work across SDK versions, and the temp-file overhead
is negligible compared with the acquisition time.
"""

import logging
import os
import tempfile
import threading
from typing import Optional

import numpy as np

from mecheye.shared import *
from mecheye.profiler import *
from mecheye.profiler_utils import *

from utils import read_ply

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Internal callback
# ──────────────────────────────────────────────────────────────

class _AcquisitionCallback(AcquisitionCallbackBase):
    """
    Thread-safe acquisition callback.

    Uses threading.Event so that the main thread can block on wait()
    instead of spinning on a mutex.  Signals on the first batch received.
    """

    def __init__(self, width: int):
        super().__init__()
        self.profile_batch = ProfileBatch(width)
        self._mutex        = threading.Lock()
        self._ready        = threading.Event()

    def run(self, batch: ProfileBatch) -> None:
        with self._mutex:
            if not batch.get_error_status().is_ok():
                logger.error("Acquisition callback: error in batch.")
                show_error(batch.get_error_status())
            self.profile_batch.append(batch)
            self._ready.set()

    def wait(self, timeout: float = 120.0) -> bool:
        """Block until data arrives or timeout (seconds). Returns True on data."""
        return self._ready.wait(timeout=timeout)


# ──────────────────────────────────────────────────────────────
# Scanner
# ──────────────────────────────────────────────────────────────

class Scanner:
    """High-level wrapper around the Mech-Eye 3D laser profiler SDK."""

    # ── Enum look-up tables ──────────────────────────────────
    _ANALOG_GAIN = {
        "Gain_1": AnalogGain.Value_Gain_1,
        "Gain_2": AnalogGain.Value_Gain_2,
        "Gain_3": AnalogGain.Value_Gain_3,
        "Gain_4": AnalogGain.Value_Gain_4,
    }
    _SPOT_SEL = {
        "Strongest": SpotSelection.Value_Strongest,
        "Nearest":   SpotSelection.Value_Nearest,
        "Farthest":  SpotSelection.Value_Farthest,
        "Invalid":   SpotSelection.Value_Invalid,
    }
    _FILTER = {
        "Mean":   Filter.Value_Mean,
        "Median": Filter.Value_Median,
    }
    _MEAN_WIN = {
        2: MeanFilterWindowSize.Value_WindowSize_2,
        4: MeanFilterWindowSize.Value_WindowSize_4,
    }
    _ENC_DIR = {
        "forward":  EncoderTriggerDirection.Value_ChannelALeading,
        "backward": EncoderTriggerDirection.Value_ChannelBLeading,
        "both":     EncoderTriggerDirection.Value_Both,
    }
    _ENC_COUNT = {
        "1x": EncoderTriggerSignalCountingMode.Value_Multiple_1,
        "2x": EncoderTriggerSignalCountingMode.Value_Multiple_2,
        "4x": EncoderTriggerSignalCountingMode.Value_Multiple_4,
    }

    def __init__(self, cfg: dict):
        self._cfg        = cfg
        self._profiler   = Profiler()
        self._user_set   = None
        self._data_width: int  = 0
        self._sw_trigger: bool = True
        self._connected:  bool = False

    # ── Lifecycle ────────────────────────────────────────────

    def connect(self) -> bool:
        """Discover and connect to the first available profiler."""
        if not find_and_connect(self._profiler):
            logger.error("No profiler found or connection failed.")
            return False
        self._user_set  = self._profiler.current_user_set()
        self._connected = True
        logger.info("Connected to Mech-Eye profiler.")
        return True

    def disconnect(self) -> None:
        if self._connected:
            self._profiler.disconnect()
            self._connected = False
            logger.info("Disconnected from Mech-Eye profiler.")

    # ── Configuration ────────────────────────────────────────

    def configure(self) -> None:
        """Push all parameters from config to the profiler's current user set."""
        s    = self._cfg["scanner"]
        mode = s["trigger_mode"]

        self._apply_exposure(s["exposure"])

        if mode == "encoder":
            self._apply_encoder_mode(s["encoder"])
        else:
            self._apply_fixed_rate_mode(s["fixed_rate"])

        # Common signal / filter parameters
        show_error(self._user_set.set_int_value(
            LaserPower.name, s["laser_power"]))
        show_error(self._user_set.set_enum_value(
            AnalogGain.name,
            self._ANALOG_GAIN.get(s["analog_gain"], AnalogGain.Value_Gain_2)))
        show_error(self._user_set.set_int_value(
            DigitalGain.name, s["digital_gain"]))
        show_error(self._user_set.set_int_value(
            MinGrayscaleValue.name, s["min_grayscale"]))
        show_error(self._user_set.set_int_value(
            MinLaserLineWidth.name, s["min_laser_line_width"]))
        show_error(self._user_set.set_int_value(
            MaxLaserLineWidth.name, s["max_laser_line_width"]))
        show_error(self._user_set.set_enum_value(
            SpotSelection.name,
            self._SPOT_SEL.get(s["spot_selection"], SpotSelection.Value_Strongest)))
        show_error(self._user_set.set_int_value(
            GapFilling.name, s["gap_filling"]))
        show_error(self._user_set.set_enum_value(
            Filter.name,
            self._FILTER.get(s["filter"], Filter.Value_Mean)))
        show_error(self._user_set.set_enum_value(
            MeanFilterWindowSize.name,
            self._MEAN_WIN.get(s["mean_filter_window"],
                               MeanFilterWindowSize.Value_WindowSize_2)))

        # Read back derived values needed at acquisition time
        err, self._data_width = self._user_set.get_int_value(DataPointsPerProfile.name)
        show_error(err)

        err, trig_src = self._user_set.get_enum_value(DataAcquisitionTriggerSource.name)
        show_error(err)
        self._sw_trigger = (trig_src == DataAcquisitionTriggerSource.Value_Software)

        logger.info(
            "Scanner configured: mode=%s, data_width=%d, software_trigger=%s.",
            mode, self._data_width, self._sw_trigger,
        )

    def _apply_exposure(self, exp: dict) -> None:
        if exp["mode"] == "timed":
            show_error(self._user_set.set_enum_value(
                ExposureMode.name, ExposureMode.Value_Timed))
            show_error(self._user_set.set_int_value(
                ExposureTime.name, exp["exposure_time_us"]))
        elif exp["mode"] == "hdr":
            show_error(self._user_set.set_enum_value(
                ExposureMode.name, ExposureMode.Value_HDR))
            show_error(self._user_set.set_int_value(
                ExposureTime.name, exp["exposure_time_us"]))
            show_error(self._user_set.set_float_value(
                HdrExposureTimeProportion1.name, exp["proportion1"]))
            show_error(self._user_set.set_float_value(
                HdrExposureTimeProportion2.name, exp["proportion2"]))
            show_error(self._user_set.set_float_value(
                HdrFirstThreshold.name, exp["first_threshold"]))
            show_error(self._user_set.set_float_value(
                HdrSecondThreshold.name, exp["second_threshold"]))

    def _apply_encoder_mode(self, enc: dict) -> None:
        show_error(self._user_set.set_enum_value(
            DataAcquisitionTriggerSource.name,
            DataAcquisitionTriggerSource.Value_Software))
        show_error(self._user_set.set_enum_value(
            LineScanTriggerSource.name, LineScanTriggerSource.Value_Encoder))
        show_error(self._user_set.set_enum_value(
            EncoderTriggerDirection.name,
            self._ENC_DIR.get(enc["trigger_direction"],
                              EncoderTriggerDirection.Value_Both)))
        show_error(self._user_set.set_enum_value(
            EncoderTriggerSignalCountingMode.name,
            self._ENC_COUNT.get(enc["signal_counting_mode"],
                                EncoderTriggerSignalCountingMode.Value_Multiple_1)))
        show_error(self._user_set.set_int_value(
            EncoderTriggerInterval.name, enc["trigger_interval"]))
        show_error(self._user_set.set_int_value(
            ScanLineCount.name, enc["scan_line_count"]))

    def _apply_fixed_rate_mode(self, fr: dict) -> None:
        show_error(self._user_set.set_enum_value(
            DataAcquisitionTriggerSource.name,
            DataAcquisitionTriggerSource.Value_Software))
        show_error(self._user_set.set_enum_value(
            LineScanTriggerSource.name, LineScanTriggerSource.Value_FixedRate))
        show_error(self._user_set.set_float_value(
            SoftwareTriggerRate.name, fr["rate_hz"]))
        show_error(self._user_set.set_int_value(
            ScanLineCount.name, fr["scan_line_count"]))

    # ── Acquisition ──────────────────────────────────────────

    def acquire_point_cloud(self) -> np.ndarray:
        """
        Acquire one batch and return a NaN-free (N, 3) float32 array in mm.

        Points are in the custom coordinate system if a transformation has
        been configured in Mech-Eye Viewer; otherwise they are in the camera
        frame (a warning is logged).

        Returns an empty (0, 3) array on failure.
        """
        cb = _AcquisitionCallback(self._data_width)

        # Set a generous callback timeout (ms)
        show_error(self._user_set.set_int_value(CallbackRetrievalTimeout.name, 60_000))

        status = self._profiler.register_acquisition_callback(cb)
        if not status.is_ok():
            logger.error("Failed to register acquisition callback.")
            show_error(status)
            return np.empty((0, 3), dtype=np.float32)

        logger.info("Starting acquisition…")
        status = self._profiler.start_acquisition()
        if not status.is_ok():
            logger.error("start_acquisition() failed.")
            show_error(status)
            return np.empty((0, 3), dtype=np.float32)

        if self._sw_trigger:
            status = self._profiler.trigger_software()
            if not status.is_ok():
                logger.error("trigger_software() failed.")
                show_error(status)
                self._profiler.stop_acquisition()
                return np.empty((0, 3), dtype=np.float32)

        if not cb.wait(timeout=120.0):
            logger.error("Acquisition timed out after 120 s.")
            self._profiler.stop_acquisition()
            return np.empty((0, 3), dtype=np.float32)

        status = self._profiler.stop_acquisition()
        if not status.is_ok():
            show_error(status)

        if cb.profile_batch.check_flag(ProfileBatch.BatchFlag_Incomplete):
            logger.warning(
                "Batch is incomplete. Valid profiles: %d.",
                cb.profile_batch.valid_height(),
            )

        return self._batch_to_numpy(cb.profile_batch)

    # ── Internal: ProfileBatch → numpy ───────────────────────

    def _batch_to_numpy(self, batch: ProfileBatch) -> np.ndarray:
        """
        Convert ProfileBatch to a (N, 3) float32 numpy array (mm).

        Steps:
          1. get_untextured_point_cloud()  – SDK call, camera frame
          2. transform_point_cloud()       – rotate to custom frame (if set)
          3. save to temp PLY via SDK      – format-stable serialisation
          4. read_ply()                    – our utils reader → numpy
          5. remove NaN / Inf rows
        """
        s            = self._cfg["scanner"]
        mode         = s["trigger_mode"]
        use_encoder  = (mode == "encoder")
        trig_intv    = s["encoder"]["trigger_interval"] if use_encoder else 1

        err, x_res = self._user_set.get_float_value(XAxisResolution.name)
        show_error(err)
        err, y_res = self._user_set.get_float_value(YResolution.name)
        show_error(err)

        pc = batch.get_untextured_point_cloud(x_res, y_res, use_encoder, trig_intv)

        # Apply coordinate transformation (set via Mech-Eye Viewer → Custom Reference Frame)
        xform = get_transformation_params(self._profiler)
        if xform.__is__valid__():
            pc = transform_point_cloud(xform, pc)
        else:
            logger.warning(
                "No coordinate transformation is configured in the profiler. "
                "Point cloud will be in the camera reference frame. "
                "ROI filtering may not work correctly. "
                "Please configure a custom reference frame in Mech-Eye Viewer."
            )

        # Save to temp PLY, reload as numpy (robust across SDK versions)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".ply")
        os.close(tmp_fd)
        try:
            ProfileBatch.save_untextured_point_cloud_static(
                pc, FileFormat_PLY, tmp_path, False, CoordinateUnit_Millimeter
            )
            points = read_ply(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        # Remove NaN and Inf
        valid  = np.isfinite(points).all(axis=1)
        points = points[valid]
        logger.info(
            "Acquisition complete: %d valid points (NaN removed).", len(points)
        )
        return points
