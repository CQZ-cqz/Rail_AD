from pathlib import Path
from time import sleep

import cv2
import numpy as np

from mecheye.shared import *
from mecheye.profiler import *
from mecheye.profiler_utils import *

from utils.pointcloud import read_ply


class EncoderSafePointCloudExporter(object):
    """
    采集一批轮廓后导出两套点云，便于对比：
    1) raw_encoder: 使用编码器值展开 Y（可能出现大数精度量化）
    2) stable_index: 不使用编码器值展开 Y（与深度图行轴一致）
    """

    def __init__(self):
        self.profiler = Profiler()
        self.user_set = None
        self.profile_batch = None
        self.data_width = 0
        self.capture_line_count = 0
        self.is_software_trigger = True

    def set_encoder_trigger(self, trigger_direction: int, trigger_signal_counting_mode: int, trigger_interval: int):
        show_error(self.user_set.set_enum_value(
            LineScanTriggerSource.name, LineScanTriggerSource.Value_Encoder))
        show_error(self.user_set.set_enum_value(
            EncoderTriggerDirection.name, trigger_direction))
        show_error(self.user_set.set_enum_value(
            EncoderTriggerSignalCountingMode.name, trigger_signal_counting_mode))
        show_error(self.user_set.set_int_value(
            EncoderTriggerInterval.name, trigger_interval))

    def set_parameters(self):
        self.user_set = self.profiler.current_user_set()

        show_error(self.user_set.set_enum_value(
            ExposureMode.name, ExposureMode.Value_Timed))
        show_error(self.user_set.set_int_value(
            ExposureTime.name, 300))

        show_error(self.user_set.set_enum_value(
            DataAcquisitionTriggerSource.name, DataAcquisitionTriggerSource.Value_Software))

        # 编码器触发
        self.set_encoder_trigger(
            EncoderTriggerDirection.Value_ChannelBLeading,
            EncoderTriggerSignalCountingMode.Value_Multiple_1,
            1,
        )

        show_error(self.user_set.set_int_value(ScanLineCount.name, 2000))
        show_error(self.user_set.set_int_value(LaserPower.name, 100))
        show_error(self.user_set.set_enum_value(AnalogGain.name, AnalogGain.Value_Gain_4))
        show_error(self.user_set.set_int_value(DigitalGain.name, 0))
        show_error(self.user_set.set_int_value(MinGrayscaleValue.name, 50))
        show_error(self.user_set.set_int_value(MinLaserLineWidth.name, 2))
        show_error(self.user_set.set_int_value(MaxLaserLineWidth.name, 20))
        show_error(self.user_set.set_enum_value(SpotSelection.name, SpotSelection.Value_Strongest))
        show_error(self.user_set.set_int_value(MinSpotIntensity.name, 51))
        show_error(self.user_set.set_int_value(MaxSpotIntensity.name, 205))
        show_error(self.user_set.set_int_value(GapFilling.name, 16))
        show_error(self.user_set.set_enum_value(Filter.name, Filter.Value_Mean))
        show_error(self.user_set.set_enum_value(
            MeanFilterWindowSize.name, MeanFilterWindowSize.Value_WindowSize_2))

        error, self.data_width = self.user_set.get_int_value(DataPointsPerProfile.name)
        show_error(error)

        error, self.capture_line_count = self.user_set.get_int_value(ScanLineCount.name)
        show_error(error)

        error, data_acquisition_trigger_source = self.user_set.get_enum_value(
            DataAcquisitionTriggerSource.name)
        show_error(error)
        self.is_software_trigger = (
            data_acquisition_trigger_source == DataAcquisitionTriggerSource.Value_Software
        )

    def acquire_profile_data(self) -> bool:
        print("Start data acquisition.")
        status = self.profiler.start_acquisition()
        if not status.is_ok():
            show_error(status)
            return False

        if self.is_software_trigger:
            status = self.profiler.trigger_software()
            if not status.is_ok():
                show_error(status)
                return False

        self.profile_batch.clear()
        self.profile_batch.reserve(self.capture_line_count)

        while self.profile_batch.height() < self.capture_line_count:
            print("Acquired {} / {} lines".format(
                self.profile_batch.height(), self.capture_line_count))
            batch = ProfileBatch(self.data_width)
            status = self.profiler.retrieve_batch_data(batch)
            if status.is_ok():
                self.profile_batch.append(batch)
                sleep(0.1)
            else:
                show_error(status)
                return False

        print("Stop data acquisition.")
        status = self.profiler.stop_acquisition()
        if not status.is_ok():
            show_error(status)
        return status.is_ok()

    def save_depth_and_intensity(self, depth_path: Path, intensity_path: Path):
        cv2.imwrite(str(depth_path), self.profile_batch.get_depth_map().data())
        cv2.imwrite(str(intensity_path), self.profile_batch.get_intensity_image().data())

    @staticmethod
    def _save_ply_and_csv(point_cloud, ply_path: Path, csv_path: Path):
        ProfileBatch.save_untextured_point_cloud_static(
            point_cloud,
            FileFormat_PLY,
            str(ply_path),
            False,
            CoordinateUnit_Millimeter,
        )

        points = read_ply(str(ply_path)).astype(np.float64)
        np.savetxt(
            str(csv_path),
            points,
            delimiter=",",
            fmt="%.6f",
            header="X,Y,Z",
            comments="",
        )
        return points

    @staticmethod
    def _print_y_stats(name: str, points: np.ndarray):
        if points.size == 0:
            print(f"[{name}] empty point cloud")
            return

        finite_mask = np.isfinite(points).all(axis=1)
        valid = points[finite_mask]
        if valid.size == 0:
            print(f"[{name}] all points are non-finite")
            return

        y = valid[:, 1]
        unique_y = np.unique(np.round(y, 6))
        print(f"[{name}] valid_points={len(valid)}, unique_y={len(unique_y)}")
        if len(unique_y) > 0:
            sample = unique_y[: min(10, len(unique_y))]
            print(f"[{name}] y_sample={sample}")

    def export_compare_point_clouds(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)

        error, x_resolution = self.user_set.get_float_value(XAxisResolution.name)
        show_error(error)
        error, y_resolution = self.user_set.get_float_value(YResolution.name)
        show_error(error)
        error, trigger_interval = self.user_set.get_int_value(EncoderTriggerInterval.name)
        show_error(error)

        # A: 原始编码器Y
        raw_pc = self.profile_batch.get_untextured_point_cloud(
            x_resolution, y_resolution, True, trigger_interval
        )

        # B: 稳定行索引Y（与深度图行轴一致，不受编码器绝对值量级影响）
        stable_pc = self.profile_batch.get_untextured_point_cloud(
            x_resolution, y_resolution, False, trigger_interval
        )

        # 如果配置了自定义坐标系，统一变换后再保存
        xform = get_transformation_params(self.profiler)
        if xform.__is__valid__():
            raw_pc = transform_point_cloud(xform, raw_pc)
            stable_pc = transform_point_cloud(xform, stable_pc)
        else:
            print("[WARN] No custom frame set in Mech-Eye Viewer, saving in camera frame.")

        raw_ply = out_dir / "pointcloud_raw_encoder.ply"
        raw_csv = out_dir / "pointcloud_raw_encoder.csv"
        stable_ply = out_dir / "pointcloud_stable_index.ply"
        stable_csv = out_dir / "pointcloud_stable_index.csv"

        raw_points = self._save_ply_and_csv(raw_pc, raw_ply, raw_csv)
        stable_points = self._save_ply_and_csv(stable_pc, stable_ply, stable_csv)

        self._print_y_stats("raw_encoder", raw_points)
        self._print_y_stats("stable_index", stable_points)

        print("Saved:")
        print(f"  {raw_ply}")
        print(f"  {raw_csv}")
        print(f"  {stable_ply}")
        print(f"  {stable_csv}")

    def main(self):
        if not find_and_connect(self.profiler):
            return -1

        if not confirm_capture():
            return -1

        self.set_parameters()
        self.profile_batch = ProfileBatch(self.data_width)

        if not self.acquire_profile_data():
            self.profiler.disconnect()
            return -1

        if self.profile_batch.check_flag(ProfileBatch.BatchFlag_Incomplete):
            print(
                "Part of the batch data is lost, valid profiles:",
                self.profile_batch.valid_height(),
            )

        out_dir = Path("./output_encoder_compare")
        self.save_depth_and_intensity(out_dir / "depth.tiff", out_dir / "intensity.png")
        self.export_compare_point_clouds(out_dir)

        self.profiler.disconnect()
        print("Disconnected from the Mech-Eye Profiler successfully")
        return 0


if __name__ == "__main__":
    runner = EncoderSafePointCloudExporter()
    runner.main()
