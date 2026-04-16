"""
camera_manager.py — 多相机生命周期管理。

负责相机的枚举、打开、关闭、参数设置、取流控制和抽帧保存。
上层（UI 或测试）通过本模块提供的接口操作相机，无需直接接触 SDK。
"""

import logging
import sys
import os
from enum import Enum
from typing import Callable, Dict, List, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from CamOperation_class import CameraOperation
from hik_SDK.MvCameraControl_class import MvCamera, MV_CC_DEVICE_INFO_LIST
from hik_SDK.CameraParams_header import (
    MV_GIGE_DEVICE, MV_USB_DEVICE,
    MV_GENTL_GIGE_DEVICE, MV_GENTL_CAMERALINK_DEVICE,
    MV_GENTL_CXP_DEVICE, MV_GENTL_XOF_DEVICE,
    POINTER, MV_CC_DEVICE_INFO,
)
from ctypes import cast

logger = logging.getLogger(__name__)


class CameraState(Enum):
    IDLE     = "idle"       # 未枚举或已关闭
    READY    = "ready"      # 已枚举，可打开
    OPEN     = "open"       # 已打开，未取流
    GRABBING = "grabbing"   # 取流中


class CameraManager:
    """
    管理最多 `max_cameras` 台海康相机。
    所有相机参数从 cfg["cameras"] 读取。
    """

    def __init__(self, cfg: dict):
        # 海康 SDK 需要先初始化，否则后续枚举/打开可能无响应
        if not getattr(CameraManager, "_sdk_initialized", False):
            try:
                MvCamera.MV_CC_Initialize()
                CameraManager._sdk_initialized = True
            except Exception:
                CameraManager._sdk_initialized = False

        cam_cfg = cfg.get("cameras", {})
        inf_cfg = cfg.get("inference_2d", {})
        self._max_cams     = int(cam_cfg.get("count", 2))
        self._roi          = cam_cfg.get("roi", {})
        self._save_interval = int(cam_cfg.get("save_interval", 25))
        self._save_dir     = cam_cfg.get("save_dir", "data/2d_frames")
        self._inference_interval = int(inf_cfg.get("interval", 30))

        self._device_list  = MV_CC_DEVICE_INFO_LIST()
        self._operations:  List[Optional[CameraOperation]] = [None] * self._max_cams
        self._selected:    List[bool] = [False] * self._max_cams
        self._valid_count: int = 0

        # 帧回调：cam_id → callable(image)
        self._frame_callbacks: Dict[int, Callable] = {}

        self.state = CameraState.IDLE

        # 状态变更通知回调，供 UI 刷新按钮用
        self.on_state_changed: Optional[Callable[[CameraState], None]] = None
        # 设备信息通知：(cam_id, serial, model_name) → UI 更新按钮文字
        self.on_device_info:   Optional[Callable[[int, str, str], None]] = None
        # 日志通知
        self.on_log:           Optional[Callable[[str], None]] = None

    # ── 内部工具 ──────────────────────────────────────────

    def _log(self, msg: str) -> None:
        logger.info(msg)
        if self.on_log:
            self.on_log(f"[2D] {msg}")

    def _set_state(self, state: CameraState) -> None:
        self.state = state
        if self.on_state_changed:
            self.on_state_changed(state)

    @staticmethod
    def _parse_device_info(device_list, index):
        """从设备列表解析序列号和型号名称。"""
        from hik_SDK.MvCameraControl_class import MV_CC_DEVICE_INFO
        info = cast(device_list.pDeviceInfo[index], POINTER(MV_CC_DEVICE_INFO)).contents
        serial, model = "", ""

        def _decode_bytes(raw) -> str:
            if raw is None:
                return ""
            try:
                if isinstance(raw, str):
                    return raw
                # ctypes array / bytes-like
                b = bytes(bytearray(raw))
                b = b.split(b"\x00", 1)[0]
                for enc in ("gbk", "utf-8", "latin1"):
                    try:
                        return b.decode(enc)
                    except Exception:
                        continue
                return ""
            except Exception:
                return ""

        def _decode(c_arr):
            s = ""
            for c in c_arr:
                if c == 0:
                    break
                s += chr(c)
            return s

        if info.nTLayerType in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE):
            serial = _decode(info.SpecialInfo.stGigEInfo.chSerialNumber)
            model = _decode_bytes(info.SpecialInfo.stGigEInfo.chModelName)
        elif info.nTLayerType == MV_USB_DEVICE:
            serial = _decode(info.SpecialInfo.stUsb3VInfo.chSerialNumber)
            model = _decode_bytes(info.SpecialInfo.stUsb3VInfo.chModelName)

        return str(serial), str(model)

    # ── 公开接口 ──────────────────────────────────────────

    def enumerate(self) -> int:
        """枚举设备，返回找到的设备数量。触发 on_device_info 回调。"""
        n_layer = (
            MV_GIGE_DEVICE | MV_USB_DEVICE |
            MV_GENTL_GIGE_DEVICE | MV_GENTL_CAMERALINK_DEVICE |
            MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE
        )
        from hik_SDK.MvCameraControl_class import MvCamera as _MvCamera, SortMethod_SerialNumber
        ret = _MvCamera.MV_CC_EnumDevicesEx2(n_layer, self._device_list, '', SortMethod_SerialNumber)
        if ret != 0:
            self._log(f"枚举设备失败，错误码: 0x{ret:x}")
            return 0

        self._valid_count = min(self._max_cams, self._device_list.nDeviceNum)
        self._log(f"找到 {self._valid_count} 台相机")

        for i in range(self._valid_count):
            serial, model = self._parse_device_info(self._device_list, i)
            if self.on_device_info:
                self.on_device_info(i, serial, model)

        self._set_state(CameraState.READY)
        return self._valid_count

    def set_selected(self, cam_id: int, selected: bool) -> None:
        if 0 <= cam_id < self._max_cams:
            self._selected[cam_id] = selected

    def open(self) -> bool:
        """打开所有已选中的相机并设置 ROI。返回是否至少打开了一台。"""
        if not any(self._selected):
            self._log("未选择任何相机")
            return False

        opened = 0
        for i in range(self._max_cams):
            if not self._selected[i]:
                continue
            cam_obj = MvCamera()
            op = CameraOperation(cam_obj, self._device_list, i)
            if op.open_device() != 0:
                self._log(f"相机 {i} 打开失败")
                continue

            # 设置 ROI
            ret = op.set_roi_parameters(
                self._roi.get("width", 1240),
                self._roi.get("height", 1240),
                self._roi.get("offset_x", 0),
                self._roi.get("offset_y", 0),
            )
            if ret != 0:
                self._log(f"相机 {i} ROI 设置失败")

            self._operations[i] = op
            opened += 1
            self._log(f"相机 {i} 打开成功")

        if opened > 0:
            self._set_state(CameraState.OPEN)
        return opened > 0

    def close(self) -> None:
        """关闭所有已打开的相机。"""
        if self.state == CameraState.GRABBING:
            self.stop_grabbing()

        for i, op in enumerate(self._operations):
            if op is not None:
                op.close_device()
                self._operations[i] = None
                self._log(f"相机 {i} 已关闭")

        self._set_state(CameraState.IDLE)

    def set_frame_callback(self, cam_id: int, fn: Callable) -> None:
        """注册帧回调，取流时每帧调用 fn(image)。"""
        self._frame_callbacks[cam_id] = fn
        op = self._operations[cam_id]
        if op is not None:
            if hasattr(op, "set_inference_interval"):
                op.set_inference_interval(self._inference_interval)
            op.set_inference_callback(fn)

    def start_grabbing(self, display_handles: list) -> bool:
        """开始取流，display_handles 是各相机对应的窗口句柄列表。"""
        if self.state != CameraState.OPEN:
            return False

        for i, op in enumerate(self._operations):
            if op is None:
                continue
            handle = display_handles[i] if i < len(display_handles) else 0
            # 注册帧回调
            if i in self._frame_callbacks:
                if hasattr(op, "set_inference_interval"):
                    op.set_inference_interval(self._inference_interval)
                op.set_inference_callback(self._frame_callbacks[i])
            ret = op.start_grabbing(i, handle)
            if ret != 0:
                self._log(f"相机 {i} 开始取流失败，错误码: 0x{ret:x}")
            else:
                self._log(f"相机 {i} 开始取流")

        self._set_state(CameraState.GRABBING)
        return True

    def stop_grabbing(self) -> None:
        for i, op in enumerate(self._operations):
            if op is not None:
                op.stop_grabbing()
                self._log(f"相机 {i} 停止取流")
        self._set_state(CameraState.OPEN)

    def set_params(self, exposure: float, gain: float, fps: float) -> None:
        for i, op in enumerate(self._operations):
            if op is None:
                continue
            op.set_exposure_time(str(exposure))
            op.set_gain(str(gain))
            op.set_frame_rate(str(fps))
        self._log(f"参数已设置：曝光={exposure}, 增益={gain}, 帧率={fps}")

    def set_trigger_mode(self, mode: str) -> None:
        """mode: 'continuous' | 'trigger'"""
        for op in self._operations:
            if op is not None:
                op.set_trigger_mode(mode)

    def set_trigger_source(self, source: str) -> None:
        """source: 'software' | 'hardware'"""
        for op in self._operations:
            if op is not None:
                op.set_trigger_source(source)

    def trigger_once(self) -> None:
        for op in self._operations:
            if op is not None:
                op.trigger_once()

    def start_frame_saving(self) -> None:
        import os
        os.makedirs(self._save_dir, exist_ok=True)
        for i, op in enumerate(self._operations):
            if op is not None:
                op.start_frame_saving(self._save_interval)
                self._log(f"相机 {i} 开始抽帧保存，间隔 {self._save_interval} 帧")

    def stop_frame_saving(self) -> None:
        for i, op in enumerate(self._operations):
            if op is not None:
                op.stop_frame_saving()
                self._log(f"相机 {i} 停止保存，共 {op.saved_count} 张")

    def set_inference_interval(self, interval: int) -> None:
        self._inference_interval = int(interval) if int(interval) > 0 else 1
        for op in self._operations:
            if op is not None and hasattr(op, "set_inference_interval"):
                op.set_inference_interval(self._inference_interval)

    def get_opened_camera_ids(self) -> List[int]:
        return [i for i, op in enumerate(self._operations) if op is not None]

    @property
    def valid_count(self) -> int:
        return self._valid_count