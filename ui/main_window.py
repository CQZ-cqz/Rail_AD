from pathlib import Path
import sys
import importlib.util
import subprocess
import threading

import yaml
from PyQt5.QtCore import QDateTime, QEvent, QObject, pyqtSignal
from PyQt5.QtWidgets import QAbstractSpinBox, QFileDialog, QLabel, QMainWindow

from ui.Ui_MainWindow import Ui_MainWindow

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TWO_D_ROOT = PROJECT_ROOT 
if str(TWO_D_ROOT) not in sys.path:
    sys.path.insert(0, str(TWO_D_ROOT))

def _load_symbol(module_file: Path, module_name: str, symbol_name: str):
    if not module_file.exists():
        return None
    spec = importlib.util.spec_from_file_location(module_name, str(module_file))
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, symbol_name, None)


def _load_module(module_file: Path, module_name: str):
    if not module_file.exists():
        return None
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(module_name, str(module_file))
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_camera_manager_module = _load_module(TWO_D_ROOT / "core" / "camera_manger.py", "two_d_camera_manager")
_inference_module = _load_module(TWO_D_ROOT / "core" / "inference_manager.py", "two_d_inference_manager")
_detector_module = _load_module(TWO_D_ROOT / "algorithms" / "detection_RD4AD.py", "two_d_detector")

CameraManager = getattr(_camera_manager_module, "CameraManager", None) if _camera_manager_module else None
CameraState = getattr(_camera_manager_module, "CameraState", None) if _camera_manager_module else None
InferenceManager = getattr(_inference_module, "InferenceManager", None) if _inference_module else None
RD4ADDetector = getattr(_detector_module, "RD4ADDetector", None) if _detector_module else None


class _NoWheelFilter(QObject):
    def eventFilter(self, obj, event):
        if isinstance(obj, QAbstractSpinBox) and event.type() == QEvent.Wheel:
            return True
        return super().eventFilter(obj, event)


class MainWindow(QMainWindow):
    log_signal = pyqtSignal(str)
    camera_state_signal = pyqtSignal(object)
    device_info_signal = pyqtSignal(int, str, str)
    inference_result_signal = pyqtSignal(int, object)
    three_d_status_signal = pyqtSignal(str)

    def __init__(self, cfg=None, config_path: str = "config.yaml", parent=None):
        super().__init__(parent)
        self.cfg = cfg or {}
        self.config_path = config_path
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.camera_manager = None
        self.inference_manager = None
        self.detector = None
        self._spin_no_wheel_filter = _NoWheelFilter(self)
        self._three_d_thread = None
        self._three_d_thread_lock = threading.Lock()
        self._three_d_stop_event = threading.Event()
        self._three_d_proc = None
        self._three_d_proc_lock = threading.Lock()

        self._setup_status_bar()
        self._connect_threadsafe_signals()
        self._connect_signals()
        self._init_ui_values()
        self._backfill_from_cfg()
        self._init_2d_backend()

    def _connect_threadsafe_signals(self):
        self.log_signal.connect(self._append_log)
        self.camera_state_signal.connect(self._on_camera_state_changed)
        self.device_info_signal.connect(self._on_device_info)
        self.inference_result_signal.connect(self._on_inference_result)
        self.three_d_status_signal.connect(self._on_3d_status_changed)

    def _setup_status_bar(self):
        self.label_status_3d = QLabel("3D: 就绪")
        self.label_status_anomaly = QLabel("异常: 0 段")
        self.ui.statusbar.addPermanentWidget(self.label_status_3d)
        self.ui.statusbar.addPermanentWidget(self.label_status_anomaly)

    def _connect_signals(self):
        # 2D 相机采集（左侧）
        self.ui.btn_enum.clicked.connect(self.on_enum_devices)
        self.ui.btn_open.clicked.connect(self.on_open_devices)
        self.ui.btn_close.clicked.connect(self.on_close_devices)
        self.ui.btn_stream.toggled.connect(self.on_toggle_stream)
        self.ui.btn_set_params.clicked.connect(self.on_set_camera_params)

        self.ui.radio_continuous.toggled.connect(self.on_trigger_mode_changed)
        self.ui.radio_trigger.toggled.connect(self.on_trigger_mode_changed)
        self.ui.check_soft_trigger.toggled.connect(self.on_soft_trigger_changed)
        self.ui.btn_soft_once.clicked.connect(self.on_soft_trigger_once)

        # 2D 参数动作
        self.ui.btn_2d_apply.clicked.connect(self.on_apply_2d_params)
        self.ui.btn_2d_save_file.clicked.connect(self.on_save_2d_params_to_file)
        self.ui.check_inference_enabled.toggled.connect(lambda _checked: self._refresh_2d_ui_state())
        self.ui.btn_browse_save_dir.clicked.connect(self.on_browse_save_dir)
        self.ui.btn_browse_checkpoint.clicked.connect(self.on_browse_checkpoint)
        self.ui.btn_browse_defect_dir.clicked.connect(self.on_browse_defect_dir)
        self.ui.btn_browse_cad_stl.clicked.connect(self.on_browse_cad_stl)

        # 右侧 2D 运行控制（仅两个按钮）
        self.ui.btn_2d_save.toggled.connect(self.on_toggle_frame_saving)
        self.ui.btn_2d_detect.toggled.connect(self.on_toggle_2d_detection)

        # 3D 运行控制
        self.ui.btn_3d_execute.clicked.connect(self.on_execute_3d)
        self.ui.btn_3d_stop.clicked.connect(self.on_stop_3d)
        self.ui.btn_3d_apply.clicked.connect(self.on_apply_3d_params)
        self.ui.btn_3d_save.clicked.connect(self.on_save_3d_params_to_file)

        # 菜单
        self.ui.action_save_config.triggered.connect(self.on_save_all_config)
        self.ui.action_exit.triggered.connect(self.close)
        self.ui.action_about.triggered.connect(self.on_about)

    def _init_ui_values(self):
        # 改为“直接输入”样式：隐藏上下箭头并禁用滚轮修改
        for spin in self.findChildren(QAbstractSpinBox):
            spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
            spin.setKeyboardTracking(False)
            spin.installEventFilter(self._spin_no_wheel_filter)

    def _init_2d_backend(self):
        if CameraManager is None:
            self._log("2D 后端未加载：请检查 2D_AD 依赖和导入路径")
            return

        self.camera_manager = CameraManager(self.cfg)
        self.camera_manager.on_log = lambda msg: self.log_signal.emit(msg)
        self.camera_manager.on_state_changed = lambda state: self.camera_state_signal.emit(state)
        self.camera_manager.on_device_info = lambda cam_id, serial, model: self.device_info_signal.emit(cam_id, serial, model)

        # 推理后端改为按需初始化，避免启动时权重路径缺失导致按钮长期不可用。
        self._try_init_inference_backend()

        self._refresh_2d_ui_state()

    def _try_init_inference_backend(self) -> bool:
        ckpt = self.ui.edit_checkpoint.text().strip()
        if not ckpt:
            self.detector = None
            self.inference_manager = None
            return False

        if RD4ADDetector is None or InferenceManager is None:
            self._log("2D 推理后端未加载：请检查 algorithms/core 路径与依赖")
            self.detector = None
            self.inference_manager = None
            return False

        try:
            self.detector = RD4ADDetector(
                ckpt,
                anomaly_threshold=float(self.ui.spin_anomaly_threshold.value()),
                min_defect_area=int(self.ui.spin_min_defect_area.value()),
            )
        except Exception as exc:
            self._log(f"检测器初始化失败：{exc}")
            self.detector = None
            self.inference_manager = None
            return False

        self.inference_manager = InferenceManager(self.cfg, self.detector)
        self.inference_manager.on_log = lambda msg: self.log_signal.emit(msg)
        self.inference_manager.set_result_callback(
            lambda cam_id, result: self.inference_result_signal.emit(cam_id, result)
        )
        return True

    def _refresh_2d_ui_state(self):
        if self.camera_manager is None or CameraState is None:
            return
        state = self.camera_manager.state
        is_ready = state in (CameraState.READY, CameraState.OPEN, CameraState.GRABBING)
        is_open = state in (CameraState.OPEN, CameraState.GRABBING)
        is_grabbing = state == CameraState.GRABBING

        self.ui.btn_open.setEnabled(is_ready)
        self.ui.btn_close.setEnabled(is_open)
        self.ui.btn_stream.setEnabled(is_open)
        self.ui.btn_set_params.setEnabled(is_open)
        self.ui.radio_continuous.setEnabled(is_open)
        self.ui.radio_trigger.setEnabled(is_open)
        self.ui.check_soft_trigger.setEnabled(is_open)
        self.ui.btn_soft_once.setEnabled(is_open and self.ui.radio_trigger.isChecked())
        self.ui.btn_2d_save.setEnabled(is_grabbing)
        self.ui.btn_2d_detect.setEnabled(is_grabbing and self.ui.check_inference_enabled.isChecked())

    def _on_camera_state_changed(self, _state):
        self._refresh_2d_ui_state()

    def _on_device_info(self, cam_id: int, serial: str, model_name: str):
        text = f"相机{cam_id + 1}"
        if serial or model_name:
            text = f"({serial}) {model_name}".strip()
        if cam_id == 0:
            self.ui.checkBox_cam1.setText(text)
            self.ui.checkBox_cam1.setEnabled(True)
        elif cam_id == 1:
            self.ui.checkBox_cam2.setText(text)
            self.ui.checkBox_cam2.setEnabled(True)

    def _on_inference_result(self, cam_id: int, result: dict):
        if result.get("has_defect", False):
            score = result.get("anomaly_score", 0.0)
            self._log(f"相机 {cam_id} 检测到缺陷，分数: {score:.4f}")

    def _on_3d_status_changed(self, text: str):
        self.label_status_3d.setText(text)

    def _backfill_from_cfg(self):
        cameras = self.cfg.get("cameras", {})
        inference = self.cfg.get("inference_2d", {})
        paths = self.cfg.get("paths", {})
        detection = self.cfg.get("detection", {})

        if cameras:
            self.ui.spin_cam_count.setValue(int(cameras.get("count", self.ui.spin_cam_count.value())))
            self.ui.spin_default_exposure.setValue(float(cameras.get("default_exposure_us", self.ui.spin_default_exposure.value())))
            self.ui.spin_default_gain.setValue(float(cameras.get("default_gain", self.ui.spin_default_gain.value())))
            self.ui.spin_default_fps.setValue(float(cameras.get("default_fps", self.ui.spin_default_fps.value())))
            self.ui.spin_save_interval.setValue(int(cameras.get("save_interval", self.ui.spin_save_interval.value())))
            self.ui.edit_save_dir.setText(str(cameras.get("save_dir", self.ui.edit_save_dir.text())))

        if inference:
            self.ui.check_inference_enabled.setChecked(bool(inference.get("enabled", self.ui.check_inference_enabled.isChecked())))
            self.ui.spin_inference_interval.setValue(int(inference.get("interval", self.ui.spin_inference_interval.value())))
            self.ui.edit_checkpoint.setText(str(inference.get("checkpoint_path", self.ui.edit_checkpoint.text())))
            self.ui.edit_defect_dir.setText(str(inference.get("defect_save_dir", self.ui.edit_defect_dir.text())))
            self.ui.spin_anomaly_threshold.setValue(float(inference.get("anomaly_threshold", self.ui.spin_anomaly_threshold.value())))
            self.ui.spin_min_defect_area.setValue(int(inference.get("min_defect_area", self.ui.spin_min_defect_area.value())))

        cad_path = detection.get("cad_stl_path") or paths.get("cad_stl")
        if cad_path:
            self.ui.edit_cad_stl.setText(str(cad_path))

        scanner = self.cfg.get("scanner", {})
        if scanner:
            trigger_mode = str(scanner.get("trigger_mode", "fixed_rate"))
            idx = self.ui.combo_trigger_mode.findText(trigger_mode)
            if idx >= 0:
                self.ui.combo_trigger_mode.setCurrentIndex(idx)

            enc = scanner.get("encoder", {})
            fr = scanner.get("fixed_rate", {})
            self.ui.spin_enc_interval.setValue(int(enc.get("trigger_interval", self.ui.spin_enc_interval.value())))
            shared_scan_lines = int(enc.get("scan_line_count", fr.get("scan_line_count", self.ui.spin_enc_scan_lines.value())))
            self.ui.spin_enc_scan_lines.setValue(shared_scan_lines)
            self.ui.spin_fr_rate_hz.setValue(float(fr.get("rate_hz", self.ui.spin_fr_rate_hz.value())))

            enc_dir = str(enc.get("trigger_direction", self.ui.combo_enc_direction.currentText()))
            idx = self.ui.combo_enc_direction.findText(enc_dir)
            if idx >= 0:
                self.ui.combo_enc_direction.setCurrentIndex(idx)

            enc_counting = str(enc.get("signal_counting_mode", self.ui.combo_enc_counting.currentText()))
            idx = self.ui.combo_enc_counting.findText(enc_counting)
            if idx >= 0:
                self.ui.combo_enc_counting.setCurrentIndex(idx)

            exposure = scanner.get("exposure", {})
            if exposure:
                self.ui.spin_exp_time.setValue(int(exposure.get("exposure_time_us", self.ui.spin_exp_time.value())))

            self.ui.combo_analog_gain.setCurrentText(str(scanner.get("analog_gain", self.ui.combo_analog_gain.currentText())))
            self.ui.spin_digital_gain.setValue(int(scanner.get("digital_gain", self.ui.spin_digital_gain.value())))
            self.ui.spin_min_gray.setValue(int(scanner.get("min_grayscale", self.ui.spin_min_gray.value())))
            self.ui.spin_min_line_width.setValue(int(scanner.get("min_laser_line_width", self.ui.spin_min_line_width.value())))
            self.ui.spin_max_line_width.setValue(int(scanner.get("max_laser_line_width", self.ui.spin_max_line_width.value())))

            self.ui.spin_det_threshold.setValue(float(detection.get("threshold_mm", self.ui.spin_det_threshold.value())))
            self.ui.spin_voxel_size.setValue(float(detection.get("voxel_size", self.ui.spin_voxel_size.value())))
            self.ui.check_use_registration.setChecked(bool(detection.get("use_registration", self.ui.check_use_registration.isChecked())))
            self.ui.check_reg_fallback.setChecked(bool(detection.get("registration_fallback_to_quick", self.ui.check_reg_fallback.isChecked())))

    def _choose_directory(self, title: str, current_text: str = "") -> str:
        start_dir = current_text or str(Path.cwd())
        chosen = QFileDialog.getExistingDirectory(self, title, start_dir)
        return chosen or ""

    def _choose_file(self, title: str, current_text: str = "", file_filter: str = "All Files (*)") -> str:
        start_file = current_text or str(Path.cwd())
        chosen, _ = QFileDialog.getOpenFileName(self, title, start_file, file_filter)
        return chosen or ""

    def _append_log(self, message: str):
        timestamp = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")
        self.ui.textEdit_log.append(f"[{timestamp}] {message}")

    def _log(self, message: str):
        self.log_signal.emit(message)

    # ---------- 2D 相机采集 ----------
    def on_enum_devices(self):
        if self.camera_manager is None:
            self._log("相机管理器未初始化")
            return
        count = self.camera_manager.enumerate()
        self._log(f"枚举完成，数量: {count}")
        self._refresh_2d_ui_state()

    def on_open_devices(self):
        if self.camera_manager is None:
            self._log("相机管理器未初始化")
            return
        selected = [self.ui.checkBox_cam1.isChecked(), self.ui.checkBox_cam2.isChecked()]
        if not any(selected):
            selected = [self.ui.checkBox_cam1.isEnabled(), self.ui.checkBox_cam2.isEnabled()]
            self.ui.checkBox_cam1.setChecked(selected[0])
            self.ui.checkBox_cam2.setChecked(selected[1])
        for i, s in enumerate(selected):
            self.camera_manager.set_selected(i, s)
        ok = self.camera_manager.open()
        self._log("打开设备成功" if ok else "打开设备失败")
        self._refresh_2d_ui_state()

    def on_close_devices(self):
        if self.inference_manager and self.inference_manager.is_running:
            self.inference_manager.stop()
        if self.camera_manager is not None:
            self.camera_manager.close()
        self.ui.btn_stream.setChecked(False)
        self.ui.btn_2d_save.setChecked(False)
        self.ui.btn_2d_detect.setChecked(False)
        self._log("关闭设备")
        self._refresh_2d_ui_state()

    def on_toggle_stream(self, checked: bool):
        if self.camera_manager is None:
            self.ui.btn_stream.setChecked(False)
            return
        self.ui.btn_stream.setText("停止取流" if checked else "开始取流")
        if checked:
            handles = [int(self.ui.cameraView_1.winId()), int(self.ui.cameraView_2.winId())]
            ok = self.camera_manager.start_grabbing(handles)
            if not ok:
                self.ui.btn_stream.setChecked(False)
                self._log("开始取流失败")
                return
            self._log("开始取流")
        else:
            self.camera_manager.stop_grabbing()
            self._log("停止取流")
            if self.ui.btn_2d_save.isChecked():
                self.ui.btn_2d_save.setChecked(False)
            if self.ui.btn_2d_detect.isChecked():
                self.ui.btn_2d_detect.setChecked(False)
        self._refresh_2d_ui_state()

    def on_set_camera_params(self):
        exposure = self.ui.spin_default_exposure.value()
        gain = self.ui.spin_default_gain.value()
        fps = self.ui.spin_default_fps.value()
        if self.camera_manager is not None:
            self.camera_manager.set_params(exposure, gain, fps)
        self._log(f"设置参数: exposure={exposure}, gain={gain}, fps={fps}")

    def on_trigger_mode_changed(self):
        mode = "连续模式" if self.ui.radio_continuous.isChecked() else "触发模式"
        if self.camera_manager is not None:
            sdk_mode = "continuous" if self.ui.radio_continuous.isChecked() else "triggermode"
            self.camera_manager.set_trigger_mode(sdk_mode)
        self._log(f"切换触发模式: {mode}")
        self._refresh_2d_ui_state()

    def on_soft_trigger_changed(self, checked: bool):
        if self.camera_manager is not None:
            self.camera_manager.set_trigger_source("software" if checked else "hardware")
        self._log("软件触发已启用" if checked else "软件触发已关闭")

    def on_soft_trigger_once(self):
        if self.camera_manager is not None:
            self.camera_manager.trigger_once()
        self._log("触发一次")

    # ---------- 2D 参数 ----------
    def on_apply_2d_params(self):
        interval = int(self.ui.spin_inference_interval.value())
        if self.camera_manager is not None and hasattr(self.camera_manager, "set_inference_interval"):
            self.camera_manager.set_inference_interval(interval)
        if self.inference_manager is not None:
            self.inference_manager.update_params(
                interval=1,
                save_dir=self.ui.edit_defect_dir.text().strip(),
                anomaly_threshold=float(self.ui.spin_anomaly_threshold.value()),
                min_defect_area=int(self.ui.spin_min_defect_area.value()),
            )
        self._log(f"应用 2D 参数: 采样间隔={interval}")

    def on_apply_3d_params(self):
        self._log("应用 3D 参数")

    def on_save_2d_params_to_file(self):
        self.cfg.setdefault("cameras", {})
        self.cfg.setdefault("inference_2d", {})

        self.cfg["cameras"]["count"] = int(self.ui.spin_cam_count.value())
        self.cfg["cameras"]["default_exposure_us"] = float(self.ui.spin_default_exposure.value())
        self.cfg["cameras"]["default_gain"] = float(self.ui.spin_default_gain.value())
        self.cfg["cameras"]["default_fps"] = float(self.ui.spin_default_fps.value())
        self.cfg["cameras"]["save_interval"] = int(self.ui.spin_save_interval.value())
        self.cfg["cameras"]["save_dir"] = self.ui.edit_save_dir.text().strip()

        self.cfg["inference_2d"]["enabled"] = bool(self.ui.check_inference_enabled.isChecked())
        self.cfg["inference_2d"]["interval"] = int(self.ui.spin_inference_interval.value())
        self.cfg["inference_2d"]["checkpoint_path"] = self.ui.edit_checkpoint.text().strip()
        self.cfg["inference_2d"]["defect_save_dir"] = self.ui.edit_defect_dir.text().strip()
        self.cfg["inference_2d"]["anomaly_threshold"] = float(self.ui.spin_anomaly_threshold.value())
        self.cfg["inference_2d"]["min_defect_area"] = int(self.ui.spin_min_defect_area.value())

        if self.inference_manager is not None:
            if self.camera_manager is not None and hasattr(self.camera_manager, "set_inference_interval"):
                self.camera_manager.set_inference_interval(self.cfg["inference_2d"]["interval"])
            self.inference_manager.update_params(
                interval=1,
                save_dir=self.cfg["inference_2d"]["defect_save_dir"],
                anomaly_threshold=self.cfg["inference_2d"]["anomaly_threshold"],
                min_defect_area=self.cfg["inference_2d"]["min_defect_area"],
            )

        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.cfg, f, allow_unicode=True, sort_keys=False)
        self._log(f"保存 2D 参数到文件: {self.config_path}")

    def on_browse_save_dir(self):
        chosen = self._choose_directory("选择 2D 保存目录", self.ui.edit_save_dir.text())
        if chosen:
            self.ui.edit_save_dir.setText(chosen)
            self._log(f"设置保存目录: {chosen}")

    def on_browse_checkpoint(self):
        chosen = self._choose_file(
            "选择模型权重文件",
            self.ui.edit_checkpoint.text(),
            "Model Files (*.pth *.pt);;All Files (*)",
        )
        if chosen:
            self.ui.edit_checkpoint.setText(chosen)
            self._log(f"设置模型权重路径: {chosen}")

    def on_browse_defect_dir(self):
        chosen = self._choose_directory("选择缺陷保存目录", self.ui.edit_defect_dir.text())
        if chosen:
            self.ui.edit_defect_dir.setText(chosen)
            self._log(f"设置缺陷保存目录: {chosen}")

    def on_browse_cad_stl(self):
        chosen = self._choose_file(
            "选择 CAD STL 文件",
            self.ui.edit_cad_stl.text(),
            "STL Files (*.stl *.STL);;All Files (*)",
        )
        if chosen:
            self.ui.edit_cad_stl.setText(chosen)
            self._log(f"设置 CAD STL 路径: {chosen}")

    def on_save_3d_params_to_file(self):
        self.cfg.setdefault("scanner", {})
        self.cfg["scanner"].setdefault("encoder", {})
        self.cfg["scanner"].setdefault("fixed_rate", {})
        self.cfg["scanner"].setdefault("exposure", {})
        self.cfg.setdefault("detection", {})
        self.cfg.setdefault("paths", {})

        self.cfg["scanner"]["trigger_mode"] = self.ui.combo_trigger_mode.currentText()
        self.cfg["scanner"]["encoder"]["trigger_direction"] = self.ui.combo_enc_direction.currentText()
        self.cfg["scanner"]["encoder"]["signal_counting_mode"] = self.ui.combo_enc_counting.currentText()
        self.cfg["scanner"]["encoder"]["trigger_interval"] = int(self.ui.spin_enc_interval.value())

        shared_scan_lines = int(self.ui.spin_enc_scan_lines.value())
        self.cfg["scanner"]["encoder"]["scan_line_count"] = shared_scan_lines
        self.cfg["scanner"]["fixed_rate"]["scan_line_count"] = shared_scan_lines
        self.cfg["scanner"]["fixed_rate"]["rate_hz"] = float(self.ui.spin_fr_rate_hz.value())

        self.cfg["scanner"]["exposure"]["exposure_time_us"] = int(self.ui.spin_exp_time.value())
        self.cfg["scanner"]["analog_gain"] = self.ui.combo_analog_gain.currentText()
        self.cfg["scanner"]["digital_gain"] = int(self.ui.spin_digital_gain.value())
        self.cfg["scanner"]["min_grayscale"] = int(self.ui.spin_min_gray.value())
        self.cfg["scanner"]["min_laser_line_width"] = int(self.ui.spin_min_line_width.value())
        self.cfg["scanner"]["max_laser_line_width"] = int(self.ui.spin_max_line_width.value())

        self.cfg["detection"]["threshold_mm"] = float(self.ui.spin_det_threshold.value())
        self.cfg["detection"]["voxel_size"] = float(self.ui.spin_voxel_size.value())
        self.cfg["detection"]["use_registration"] = bool(self.ui.check_use_registration.isChecked())
        self.cfg["detection"]["registration_fallback_to_quick"] = bool(self.ui.check_reg_fallback.isChecked())

        cad_path = self.ui.edit_cad_stl.text().strip()
        self.cfg["paths"]["cad_stl"] = cad_path
        self.cfg["detection"]["cad_stl_path"] = cad_path

        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.cfg, f, allow_unicode=True, sort_keys=False)
        self._log(f"保存 3D 参数到文件: {self.config_path}")

    # ---------- 2D 运行控制 ----------
    def on_toggle_frame_saving(self, checked: bool):
        self.ui.btn_2d_save.setText("停止抽帧保存" if checked else "开始抽帧保存")
        if self.camera_manager is not None:
            if checked:
                self.camera_manager.start_frame_saving()
            else:
                self.camera_manager.stop_frame_saving()
        self._log("开始抽帧保存" if checked else "停止抽帧保存")

    def on_toggle_2d_detection(self, checked: bool):
        self.ui.btn_2d_detect.setText("停止2D检测" if checked else "开始2D检测")
        if checked and not self.ui.check_inference_enabled.isChecked():
            self._log("2D 推理未启用，请先勾选“启用2D推理”")
            self.ui.btn_2d_detect.setChecked(False)
            return

        if checked and self.inference_manager is None:
            if not self._try_init_inference_backend():
                self._log("推理管理器未初始化，无法启动检测")
                self.ui.btn_2d_detect.setChecked(False)
                self._refresh_2d_ui_state()
                return

        if checked:
            opened_cam_ids = []
            if self.camera_manager is not None and hasattr(self.camera_manager, "get_opened_camera_ids"):
                opened_cam_ids = self.camera_manager.get_opened_camera_ids()
            if not opened_cam_ids:
                opened_cam_ids = [0, 1]

            # 前移门控：在相机线程侧按 interval 抽帧，减少每帧转换/回调开销
            inference_interval = int(self.ui.spin_inference_interval.value())
            if self.camera_manager is not None and hasattr(self.camera_manager, "set_inference_interval"):
                self.camera_manager.set_inference_interval(inference_interval)
            # 推理线程侧仅消费已抽样帧，避免双重间隔导致抽样过稀
            self.inference_manager.update_params(interval=1)

            self.inference_manager.set_active_cameras(opened_cam_ids)
            for cam_id in opened_cam_ids:
                self.camera_manager.set_frame_callback(cam_id, lambda img, cid=cam_id: self.inference_manager.on_frame(cid, img))
            self.inference_manager.start()
            self._log(f"2D检测已绑定相机: {opened_cam_ids}, 采样间隔: {inference_interval}")
        else:
            self.inference_manager.stop()
        self._log("开始2D检测" if checked else "停止2D检测")

    def closeEvent(self, event):
        try:
            self._stop_3d_task()
            if self.inference_manager and self.inference_manager.is_running:
                self.inference_manager.stop()
            if self.camera_manager is not None:
                self.camera_manager.close()
        finally:
            super().closeEvent(event)

    # ---------- 3D 运行控制 ----------
    def _parse_3d_mode(self) -> str:
        text = self.ui.combo_3d_mode.currentText().lower()
        if "(scan)" in text:
            return "scan"
        if "(detect)" in text:
            return "detect"
        if "(reset)" in text:
            return "reset"
        if "(auto)" in text:
            return "auto"
        return "scan"

    def _is_3d_running(self) -> bool:
        with self._three_d_thread_lock:
            t = self._three_d_thread
        return t is not None and t.is_alive()

    def _terminate_process_tree(self, proc: subprocess.Popen) -> None:
        if proc is None:
            return
        try:
            if sys.platform.startswith("win"):
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                proc.terminate()
        except Exception:
            pass

    def _stop_3d_task(self):
        self._three_d_stop_event.set()
        with self._three_d_proc_lock:
            proc = self._three_d_proc
        if proc is not None:
            self._terminate_process_tree(proc)

        with self._three_d_thread_lock:
            t = self._three_d_thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)

    def _run_3d_commands_worker(self, commands):
        py_exe = sys.executable
        main_py = PROJECT_ROOT / "main.py"
        cwd = str(PROJECT_ROOT)

        try:
            for command in commands:
                if self._three_d_stop_event.is_set():
                    self._log("3D 任务已收到停止信号")
                    break

                self.three_d_status_signal.emit(f"3D: 运行中 ({command})")
                self._log(f"启动 3D 命令: {command}")

                proc = subprocess.Popen(
                    [py_exe, str(main_py), command, "--config", self.config_path],
                    cwd=cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                with self._three_d_proc_lock:
                    self._three_d_proc = proc

                if proc.stdout is not None:
                    for line in proc.stdout:
                        if self._three_d_stop_event.is_set():
                            self._terminate_process_tree(proc)
                            break
                        line = line.rstrip("\r\n")
                        if line:
                            self._log(f"[3D/{command}] {line}")

                code = proc.wait()
                with self._three_d_proc_lock:
                    if self._three_d_proc is proc:
                        self._three_d_proc = None

                if code != 0:
                    self._log(f"3D 命令失败: {command}, exit_code={code}")
                    break
                self._log(f"3D 命令完成: {command}")
        except Exception as exc:
            self._log(f"3D 任务异常: {exc}")
        finally:
            self.three_d_status_signal.emit("3D: 就绪")
            self._three_d_stop_event.clear()
            with self._three_d_proc_lock:
                self._three_d_proc = None
            with self._three_d_thread_lock:
                self._three_d_thread = None

    def on_execute_3d(self):
        if self._is_3d_running():
            self._log("已有 3D 任务在运行，请先停止当前任务")
            return

        # 运行前将 UI 参数落盘，确保 main.py 读取到最新 3D 配置
        self.on_save_3d_params_to_file()

        mode = self._parse_3d_mode()
        if mode == "auto":
            commands = ["scan", "detect", "status"]
        else:
            commands = [mode]

        self._three_d_stop_event.clear()
        t = threading.Thread(
            target=self._run_3d_commands_worker,
            args=(commands,),
            daemon=True,
            name="ThreeDRunner",
        )
        with self._three_d_thread_lock:
            self._three_d_thread = t
        t.start()
        self._log(f"执行 3D 任务模式: {mode}, 命令序列: {commands}")

    def on_stop_3d(self):
        self._stop_3d_task()
        self._log("停止 3D 任务")

    # ---------- 菜单 ----------
    def on_save_all_config(self):
        self.on_save_2d_params_to_file()
        self.on_save_3d_params_to_file()

    def on_about(self):
        self._log("关于")
