"""
inference_manager.py — 推理线程与帧队列管理。

职责：
  - 维护每台相机的帧队列
  - 按配置的抽帧间隔控制入队频率
  - 在后台线程中批量推理，结果通过回调通知上层
  - 与相机硬件和 UI 完全解耦
"""

import logging
import queue
import threading
import time
import traceback
from typing import Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class InferenceManager:
    """
    管理推理线程和帧队列。

    使用方式：
      mgr = InferenceManager(cfg, detector)
      mgr.set_result_callback(fn)   # fn(cam_id, result) 在推理完成后调用
      mgr.start()
      # 由相机帧回调调用：
      mgr.on_frame(cam_id, image)
      mgr.stop()
    """

    def __init__(self, cfg: dict, detector):
        inf_cfg = cfg.get("inference_2d", {})
        self._interval    = int(inf_cfg.get("interval", 30))
        self._save_dir    = inf_cfg.get("defect_save_dir", "data/defect_images")
        self._anomaly_threshold = float(inf_cfg.get("anomaly_threshold", 0.006))
        self._min_defect_area = int(inf_cfg.get("min_defect_area", 10))
        self._n_cams      = int(cfg.get("cameras", {}).get("count", 2))
        self._detector    = detector
        if hasattr(self._detector, "set_postprocess_params"):
            self._detector.set_postprocess_params(
                anomaly_threshold=self._anomaly_threshold,
                min_defect_area=self._min_defect_area,
            )

        # 每台相机独立队列，最多缓存 10 帧
        self._queues: List[queue.Queue] = [
            queue.Queue(maxsize=10) for _ in range(self._n_cams)
        ]
        self._frame_counters = [0] * self._n_cams
        self._active_cams = list(range(self._n_cams))

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._exit_event = threading.Event()
        self._last_heartbeat_ts = time.monotonic()

        # 运行时诊断指标
        self._frames_received = 0
        self._frames_enqueued = 0
        self._frames_dropped = 0
        self._detect_calls = 0
        self._detect_errors = 0
        self._last_detect_ms = 0.0

        # 结果回调：fn(cam_id, result_dict)
        self._result_cb: Optional[Callable] = None
        # 日志回调
        self.on_log: Optional[Callable[[str], None]] = None

    # ── 内部工具 ──────────────────────────────────────────

    def _log(self, msg: str) -> None:
        logger.info(msg)
        if self.on_log:
            self.on_log(f"[2D] {msg}")

    def _worker(self) -> None:
        """推理线程主循环，等待两台相机都有帧时批量推理。"""
        self._log("推理线程已启动")
        while self._running and not self._exit_event.is_set():
            images, cam_ids = [], []

            for cam_id in self._active_cams:
                if not self._queues[cam_id].empty():
                    try:
                        frame_data = self._queues[cam_id].get_nowait()
                        if frame_data["image"] is not None and frame_data["image"].size > 0:
                            images.append(frame_data["image"])
                            cam_ids.append(cam_id)
                        self._queues[cam_id].task_done()
                    except queue.Empty:
                        continue

            # 只要有可用帧就推理，避免单相机或某一路暂时无帧导致整体停滞
            if len(images) == 0:
                time.sleep(0.05)
                continue

            try:
                t0 = time.perf_counter()
                results = self._detector.detect_batch(
                    images=images,
                    save_dir=self._save_dir,
                    camera_ids=cam_ids,
                )
                self._detect_calls += 1
                self._last_detect_ms = (time.perf_counter() - t0) * 1000.0
                for i, result in enumerate(results):
                    if result["has_defect"]:
                        self._log(
                            f"相机 {cam_ids[i]} 检测到缺陷，"
                            f"异常分数: {result['anomaly_score']:.4f}"
                        )
                    if self._result_cb:
                        self._result_cb(cam_ids[i], result)
            except Exception:
                self._detect_errors += 1
                self._log(f"推理错误：{traceback.format_exc()}")

            self._emit_heartbeat()

            time.sleep(0.05)

        self._log("推理线程已退出")

    def _emit_heartbeat(self) -> None:
        now = time.monotonic()
        if now - self._last_heartbeat_ts < 1.0:
            return
        self._last_heartbeat_ts = now
        q_sizes = [self._queues[i].qsize() for i in self._active_cams if 0 <= i < len(self._queues)]
        self._log(
            "心跳: "
            f"active={self._active_cams}, "
            f"queue={q_sizes}, "
            f"recv={self._frames_received}, "
            f"enq={self._frames_enqueued}, "
            f"drop={self._frames_dropped}, "
            f"detect_calls={self._detect_calls}, "
            f"detect_err={self._detect_errors}, "
            f"last_detect_ms={self._last_detect_ms:.1f}"
        )

    # ── 公开接口 ──────────────────────────────────────────

    def set_result_callback(self, fn: Callable) -> None:
        """注册推理结果回调，fn(cam_id: int, result: dict)。"""
        self._result_cb = fn

    def on_frame(self, cam_id: int, image: np.ndarray) -> None:
        """
        相机帧回调入口。按 interval 抽帧后入队。
        由 CameraManager 的帧回调调用，运行在取流线程中。
        """
        if not self._running or image is None or image.size == 0:
            return
        if cam_id < 0 or cam_id >= self._n_cams:
            return

        self._frames_received += 1

        self._frame_counters[cam_id] += 1
        if self._frame_counters[cam_id] % self._interval != 0:
            return

        q = self._queues[cam_id]
        # 队列满时丢弃最旧帧
        if q.full():
            try:
                q.get_nowait()
                q.task_done()
                self._frames_dropped += 1
            except queue.Empty:
                pass

        try:
            q.put_nowait({"image": image.copy(), "camera_id": cam_id})
            self._frames_enqueued += 1
        except queue.Full:
            self._frames_dropped += 1

    def start(self) -> None:
        """启动推理线程。"""
        if self._running:
            return
        self._exit_event.clear()
        self._running = True
        self._last_heartbeat_ts = time.monotonic()
        self._frames_received = 0
        self._frames_enqueued = 0
        self._frames_dropped = 0
        self._detect_calls = 0
        self._detect_errors = 0
        self._last_detect_ms = 0.0
        self.reset_counters()
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="InferenceWorker"
        )
        self._thread.start()

    def stop(self) -> None:
        """停止推理线程并清空队列。"""
        self._running = False
        self._exit_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._flush_queues()

    def reset_counters(self) -> None:
        self._frame_counters = [0] * self._n_cams

    def set_active_cameras(self, cam_ids: List[int]) -> None:
        valid = sorted({cid for cid in cam_ids if 0 <= cid < self._n_cams})
        self._active_cams = valid if valid else list(range(self._n_cams))

    def update_params(
        self,
        interval: Optional[int] = None,
        save_dir: Optional[str] = None,
        anomaly_threshold: Optional[float] = None,
        min_defect_area: Optional[int] = None,
    ) -> None:
        if interval is not None and interval > 0:
            self._interval = int(interval)
        if save_dir is not None and save_dir.strip():
            self._save_dir = save_dir.strip()
        if anomaly_threshold is not None:
            self._anomaly_threshold = float(anomaly_threshold)
        if min_defect_area is not None and min_defect_area > 0:
            self._min_defect_area = int(min_defect_area)

        if hasattr(self._detector, "set_postprocess_params"):
            self._detector.set_postprocess_params(
                anomaly_threshold=self._anomaly_threshold,
                min_defect_area=self._min_defect_area,
            )

    def _flush_queues(self) -> None:
        for q in self._queues:
            while not q.empty():
                try:
                    q.get_nowait()
                    q.task_done()
                except queue.Empty:
                    break

    def get_diagnostics(self) -> dict:
        return {
            "running": self._running,
            "active_cams": list(self._active_cams),
            "queue_sizes": [q.qsize() for q in self._queues],
            "frames_received": self._frames_received,
            "frames_enqueued": self._frames_enqueued,
            "frames_dropped": self._frames_dropped,
            "detect_calls": self._detect_calls,
            "detect_errors": self._detect_errors,
            "last_detect_ms": self._last_detect_ms,
        }

    @property
    def is_running(self) -> bool:
        return self._running