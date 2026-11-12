#!/usr/bin/env python3
"""環境相機場景節點 env_scene_node（參考 vision_pkg/env_vision_node，與它並存、不改動它）。

影像中由左至右：A（黑框，瓶蓋待取）| B | C（8 個方格）；相機在使用者這側（影像下方 = 使用者）。

  srv     /vision/env_scene/get_scene   env_scene_interfaces/GetEnvScene   取得最新場景（不發布 topic）
  srv     /vision/env_scene/calibrate   std_srvs/Trigger   場地/黑框/8 格重新定位（C 區 8 格請淨空、線材移開）

標定結果存在 <calib_dir>/env_scene_layout.yaml，重啟沿用；沒有存檔時開機自動標定。
"""
from __future__ import annotations

import dataclasses
import sys
import threading
import time
import traceback
from typing import List, Optional

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger

from env_scene_interfaces.msg import EnvScene, SceneObject
from env_scene_interfaces.srv import GetEnvScene
from vision_pkg import calib_io
from vision_pkg.node_common import (Debouncer, FrameBuffer, LatestValue, image_msg_to_bgr, image_problem,
                                    spin_node, stamp)
from vision_pkg.vision_core.color_zone import MaskParams, ZoneFindParams

from env_scene import scene_core as sc

CAM_FRAME = 'env_camera_optical_frame'
LAYOUT_FILE = 'env_scene_layout.yaml'
MEAS_FILE = 'env_field_homography.yaml'     # calibrate_points 產生
NESTED = {'field_mask': MaskParams, 'field_find': ZoneFindParams, 'obj_mask': MaskParams}


def _declare_scene_params(node: Node) -> None:
    for f in dataclasses.fields(sc.SceneParams):
        if f.name in NESTED:
            for g in dataclasses.fields(NESTED[f.name]):
                node.declare_parameter(f'scene.{f.name}.{g.name}', getattr(f.default, g.name))
        else:
            node.declare_parameter(f'scene.{f.name}', f.default)


def _read_scene_params(node: Node) -> sc.SceneParams:
    kw = {}
    for f in dataclasses.fields(sc.SceneParams):
        if f.name in NESTED:
            kw[f.name] = NESTED[f.name](**{g.name: node.get_parameter(f'scene.{f.name}.{g.name}').value
                                           for g in dataclasses.fields(NESTED[f.name])})
        else:
            kw[f.name] = node.get_parameter(f'scene.{f.name}').value
    return sc.SceneParams(**kw)


class EnvSceneNode(Node):
    def __init__(self):
        super().__init__('env_scene_node')
        for name, default in [('image_topic', '/env_cam/image_raw'), ('show_debug', True), ('process_hz', 10.0),
                              ('image_timeout_s', 1.0), ('service_timeout_s', 3.0), ('calib_dir', ''),
                              ('calib_frames', 5), ('boot_wait_s', 10.0), ('occupied_frames', 3)]:
            self.declare_parameter(name, default)
        _declare_scene_params(self)
        g = lambda n: self.get_parameter(n).value
        self.p = _read_scene_params(self)
        self.show_debug = bool(g('show_debug'))
        self.timeout = float(g('image_timeout_s'))
        self.srv_timeout = float(g('service_timeout_s'))
        self.calib_n = int(g('calib_frames'))
        self.boot_wait = float(g('boot_wait_s'))
        self.occ_frames = int(g('occupied_frames'))
        self.calib_dir = calib_io.resolve_calib_dir(str(g('calib_dir')))
        self.layout_path = self.calib_dir / LAYOUT_FILE
        self.meas_path = self.calib_dir / MEAS_FILE

        self._lock = threading.Lock()
        self.frames = FrameBuffer(maxlen=max(8, self.calib_n + 2))
        self.state = LatestValue()
        self.layout: Optional[sc.Layout] = None
        self._masks = None
        self._debs: List[Debouncer] = []
        self._last_seq = 0
        self._t0 = time.monotonic()
        self._debug_img: Optional[np.ndarray] = None
        self._problem = ''                      # 最近一次無法產生結果的原因（給 service 回報）
        self.hsv = calib_io.ReloadableConfig(lambda: calib_io.load_hsv_ranges(self.calib_dir),
                                             calib_io.hsv_paths(self.calib_dir), 2.0)
        # 開機一律用即時影像重新標定（相機可能動過）；失敗才讀上次存檔

        g_img, g_proc, g_srv = (MutuallyExclusiveCallbackGroup() for _ in range(3))
        self.create_subscription(Image, str(g('image_topic')), self._on_image, qos_profile_sensor_data,
                                 callback_group=g_img)
        self.create_timer(1.0 / max(1.0, float(g('process_hz'))), self._on_tick, callback_group=g_proc)
        self.create_service(GetEnvScene, '/vision/env_scene/get_scene', self._srv_get, callback_group=g_srv)
        self.create_service(Trigger, '/vision/env_scene/calibrate', self._srv_calibrate, callback_group=g_srv)
        self.get_logger().info(f'env_scene_node 就緒；layout 檔 {self.layout_path}')

    # ---------------- 標定 ----------------
    def _load_layout(self) -> None:
        if not self.layout_path.exists():
            return
        try:
            self._set_layout(sc.layout_from_dict(calib_io.load_yaml(self.layout_path)))
            self.get_logger().info(f'讀取上次標定: {self.layout.message}')
        except Exception as e:
            self.get_logger().error(f'讀取 {self.layout_path} 失敗（將重新標定）: {e}')

    def _apply_measured(self, lay: sc.Layout) -> None:
        """有 calibrate_points 的標定檔就改用它（不受場地裁切影響，給手臂用）。"""
        if not lay.ok or not self.meas_path.exists():
            return
        try:
            d = calib_io.load_yaml(self.meas_path)
            if tuple(d['image_size']) != tuple(lay.image_size):
                self.get_logger().error(f'{self.meas_path} 影像大小 {d["image_size"]} 與目前 {lay.image_size} 不同，忽略')
                return
            sc.apply_measured_homography(lay, np.array(d['H'], np.float64))
            if d.get('estimated'):
                lay.coords_source = 'estimated'
                self.get_logger().warning(f'座標用「估算值」標定 {self.meas_path}（不是實際尺量，只供試用）')
            else:
                self.get_logger().info(f'座標改用尺量標定 {self.meas_path}（RMS {d.get("rms_error_mm", float("nan")):.2f} mm）')
        except Exception as e:
            self.get_logger().error(f'讀取 {self.meas_path} 失敗（座標只靠場地四角，誤差 1–2 cm）: {e}')

    def _set_layout(self, lay: sc.Layout) -> None:
        self._apply_measured(lay)
        with self._lock:
            self.layout = lay
            self._masks = sc.roi_masks(lay, self.p) if lay.ok else None
            self._debs = [Debouncer(self.occ_frames, self.occ_frames) for _ in range(len(lay.slots))]

    def _reuse_saved_slots(self, lay: sc.Layout) -> None:
        """這次 8 格沒找到（例如方格旁邊有瓶蓋）：沿用上次成功的格位像素位置。相機動過時會錯，所以記警告。"""
        try:
            old = sc.layout_from_dict(calib_io.load_yaml(self.layout_path)) if self.layout_path.exists() else None
        except Exception:
            old = None
        if old is None or not old.slots_ok or tuple(old.image_size) != tuple(lay.image_size):
            return
        lay.slots = old.slots
        for s in lay.slots:
            s.x_mm, s.y_mm = (float(v) for v in lay.img_to_field([[s.u, s.v]])[0])
        lay.slots_ok = True
        sc._check_slots(lay, self.p.slot_check_max_mm)
        self.get_logger().warning(f'C 區 8 格這次沒找到（{lay.slots_message}），沿用上次存檔的格位；'
                                  '相機若動過，請在方格淨空時呼叫 /vision/env_scene/calibrate')
        lay.slots_message = '沿用上次格位'
        lay.message += '；C 區 8 格沿用上次存檔'

    def _calibrate(self, frames) -> sc.Layout:
        hsv = self.hsv.value
        if hsv is None:
            return sc.Layout(False, f'HSV 載入失敗: {self.hsv.error}')
        lay = sc.calibrate(frames, hsv, self.p, min_detections=max(1, len(frames) // 2 + 1))
        if lay.ok and not lay.slots_ok:
            self._reuse_saved_slots(lay)
        if lay.ok:
            to_save = sc.layout_to_dict(lay)      # 存「場地四角」版本；載入時再套尺量標定
            self._set_layout(lay)
            try:
                calib_io.save_yaml(self.layout_path, to_save,
                                   '# env_scene_node 自動產生（影像像素 / 場地 mm）\n')
            except Exception as e:
                self.get_logger().warning(f'layout 存檔失敗: {e}')
        return lay

    # ---------------- 影像 / 處理 ----------------
    def _on_image(self, msg: Image) -> None:
        try:
            self.frames.push(image_msg_to_bgr(msg))
        except Exception as e:
            self.get_logger().warning(f'影像轉換失敗: {e}', throttle_duration_sec=5.0)

    def _on_tick(self) -> None:
        try:
            self._tick()
        except Exception:
            self.get_logger().error('處理例外:\n' + traceback.format_exc(), throttle_duration_sec=5.0)

    def _tick(self) -> None:
        self.hsv.poll()
        now = time.monotonic()
        latest = self.frames.latest()
        problem = image_problem(latest, now, self.timeout)
        if problem:
            self._problem = problem
            return
        seq, _, frame = latest
        if seq == self._last_seq:
            return
        self._last_seq = seq
        with self._lock:
            lay = self.layout
        if lay is None or not lay.ok or tuple(lay.image_size) != (frame.shape[1], frame.shape[0]):
            got = self.frames.latest_n(self.calib_n)
            if len(got) >= self.calib_n or now - self._t0 > self.boot_wait:
                lay = self._calibrate([f for _, _, f in got])
                (self.get_logger().info if lay.ok and lay.slots_ok else self.get_logger().warning)(
                    f'自動標定: {lay.message}')
                if (not lay.ok or not lay.slots_ok) and self.layout is None:
                    self._load_layout()
                    lay = self.layout or lay
            if lay is None or not lay.ok:
                self._problem = '場地尚未定位' + (f'：{lay.message}' if lay else '')
                return
        hsv = self.hsv.value
        if hsv is None:
            self._problem = f'HSV 載入失敗: {self.hsv.error}'
            return
        with self._lock:
            masks, debs = self._masks, self._debs
        res, _ = sc.analyze(frame, lay, hsv, self.p, masks)
        if res.ok and res.slot_occupied:
            res.slot_occupied = [d.update(o) for d, o in zip(debs, res.slot_occupied)]
        self._problem = ''
        self.state.set((lay, res))
        if self.show_debug:
            self._debug_img = sc.draw(frame, lay, res)

    # ---------------- 訊息 ----------------
    def _to_msg(self, lay: Optional[sc.Layout], res) -> EnvScene:
        m = EnvScene()
        if lay is None or isinstance(res, str) or not res.ok:
            m.valid = False
            m.message = res if isinstance(res, str) else res.message
            return stamp(self, m, CAM_FRAME)
        m.valid, m.message = True, res.message
        m.coords_source = lay.coords_source
        m.calib_check_mm = float(lay.check_mm)
        ok = lay.slots_ok and np.isfinite(lay.check_mm) and lay.check_mm <= self.p.slot_check_max_mm
        m.arm1_valid = bool(ok or lay.coords_source == 'measured')
        m.arm2_valid = bool(m.arm1_valid and self.p.arm2_enabled)
        m.a_visible_ratio = float(lay.zones_visible.get('a', 0.0))

        def obj(o: sc.Obj) -> SceneObject:
            a1, a2 = sc.field_to_arm(o.x_mm, o.y_mm, 'arm1'), sc.field_to_arm(o.x_mm, o.y_mm, 'arm2')
            return SceneObject(zone=o.zone, u=float(o.u), v=float(o.v), x_mm=float(o.x_mm), y_mm=float(o.y_mm),
                               diameter_mm=float(o.diameter_mm), arm1_x=a1[0], arm1_y=a1[1],
                               arm2_x=a2[0], arm2_y=a2[1])
        m.a_objects = [obj(o) for o in res.a_objects]
        m.b_objects = [obj(o) for o in res.b_objects]
        m.c_slots_valid = bool(lay.slots_ok and len(res.slot_occupied) == sc.N_SLOTS)
        if m.c_slots_valid:
            m.c_occupied = [bool(x) for x in res.slot_occupied]
            m.c_fill_ratio = [float(x) for x in res.slot_ratio]
            m.c_u = [float(s.u) for s in lay.slots]
            m.c_v = [float(s.v) for s in lay.slots]
            m.c_x_mm = [float(s.x_mm) for s in lay.slots]
            m.c_y_mm = [float(s.y_mm) for s in lay.slots]
            m.c_occupied_count = int(sum(m.c_occupied))
            a1 = [sc.field_to_arm(s.x_mm, s.y_mm, 'arm1') for s in lay.slots]
            a2 = [sc.field_to_arm(s.x_mm, s.y_mm, 'arm2') for s in lay.slots]
            m.c_arm1_x, m.c_arm1_y = [a[0] for a in a1], [a[1] for a in a1]
            m.c_arm2_x, m.c_arm2_y = [a[0] for a in a2], [a[1] for a in a2]
        return stamp(self, m, CAM_FRAME)

    # ---------------- services ----------------
    def _srv_get(self, req, resp):
        try:
            v, why = self.state.wait_fresh(self.timeout, self.srv_timeout)
            if v is None:
                why = self._problem or why
                resp.success, resp.message = False, why
                resp.scene = self._to_msg(None, why)
            else:
                resp.scene = self._to_msg(*v)
                resp.success, resp.message = resp.scene.valid, resp.scene.message
        except Exception as e:
            resp.success, resp.message = False, f'內部錯誤: {e!r}'
        return resp

    def _srv_calibrate(self, req, resp):
        try:
            self.hsv.poll(force=True)
            frames = self.frames.wait_fresh(self.calib_n, self.srv_timeout)
            if len(frames) < self.calib_n:
                resp.success, resp.message = False, f'逾時：只收到 {len(frames)}/{self.calib_n} 張新影像'
                return resp
            lay = self._calibrate(frames)
            resp.success = bool(lay.ok and lay.slots_ok)
            resp.message = lay.message
        except Exception as e:
            self.get_logger().error('calibrate 例外:\n' + traceback.format_exc())
            resp.success, resp.message = False, f'內部錯誤: {e!r}'
        return resp

    def get_debug_image(self):
        return self._debug_img


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = EnvSceneNode()
        ex = MultiThreadedExecutor(num_threads=4)
        ex.add_node(node)
        spin_node(rclpy, node, ex, node.show_debug, 'env_scene')
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
