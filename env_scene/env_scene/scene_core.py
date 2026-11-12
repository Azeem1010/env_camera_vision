"""環境相機場景分析（純函式，不含 ROS）。

場地採用 dual_arm_system 的標準定義（docs/COORDINATE_FRAMES.md、scripts/field_layout.py）：

    影像左 ─────────────────────────────────────────── 影像右
    |   A 區 x=-300..-100   |  B 區 -100..100  |  C 區 100..300（8 格）|     手臂側 y=+120
    ─────────────── 使用者這側（影像下方）y=-120 ───────────────

  world 座標（mm）：原點 = 場地中心（兩塊板子的接縫中點），x 由 A 往 C，y 遠離使用者（朝手臂），場地 600 x 240 mm。
  手臂基座 armN_base（m）：原點 = 底座旋轉軸；x 朝使用者（= -y_world），y = A→C（= +x_world）
      arm1 軸心 world (-100, 125) mm，arm2 (+100, 125) mm（手臂側長邊外 5 mm）
  C 區 8 格標準位置：x = 135（單號，靠 B）/ 170（雙號），y = 50 / 15 / -20 / -55；1、2 最靠手臂。

標定（不需要看到場地左右兩端，相機無法再拉高）：
  y = ±120  場地上下兩條長邊（逐欄穩健擬合直線）
  x = 0     兩塊板子的接縫（逐列找垂直邊緣，穩健擬合直線）
  比例       接縫處場地高度 = 240 mm（x、y 同比例）
  驗證       C 區 8 個方格中心 vs 標準格位，誤差寫在 Layout.check_message

流程
  calibrate()：場地淨空時呼叫一次（瓶蓋可在 A 區）
  analyze()  ：每張影格 → A、B 區瓶蓋座標，C 區 8 格佔用
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from vision_pkg.vision_core.cap_detect import CapParams, detect_caps
from vision_pkg.vision_core.color_zone import (HsvRange, MaskParams, ZoneFindParams, fuse_corners, hsv_mask,
                                               in_range_hsv, largest_contour, max_area_quad, to_hsv)
from vision_pkg.vision_core.geometry import apply_homography, order_corners, polygon_area, polygon_mask

FIELD_W_MM, FIELD_H_MM = 600.0, 240.0
HALF_W, HALF_H = FIELD_W_MM / 2, FIELD_H_MM / 2
ZONES_MM = {'a': (-300.0, -100.0), 'b': (-100.0, 100.0), 'c': (100.0, 300.0)}
STD_SLOTS_MM = [(x, y) for y in (50.0, 15.0, -20.0, -55.0) for x in (135.0, 170.0)]   # 格位 1..8
ARM_BASES_MM = {'arm1': (-100.0, 125.0), 'arm2': (100.0, 125.0)}
N_SLOTS = 8


# --------------------------------------------------------------------------- #
# 參數
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SceneParams:
    field_mask: MaskParams = MaskParams(5, 3, 9)
    field_find: ZoneFindParams = ZoneFindParams(20000.0, 0.85)
    bridge_close_px: int = 31            # 補起黑框膠帶（> 膠帶在影像中的寬度，約 20 px）
    strip_open_px: int = 41              # 削掉場地邊上較細的粉紅凸出物（< 場地高度）
    obj_mask: MaskParams = MaskParams(3, 3, 5)
    zone_erode_px: int = 4               # A/B 區 ROI 內縮
    edge_fit_from: float = 0.3           # 上下邊擬合用的欄範圍（影像寬度比例）；避開 A 區黑框膠帶
    edge_fit_to: float = 0.98
    seam_search_from: float = 0.3        # 接縫搜尋範圍（影像寬度比例）
    seam_search_to: float = 0.7
    # 瓶蓋（mm，換算成像素）
    cap_diameter_mm: float = 20.0
    cap_min_area_ratio: float = 0.35
    cap_max_area_ratio: float = 2.2
    cap_min_circularity: float = 0.6
    cap_min_fill: float = 0.45
    # C 區方格
    slot_min_side_mm: float = 15.0
    slot_max_side_mm: float = 45.0
    slot_max_aspect: float = 1.6
    slot_min_fill: float = 0.55
    slot_bright_v_min: int = 140         # 白紙：非粉紅、V >= 此值
    slot_margin_ratio: float = 0.12      # 計算佔用時，方格四邊各內縮的比例
    slot_occupied_threshold: float = 0.25  # 格內藍色比例 >= 此值 = 已佔用
    slot_first_row: str = 'far'          # 'far'：1、2 最靠手臂（標準、實測）；'near'：反過來
    slot_left_to_right: bool = True      # 同一排靠 B（x 小）的是單號
    slot_check_max_mm: float = 10.0      # 方格中心與標準格位相差超過此值 -> 警告（標定可能有誤）
    arm2_enabled: bool = True            # arm2_base 軸向依 COORDINATE_FRAMES.md 建議值（同手臂 1）


# --------------------------------------------------------------------------- #
# 結果資料
# --------------------------------------------------------------------------- #
@dataclass
class Slot:
    index: int                      # 1..8
    corners_px: np.ndarray          # (4,2) tl,tr,br,bl
    u: float
    v: float
    x_mm: float                     # world
    y_mm: float


@dataclass
class Layout:
    ok: bool
    message: str
    image_size: Tuple[int, int] = (0, 0)              # (w, h)
    H: Optional[np.ndarray] = None                    # 影像像素 -> world mm（接縫 + 上下邊）
    px_per_mm: float = 0.0
    zones_px: Dict[str, np.ndarray] = field(default_factory=dict)   # 各區四角（tl,tr,br,bl；可能部分在畫面外）
    zones_visible: Dict[str, float] = field(default_factory=dict)   # 各區在畫面內的比例
    slots: List[Slot] = field(default_factory=list)   # ok 時 8 個，依格位 1..8
    slots_ok: bool = False
    slots_message: str = ''
    check_mm: float = float('nan')                    # 8 格中心 vs 標準格位的最大誤差 (mm)
    check_message: str = ''
    H_meas: Optional[np.ndarray] = None               # 瓶蓋+尺量標定（有的話優先）
    coords_source: str = 'standard'                   # 'standard'（接縫+上下邊）| 'measured' | 'estimated'

    def _H(self) -> np.ndarray:
        return self.H_meas if self.H_meas is not None else self.H

    def img_to_field(self, pts) -> np.ndarray:
        """影像像素 -> world mm。"""
        return apply_homography(self._H(), pts)

    def field_to_img(self, pts) -> np.ndarray:
        return apply_homography(np.linalg.inv(self._H()), pts)


@dataclass
class Obj:
    zone: str
    u: float
    v: float
    x_mm: float
    y_mm: float
    diameter_mm: float


def field_to_arm(x_mm: float, y_mm: float, arm: str) -> Tuple[float, float]:
    """world mm -> armN_base (x, y) m。x 朝使用者（= -y_world），y = A→C（= +x_world）。"""
    ox, oy = ARM_BASES_MM[arm]
    return (oy - y_mm) / 1000.0, (x_mm - ox) / 1000.0


def apply_measured_homography(lay: 'Layout', H: np.ndarray) -> None:
    """改用尺量標定的 homography：重算各區範圍與 8 格的 world 座標。"""
    lay.H_meas = np.asarray(H, np.float64)
    lay.coords_source = 'measured'
    _update_zones(lay)
    for s in lay.slots:
        s.x_mm, s.y_mm = (float(v) for v in apply_homography(lay.H_meas, [[s.u, s.v]])[0])
    _check_slots(lay, None)


@dataclass
class SceneResult:
    ok: bool
    message: str
    a_objects: List[Obj] = field(default_factory=list)
    b_objects: List[Obj] = field(default_factory=list)
    slot_occupied: List[bool] = field(default_factory=list)
    slot_ratio: List[float] = field(default_factory=list)
    rejected: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# 幾何
# --------------------------------------------------------------------------- #
def _line_intersect(p1, p2, q1, q2) -> np.ndarray:
    """直線 p1p2 與 q1q2 的交點。"""
    p1, p2, q1, q2 = (np.asarray(v, np.float64) for v in (p1, p2, q1, q2))
    d1, d2 = p2 - p1, q2 - q1
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-9:
        raise ValueError('平行線')
    t = ((q1[0] - p1[0]) * d2[1] - (q1[1] - p1[1]) * d2[0]) / den
    return p1 + t * d1


def _robust_line(xs, ys) -> Tuple[float, float, np.ndarray]:
    """y = a x + b 的穩健擬合（反覆剔除離群值），回傳 (a, b, inlier 遮罩)。"""
    xs, ys = np.asarray(xs, np.float64), np.asarray(ys, np.float64)
    keep = np.ones(len(ys), bool)
    for _ in range(5):
        a, b = np.polyfit(xs[keep], ys[keep], 1)
        res = np.abs(ys - (a * xs + b))
        keep = res <= max(2.0, 2.5 * np.median(res[keep]))
    a, b = np.polyfit(xs[keep], ys[keep], 1)
    return float(a), float(b), keep


def _zone_quad_px(lay: Layout, x0: float, x1: float) -> np.ndarray:
    Hinv = np.linalg.inv(lay._H())
    return apply_homography(Hinv, [[x0, HALF_H], [x1, HALF_H], [x1, -HALF_H], [x0, -HALF_H]]).astype(np.float32)


def _update_zones(lay: Layout) -> None:
    w, h = lay.image_size
    lay.zones_px, lay.zones_visible = {}, {}
    for z, (x0, x1) in ZONES_MM.items():
        q = _zone_quad_px(lay, x0, x1)
        lay.zones_px[z] = q
        full = polygon_area(q)
        vis = cv2.countNonZero(polygon_mask((h, w), q, 0))
        lay.zones_visible[z] = float(min(1.0, vis / max(1.0, full)))


def _check_slots(lay: Layout, max_mm: Optional[float]) -> None:
    """8 格中心 vs 標準格位（2026-11-11 實測，直尺 0.5 cm）。"""
    if not lay.slots_ok:
        lay.check_mm, lay.check_message = float('nan'), ''
        return
    err = [math.hypot(s.x_mm - sx, s.y_mm - sy) for s, (sx, sy) in zip(lay.slots, STD_SLOTS_MM)]
    lay.check_mm = float(max(err))
    lay.check_message = f'8 格 vs 標準格位：平均 {np.mean(err):.1f} mm、最大 {max(err):.1f} mm'
    if max_mm is not None and lay.check_mm > max_mm:
        lay.check_message += f'（> {max_mm:.0f} mm，標定或接縫偵測可能有誤）'


# --------------------------------------------------------------------------- #
# 標定（場地淨空時）
# --------------------------------------------------------------------------- #
def _edge_lines(pink_mask: np.ndarray, x0: int, x1: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """在 x0..x1 欄間，逐欄找場地（最長粉紅連續段）的上緣、下緣，穩健擬合兩條直線 y = a x + b。

    凸出物（手臂底座泡棉、旁邊另一塊粉紅板）只影響少數欄，會被當成離群值排除。
    """
    xs, top, bot = [], [], []
    for x in range(max(0, x0), min(pink_mask.shape[1], x1), 4):
        ys = np.flatnonzero(pink_mask[:, x])
        if len(ys) < 20:
            continue
        runs = np.split(ys, np.flatnonzero(np.diff(ys) > 3) + 1)
        r = max(runs, key=len)
        xs.append(x), top.append(r[0]), bot.append(r[-1])
    if len(xs) < 10:
        return None
    at, bt, _ = _robust_line(xs, top)
    ab, bb, _ = _robust_line(xs, bot)
    return np.array([at, bt]), np.array([ab, bb])


def _find_seam(gray: np.ndarray, pink_raw: np.ndarray, top: np.ndarray, bot: np.ndarray, u0: int, u1: int
               ) -> Tuple[Optional[Tuple[float, float]], str]:
    """兩塊板子的接縫：在場地上下邊之間逐列找最強的垂直邊緣，穩健擬合 u = a v + b。

    只看「左右兩邊都是粉紅」的位置：細細的接縫用小閉運算補起來後仍在範圍內；
    黑框膠帶、B 區黑色小方塊、瓶蓋等較寬的非粉紅物體及其邊緣都被排除。
    """
    k = np.ones((7, 7), np.uint8)
    inside = cv2.erode(cv2.morphologyEx(pink_raw, cv2.MORPH_CLOSE, k), k)
    gx = np.abs(cv2.Sobel(cv2.GaussianBlur(gray, (5, 5), 0), cv2.CV_64F, 1, 0, ksize=3))
    gx[inside == 0] = 0
    uc = (u0 + u1) / 2.0
    v_top = top[0] * uc + top[1]
    v_bot = bot[0] * uc + bot[1]
    vs, us = [], []
    for v in range(int(v_top) + 12, int(v_bot) - 12, 3):
        prof = gx[max(0, v - 4):v + 5, u0:u1].mean(axis=0)
        k = int(np.argmax(prof))
        if prof[k] > 3.0 * max(np.median(prof[prof > 0]) if (prof > 0).any() else 0.0, 0.5):   # 夠明顯才採用
            vs.append(v), us.append(u0 + k)
    if len(vs) < 20:
        return None, f'接縫邊緣太少（{len(vs)} 列）'
    a, b, keep = _robust_line(vs, us)
    if keep.mean() < 0.5:
        return None, f'接縫不是一條直線（只有 {keep.mean():.0%} 的列一致）'
    return (a, b), f'接縫 {keep.sum()}/{len(vs)} 列'


def _slot_mask(bgr: np.ndarray, pink: HsvRange, blue: HsvRange, p: SceneParams) -> np.ndarray:
    hsv = to_hsv(bgr, 3)
    not_pink = cv2.bitwise_not(in_range_hsv(hsv, pink))
    bright = cv2.inRange(hsv, np.array((0, 0, p.slot_bright_v_min), np.uint8), np.array((179, 255, 255), np.uint8))
    m = cv2.bitwise_or(cv2.bitwise_and(not_pink, bright), in_range_hsv(hsv, blue))   # 白紙 或 格內瓶蓋
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def _find_slots(bgr: np.ndarray, pink: HsvRange, blue: HsvRange, c_quad: np.ndarray, H_to_field: np.ndarray,
                px_per_mm: float, p: SceneParams) -> Tuple[List[Slot], str]:
    h, w = bgr.shape[:2]
    roi = polygon_mask((h, w), c_quad, max(2, p.zone_erode_px))
    m = cv2.bitwise_and(_slot_mask(bgr, pink, blue, p), roi)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    amin = (p.slot_min_side_mm * px_per_mm) ** 2
    amax = (p.slot_max_side_mm * px_per_mm) ** 2
    good, notes = [], []
    for c in cnts:
        if cv2.contourArea(c) < amin * 0.5:
            continue
        c = cv2.convexHull(c)          # 被線材壓出缺口的方格仍取完整外形
        area = cv2.contourArea(c)
        rect = cv2.minAreaRect(c)
        rw, rh = rect[1]
        aspect = max(rw, rh) / max(1e-6, min(rw, rh))
        fill = area / max(1e-6, rw * rh)
        why = []
        if not amin <= area <= amax:
            why.append(f'面積{area:.0f}')
        if aspect > p.slot_max_aspect:
            why.append(f'長寬比{aspect:.2f}')
        if fill < p.slot_min_fill:
            why.append(f'填滿率{fill:.2f}')
        if why:
            notes.append(f'@({rect[0][0]:.0f},{rect[0][1]:.0f}) ' + ','.join(why))
            continue
        good.append(order_corners(cv2.boxPoints(rect)).astype(np.float32))
    if len(good) != N_SLOTS:
        msg = f'C 區找到 {len(good)} 個方格（預期 {N_SLOTS}；方格旁邊不要放瓶蓋）'
        if notes:
            msg += '；排除：' + '；'.join(notes[:4])
        return [], msg

    ctr_px = np.array([q.mean(axis=0) for q in good])
    ctr_mm = apply_homography(H_to_field, ctr_px)
    # 排成 4 排（world y 方向），每排 2 個（world x 方向）
    by_y = np.argsort(-ctr_mm[:, 1]) if p.slot_first_row == 'far' else np.argsort(ctr_mm[:, 1])
    rows = [list(by_y[2 * r:2 * r + 2]) for r in range(4)]
    row_y = [float(ctr_mm[r, 1].mean()) for r in rows]
    gaps = [abs(row_y[i] - row_y[i + 1]) for i in range(3)]
    spread = max(abs(float(ctr_mm[r[0], 1] - ctr_mm[r[1], 1])) for r in rows)
    if spread > 0.5 * min(gaps):
        return [], f'方格排列不像 4 排 x 2 個（同排 y 差 {spread:.0f} mm，排距最小 {min(gaps):.0f} mm）'
    slots: List[Slot] = []
    for r in rows:
        r = sorted(r, key=lambda i: ctr_mm[i, 0], reverse=not p.slot_left_to_right)
        for i in r:
            slots.append(Slot(len(slots) + 1, good[i], float(ctr_px[i, 0]), float(ctr_px[i, 1]),
                              float(ctr_mm[i, 0]), float(ctr_mm[i, 1])))
    return slots, 'ok'


def field_mask(bgr: np.ndarray, hsv: Dict[str, HsvRange], p: SceneParams) -> np.ndarray:
    """粉紅場地 + 放在場地上的瓶蓋，閉運算補起黑框膠帶與接縫。"""
    m = hsv_mask(bgr, hsv['pink'], p.field_mask)
    if 'blue' in hsv:
        m = cv2.bitwise_or(m, hsv_mask(bgr, hsv['blue'], p.obj_mask))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.bridge_close_px | 1,) * 2)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)


def find_field_quad(bgr: np.ndarray, pink: HsvRange, p: SceneParams,
                    blue: Optional[HsvRange] = None) -> Tuple[Optional[np.ndarray], str]:
    """可見場地範圍的四角（tl,tr,br,bl；左右可能是畫面邊緣）。只用來估計比例與範圍，不用來定座標。"""
    hsv = {'pink': pink} if blue is None else {'pink': pink, 'blue': blue}
    m = field_mask(bgr, hsv, p)
    w = m.shape[1]
    edges = _edge_lines(m, int(p.edge_fit_from * w), int(p.edge_fit_to * w))
    if edges is None:
        return None, '找不到粉紅場地的上下邊（HSV pink？場地被大面積遮住？）'
    (at, bt), (ab, bb) = edges
    xs = [x for x in range(0, w, 2)
          if m[max(0, int(at * x + bt)):max(0, int(ab * x + bb)) + 1, x].mean() > 0.8 * 255]
    if len(xs) < 10:
        return None, '場地上下邊之間大多不是粉紅色'
    xl, xr = float(min(xs)), float(max(xs))
    return np.float32([[xl, at * xl + bt], [xr, at * xr + bt], [xr, ab * xr + bb], [xl, ab * xl + bb]]), ''


def _calibrate_one(bgr: np.ndarray, hsv: Dict[str, HsvRange], p: SceneParams
                   ) -> Tuple[Optional[np.ndarray], float, str]:
    """單張影格：回傳 (H 影像->world mm, px_per_mm, 說明)；失敗時 H=None。"""
    h, w = bgr.shape[:2]
    m = field_mask(bgr, hsv, p)
    edges = _edge_lines(m, int(p.edge_fit_from * w), int(p.edge_fit_to * w))
    if edges is None:
        return None, 0.0, '找不到粉紅場地的上下邊'
    top, bot = edges
    seam, why = _find_seam(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64),
                           hsv_mask(bgr, hsv['pink'], p.field_mask), top, bot,
                           int(p.seam_search_from * w), int(p.seam_search_to * w))
    if seam is None:
        return None, 0.0, f'找不到中間接縫（x=0）：{why}'
    sa, sb = seam
    s1, s2 = (sb, 0.0), (sa * h + sb, float(h))                  # 接縫上兩點 (u, v)
    t1, t2 = (0.0, top[1]), (float(w), top[0] * w + top[1])
    b1, b2 = (0.0, bot[1]), (float(w), bot[0] * w + bot[1])
    pt = _line_intersect(s1, s2, t1, t2)
    pb = _line_intersect(s1, s2, b1, b2)
    ppm = float(np.linalg.norm(pb - pt) / FIELD_H_MM)
    dt = (np.subtract(t2, t1)) / np.linalg.norm(np.subtract(t2, t1))
    db = (np.subtract(b2, b1)) / np.linalg.norm(np.subtract(b2, b1))
    src = np.float32([pt - HALF_W * ppm * dt, pt + HALF_W * ppm * dt, pb + HALF_W * ppm * db, pb - HALF_W * ppm * db])
    dst = np.float32([[-HALF_W, HALF_H], [HALF_W, HALF_H], [HALF_W, -HALF_H], [-HALF_W, -HALF_H]])
    return cv2.getPerspectiveTransform(src, dst), ppm, why


def calibrate(frames: Sequence[np.ndarray], hsv: Dict[str, HsvRange], p: SceneParams,
              min_detections: int = 1) -> Layout:
    """hsv 需含 'pink' 'blue'。多張影格時，四個參考點取逐點中位數。"""
    if not frames:
        return Layout(False, '沒有影像')
    h, w = frames[0].shape[:2]
    quads, ppms, reasons = [], [], []
    dst = np.float32([[-HALF_W, HALF_H], [HALF_W, HALF_H], [HALF_W, -HALF_H], [-HALF_W, -HALF_H]])
    note = ''
    for f in frames:
        H, ppm, why = _calibrate_one(f, hsv, p)
        if H is None:
            reasons.append(why)
            continue
        quads.append(apply_homography(np.linalg.inv(H), dst))
        ppms.append(ppm)
        note = why
    q = fuse_corners(quads, min_detections)
    if q is None:
        return Layout(False, f'標定失敗（{len(quads)}/{len(frames)} 張成功）：{reasons[0] if reasons else ""}', (w, h))
    lay = Layout(True, '', (w, h), cv2.getPerspectiveTransform(np.float32(q), dst), float(np.median(ppms)))
    _update_zones(lay)
    ref = frames[len(frames) // 2]
    lay.slots, lay.slots_message = _find_slots(ref, hsv['pink'], hsv['blue'], lay.zones_px['c'], lay.H,
                                               lay.px_per_mm, p)
    lay.slots_ok = len(lay.slots) == N_SLOTS
    _check_slots(lay, p.slot_check_max_mm)
    msgs = [f'標準場地 600x240：{note}，{lay.px_per_mm:.3f} px/mm',
            '畫面內 ' + ' '.join(f'{z.upper()}={v:.0%}' for z, v in lay.zones_visible.items()),
            lay.check_message if lay.slots_ok else lay.slots_message]
    lay.message = '；'.join(m for m in msgs if m)
    return lay


# --------------------------------------------------------------------------- #
# 每張影格
# --------------------------------------------------------------------------- #
def _cap_params(lay: Layout, p: SceneParams) -> CapParams:
    a = math.pi * (p.cap_diameter_mm * lay.px_per_mm / 2) ** 2
    return CapParams(min_area_px=a * p.cap_min_area_ratio, max_area_px=a * p.cap_max_area_ratio,
                     min_circularity=p.cap_min_circularity, min_fill=p.cap_min_fill,
                     reject_border_touch=False, border_px=0)


def roi_masks(lay: Layout, p: SceneParams) -> Dict[str, np.ndarray]:
    w, h = lay.image_size
    return {'a': polygon_mask((h, w), lay.zones_px['a'], p.zone_erode_px),
            'b': polygon_mask((h, w), lay.zones_px['b'], p.zone_erode_px)}


def slot_ratios(blue_mask: np.ndarray, slots: Sequence[Slot], margin_ratio: float) -> List[float]:
    out = []
    for s in slots:
        q = s.corners_px
        c = q.mean(axis=0)
        q = c + (q - c) * (1.0 - 2.0 * margin_ratio)
        m = np.zeros(blue_mask.shape[:2], np.uint8)
        cv2.fillConvexPoly(m, np.round(q).astype(np.int32), 255)
        n = cv2.countNonZero(m)
        out.append(cv2.countNonZero(cv2.bitwise_and(blue_mask, m)) / n if n else 0.0)
    return out


def split_touching(contour: np.ndarray, shape_hw, single_area: float, max_k: int = 4
                   ) -> List[Tuple[float, float, float]]:
    """相黏的瓶蓋：依面積估計顆數 k（2..max_k），對色塊像素做 k-means，回傳 [(u, v, 直徑px)]；無法拆分回傳 []。"""
    area = cv2.contourArea(contour)
    k = int(round(area / max(single_area, 1e-6)))
    if not 2 <= k <= max_k:
        return []
    m = np.zeros(shape_hw[:2], np.uint8)
    cv2.drawContours(m, [contour], -1, 255, -1)
    ys, xs = np.nonzero(m)
    data = np.stack([xs, ys], axis=1).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, labels, centers = cv2.kmeans(data, k, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    out = []
    for i, (u, v) in enumerate(centers):
        n = int((labels.ravel() == i).sum())
        out.append((float(u), float(v), 2.0 * math.sqrt(n / math.pi)))
    return out


def analyze(bgr: np.ndarray, lay: Layout, hsv: Dict[str, HsvRange], p: SceneParams,
            masks: Optional[Dict[str, np.ndarray]] = None) -> Tuple[SceneResult, np.ndarray]:
    """回傳 (結果, 藍色遮罩)。masks 可傳入 roi_masks() 的快取。"""
    if not lay.ok:
        return SceneResult(False, lay.message), np.zeros(bgr.shape[:2], np.uint8)
    if (bgr.shape[1], bgr.shape[0]) != tuple(lay.image_size):
        return SceneResult(False, f'影像大小 {bgr.shape[1]}x{bgr.shape[0]} 與標定時 {lay.image_size} 不同'), \
            np.zeros(bgr.shape[:2], np.uint8)
    masks = masks or roi_masks(lay, p)
    blue = hsv_mask(bgr, hsv['blue'], p.obj_mask)
    cp = _cap_params(lay, p)
    res = SceneResult(True, 'ok')
    single_area = math.pi * (p.cap_diameter_mm * lay.px_per_mm / 2) ** 2
    for z in ('a', 'b'):
        caps, rej = detect_caps(cv2.bitwise_and(blue, masks[z]), cp)
        pts = [(c.cx, c.cy, c.diameter) for c in caps]
        still_rej = []
        for r in rej:
            split = split_touching(r.contour, blue.shape, single_area) if r.too_large or r.circularity < \
                p.cap_min_circularity else []
            if split:
                pts += split
            else:
                still_rej.append(r)
        rej = still_rej
        objs = []
        if pts:
            mm = lay.img_to_field([(u, v) for u, v, _ in pts])
            for (u, v, d), (x, y) in zip(pts, mm):
                objs.append(Obj(z, float(u), float(v), float(x), float(y), d / lay.px_per_mm))
        objs.sort(key=lambda o: (-o.y_mm, o.x_mm))
        setattr(res, f'{z}_objects', objs)
        res.rejected += [f'{z.upper()}@({r.cx:.0f},{r.cy:.0f}) {r.reject_reason}' for r in rej]
    if lay.slots_ok:
        res.slot_ratio = slot_ratios(blue, lay.slots, p.slot_margin_ratio)
        res.slot_occupied = [r >= p.slot_occupied_threshold for r in res.slot_ratio]
    else:
        res.message = f'C 區方格未定位：{lay.slots_message}'
    if lay.coords_source == 'estimated':
        res.message += '；座標為估算值標定（試用）'
    return res, blue


# --------------------------------------------------------------------------- #
# 存檔
# --------------------------------------------------------------------------- #
def layout_to_dict(lay: Layout) -> dict:
    """存「標準（接縫+上下邊）」版本；尺量標定另外存在 env_field_homography.yaml，載入時再套。"""
    return {
        'version': 2, 'frame': 'world_mm (dual_arm_system 標準場地 600x240，原點=場地中心)',
        'ok': lay.ok, 'message': lay.message, 'image_size': list(lay.image_size),
        'H': np.asarray(lay.H).tolist(), 'px_per_mm': lay.px_per_mm,
        'slots_ok': lay.slots_ok, 'slots_message': lay.slots_message,
        'slots': [{'index': s.index, 'corners_px': np.asarray(s.corners_px).tolist(), 'u': s.u, 'v': s.v,
                   'x_mm': s.x_mm, 'y_mm': s.y_mm} for s in lay.slots],
    }


def layout_from_dict(d: dict) -> Layout:
    if int(d.get('version', 1)) < 2:
        raise ValueError('舊版 layout（300 mm 場地）已不適用，需重新標定')
    f = lambda v: np.asarray(v, np.float32)
    lay = Layout(bool(d['ok']), str(d.get('message', '')), tuple(d['image_size']), np.asarray(d['H'], np.float64),
                 float(d['px_per_mm']))
    _update_zones(lay)
    lay.slots = [Slot(int(s['index']), f(s['corners_px']), s['u'], s['v'], s['x_mm'], s['y_mm']) for s in d['slots']]
    lay.slots_ok, lay.slots_message = bool(d['slots_ok']), str(d.get('slots_message', ''))
    _check_slots(lay, None)
    return lay


# --------------------------------------------------------------------------- #
# debug 繪圖
# --------------------------------------------------------------------------- #
ZONE_BGR = {'a': (255, 128, 0), 'b': (0, 200, 0), 'c': (0, 220, 255)}


def draw(bgr: np.ndarray, lay: Layout, res: Optional[SceneResult]) -> np.ndarray:
    img = bgr.copy()
    if not lay.ok:
        cv2.putText(img, 'FIELD NOT FOUND', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return img

    def poly(q, col, t=2):
        cv2.polylines(img, [np.round(q).astype(np.int32).reshape(-1, 1, 2)], True, col, t, cv2.LINE_AA)

    def text(s, org, col, sc=0.45):
        org = (int(org[0]), int(org[1]))
        cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1, cv2.LINE_AA)

    w, h = lay.image_size
    for z, q in lay.zones_px.items():
        poly(q, ZONE_BGR[z], 2)
    na = len(res.a_objects) if res and res.ok else '?'
    nb = len(res.b_objects) if res and res.ok else '?'
    for z, lab in (('a', f'A n={na}'), ('b', f'B n={nb}'), ('c', 'C')):
        q = lay.zones_px[z]
        text(lab, (min(max(q[0][0], 0) + 4, w - 60), q[0][1] + 16), ZONE_BGR[z], 0.55)
    # 接縫（world x=0）
    seam = lay.field_to_img([[0, HALF_H], [0, -HALF_H]]).astype(int)
    cv2.line(img, tuple(seam[0]), tuple(seam[1]), (255, 255, 255), 1, cv2.LINE_AA)
    if res and res.ok:
        for o in res.a_objects + res.b_objects:
            cv2.circle(img, (int(o.u), int(o.v)), 3, (0, 255, 255), -1)
            text(f'{o.x_mm:.0f},{o.y_mm:.0f}', (o.u - 22, o.v - 15), (0, 255, 255), 0.38)
    for i, s in enumerate(lay.slots):
        occ = bool(res and res.ok and i < len(res.slot_occupied) and res.slot_occupied[i])
        col = (0, 0, 255) if occ else (0, 220, 255)
        poly(s.corners_px, col, 2)
        text(str(s.index), (s.u - 4, s.v + 5), col, 0.5)
    if lay.coords_source == 'measured':
        status, bad = 'coords: world, measured (ruler)', False
    elif lay.coords_source == 'estimated':
        status, bad = 'coords: world, ESTIMATED calib (trial)', True
    else:
        status, bad = 'coords: world (seam + edges, 600x240)', False
    if lay.slots_ok and math.isfinite(lay.check_mm):
        status += f'  slot chk {lay.check_mm:.0f}mm'
    text(status, (8, 18), (0, 0, 255) if bad else (255, 255, 255), 0.5)
    if not lay.slots_ok:
        text('C slots NOT located', (8, 36), (0, 0, 255), 0.5)
    return img
