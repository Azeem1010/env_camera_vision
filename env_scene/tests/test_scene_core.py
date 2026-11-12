"""用實拍影像測試 scene_core（dual_arm_system 標準場地 600x240，world 原點 = 場地中心）。

tests/data/
  env_live_view2.png            目前相機位置：A 區 8 顆，B/C 空
  env_live_c_caps_on_edges.png  目前相機位置：C 區 8 顆（有的貼著方格、有的壓在場地邊）
  env_live_8caps_in_a.png       舊相機位置（黑色線材壓在方格上）
  env_live_touching_caps.png    舊相機位置，A 區兩顆相黏
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from vision_pkg import calib_io
from env_scene import scene_core as sc

DATA = Path(__file__).parent / 'data'
HSV = calib_io.load_hsv_ranges(Path('/nonexistent'))   # 只用套件內建預設 HSV
P = sc.SceneParams()


def _load(name):
    img = cv2.imread(str(DATA / name))
    assert img is not None, name
    return img


@pytest.fixture(scope='module')
def view2():
    img = _load('env_live_view2.png')
    return img, sc.calibrate([img], HSV, P)


def _paste_cap(img, u, v, r=12):
    out = img.copy()
    cv2.circle(out, (int(u), int(v)), r, (160, 40, 20), -1)   # BGR 深藍
    return out


def test_standard_calibration(view2):
    img, lay = view2
    assert lay.ok and lay.slots_ok, lay.message
    assert lay.coords_source == 'standard'
    assert 1.15 < lay.px_per_mm < 1.35                       # 場地高度 240 mm
    assert lay.check_mm < 6.0                                # 8 格 vs 標準格位
    assert lay.zones_visible['b'] > 0.95 and 0.5 < lay.zones_visible['a'] < 0.95   # A 區最左端在畫面外
    # 接縫（影像中約 u=306）就是 world x=0
    x, y = lay.img_to_field([[306, 220]])[0]
    assert abs(x) < 5 and abs(y) < 60


def test_slot_numbering(view2):
    _, lay = view2
    for s, (sx, sy) in zip(lay.slots, sc.STD_SLOTS_MM):
        assert abs(s.x_mm - sx) < 6 and abs(s.y_mm - sy) < 6, (s.index, s.x_mm, s.y_mm)
    assert lay.slots[0].v < lay.slots[6].v                   # 1 號在影像上方（靠手臂）
    assert lay.slots[0].u < lay.slots[1].u                   # 單號靠 B（左）


def test_a_caps_and_arm_coords(view2):
    img, lay = view2
    res, _ = sc.analyze(img, lay, HSV, P)
    assert len(res.a_objects) == 8 and not res.b_objects
    for o in res.a_objects:
        assert -300 < o.x_mm < -100 and -120 < o.y_mm < 120
        assert 17 < o.diameter_mm < 24                       # 瓶蓋 20 mm
    assert res.slot_occupied == [False] * 8


def test_b_cap_and_c_occupancy(view2):
    img, lay = view2
    s3 = lay.slots[2]
    b_uv = lay.field_to_img([[40.0, 60.0]])[0]               # B 區（避開黑色小方塊）
    img2 = _paste_cap(_paste_cap(img, s3.u, s3.v), *b_uv)
    res, _ = sc.analyze(img2, lay, HSV, P)
    assert len(res.b_objects) == 1
    assert abs(res.b_objects[0].x_mm - 40) < 3 and abs(res.b_objects[0].y_mm - 60) < 3
    assert res.slot_occupied == [i == 2 for i in range(8)]


def test_field_to_arm():
    # arm1 軸心 world (-100, 125)：A/B 分界、手臂側板邊 (−100, 120) -> arm1 (0.005, 0)
    assert np.allclose(sc.field_to_arm(-100.0, 120.0, 'arm1'), (0.005, 0.0))
    # C 區格位 1 標準位置 (135, 50) -> arm2 (0.075, 0.035)：F=75 mm（實測表）
    assert np.allclose(sc.field_to_arm(135.0, 50.0, 'arm2'), (0.075, 0.035))
    # 同一點對 arm1：L = 235 mm（實測表「L 從 arm1 軸心沿長邊朝 C」）
    assert np.allclose(sc.field_to_arm(135.0, 50.0, 'arm1'), (0.075, 0.235))


def test_caps_on_field_edge_and_next_to_slots():
    """瓶蓋壓在場地邊、貼著方格：場地仍要找到；8 格偵測會失敗並明確說明（不猜）。"""
    img = _load('env_live_c_caps_on_edges.png')
    lay = sc.calibrate([img], HSV, P)
    assert lay.ok, lay.message
    assert not lay.slots_ok and '方格旁邊不要放瓶蓋' in lay.slots_message


@pytest.mark.parametrize('name', ['env_live_8caps_in_a.png', 'env_live_touching_caps.png'])
def test_old_camera_position(name):
    """舊相機位置也能標定（接縫不能被黑框邊緣誤導），A 區相黏的瓶蓋要拆開。"""
    img = _load(name)
    lay = sc.calibrate([img], HSV, P)
    assert lay.ok and lay.slots_ok, lay.message
    assert lay.check_mm < 10.0
    res, _ = sc.analyze(img, lay, HSV, P)
    assert len(res.a_objects) == 8


def test_layout_roundtrip(view2):
    _, lay = view2
    lay2 = sc.layout_from_dict(sc.layout_to_dict(lay))
    assert lay2.slots_ok and len(lay2.slots) == 8
    assert np.allclose(lay2.img_to_field([[300, 200]]), lay.img_to_field([[300, 200]]))
    with pytest.raises(ValueError):                          # 舊版（300 mm 場地）不可再用
        sc.layout_from_dict({'version': 1})


def test_calibrate_points_zone_by_zone(tmp_path, view2):
    """一區一區量：A、B、C 分三次執行，點會累加；三區都 >= 3 點才寫出 homography，且能還原「真實」對應。"""
    from env_scene import calibrate_points as cp
    base, lay = view2
    shift = np.array([-4.0, 2.0])              # 模擬「真實」位置與目前估計差一點，尺量標定應修正
    calib = tmp_path / 'calib'
    calib.mkdir()

    def run(img, name, extra_args=()):
        path = tmp_path / f'{name}.png'
        cv2.imwrite(str(path), img)
        caps, _ = cp.detect_all_caps(img, HSV, P)
        caps.sort(key=lambda c: (round(c.cy / 25.0), c.cx))
        true_mm = lay.img_to_field([(c.cx, c.cy) for c in caps]) + shift
        meas = tmp_path / f'{name}.txt'
        meas.write_text('\n'.join(f'{120 - y:.2f} {x:.2f}' for x, y in true_mm))   # front x
        assert cp.main(['--image', str(path), '--measure', str(meas), '--calib-dir', str(calib), '--yes',
                        *extra_args]) == 0
        return caps, true_mm

    def with_caps(mm_list):
        img = base.copy()
        for q in ([[-300, 125], [-95, 125], [-95, -125], [-300, -125]],):   # 把 A 區瓶蓋「拿走」（塗成粉紅）
            cv2.fillConvexPoly(img, np.round(lay.field_to_img(q)).astype(np.int32), (190, 170, 240))
        for uv in lay.field_to_img(mm_list):
            cv2.circle(img, (int(uv[0]), int(uv[1])), 12, (160, 40, 20), -1)
        return img

    run(base, 'a', ['--new'])                                           # A 區 8 顆
    assert not (calib / cp.OUT_NAME).exists()                          # 只有 A，不寫 homography
    run(with_caps([[-70, 90], [70, 90], [70, -90], [-70, -90]]), 'b')
    assert not (calib / cp.OUT_NAME).exists()
    caps, true_mm = run(with_caps([[115, 95], [230, 95], [230, -95], [115, -95]]), 'c')
    pts = calib_io.load_yaml(calib / cp.POINTS_NAME)['points']
    assert {q['zone'] for q in pts} == {'a', 'b', 'c'} and len(pts) == 16
    d = calib_io.load_yaml(calib / cp.OUT_NAME)
    assert d['rms_error_mm'] < 1.0
    sc.apply_measured_homography(lay, np.array(d['H']))
    assert np.allclose(lay.img_to_field([(caps[0].cx, caps[0].cy)])[0], true_mm[0], atol=1.0)
    assert lay.coords_source == 'measured'
