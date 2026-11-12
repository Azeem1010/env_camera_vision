#!/usr/bin/env python3
"""calibrate_points：用「瓶蓋 + 尺量」標定環境相機 像素 -> 場地 mm（給手臂直接用的座標）。

不靠場地四角，所以場地超出畫面也沒關係；量的是瓶蓋頂面高度 = 夾取高度（沒有視差誤差）。

座標採 dual_arm_system 標準場地 world（原點 = 場地中心 = 兩塊板子接縫中點，x 由 A 往 C，y 遠離使用者，場地寬 240 mm）。

量測定義（mm，量到「瓶蓋中心」；量到瓶蓋邊緣請自己加半徑 10 mm）：
  front  瓶蓋中心到「手臂側長邊」（影像上方那條場地邊）的距離
  x      瓶蓋中心到「中間接縫」的距離：C 側為正、A 側為負（A 區、B 區左半要加負號）
  換算：world x = x，y = 120 - front

可以一區一區量（相機不能動）：每次執行量到的點會累加到 <calib_dir>/env_caps_points.yaml。
  第 1 次 --new：瓶蓋放 A 區（4 顆以上，分散四角與中間）→ 量 → 存
  第 2 次：瓶蓋移到 B 區 → 量 → 存（自動接在前面的點後面）
  第 3 次：瓶蓋移到 C 區（放在方格之間或方格裡）→ 量 → 存
  A、B、C 每區都有 >= 3 點後，才會寫出 <calib_dir>/env_field_homography.yaml（只有一區的點，外推到別區會不準）。
  之後重啟 env_scene_node 生效。

每次執行
  1. 手臂移出畫面，瓶蓋彼此不要相碰。
  2. ros2 run env_scene calibrate_points        （第一次加 --new 清掉舊點）
     -> 存 <calib_dir>/env_caps_numbered.png（含編號），終端機列出每顆的像素位置
  3. 依編號輸入 "front x"（mm），不想用的那顆直接 Enter 跳過；或事先寫檔用 --measure（每行 "front x"）
  4. 看誤差與各區點數，輸入 y 存檔

    ros2 run env_scene calibrate_points --new                     # 開始新的一組（清掉舊點）
    ros2 run env_scene calibrate_points --measure caps_mm.txt
    ros2 run env_scene calibrate_points --image env_caps_raw.png --measure caps_mm.txt    # 離線
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from vision_pkg import calib_io
from vision_pkg.vision_core.cap_detect import detect_caps
from vision_pkg.vision_core.color_zone import hsv_mask
from vision_pkg.vision_core.geometry import apply_homography, compute_homography, polygon_mask

from env_scene import scene_core as sc

OUT_NAME = 'env_field_homography.yaml'
POINTS_NAME = 'env_caps_points.yaml'
MIN_PER_ZONE = 3


def grab_frames(topic: str, n: int, timeout_s: float = 10.0):
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    from vision_pkg.node_common import image_msg_to_bgr
    rclpy.init()
    node = rclpy.create_node('env_calibrate_points')
    got = []
    node.create_subscription(Image, topic, lambda m: got.append(image_msg_to_bgr(m)), qos_profile_sensor_data)
    t0 = time.monotonic()
    while len(got) < n and time.monotonic() - t0 < timeout_s:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()
    return got


def detect_all_caps(img, hsv, p: sc.SceneParams):
    q, why = sc.find_field_quad(img, hsv['pink'], p, hsv['blue'])
    if q is None:
        raise SystemExit(f'找不到場地：{why}')
    px_per_mm = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2 / sc.FIELD_H_MM
    lay = sc.Layout(True, '', (img.shape[1], img.shape[0]), px_per_mm=float(px_per_mm))
    blue = hsv_mask(img, hsv['blue'], p.obj_mask)
    # 只找場地範圍內（上下各外擴 15 mm，瓶蓋可能壓在邊上），排除場地外的藍色物品
    c = q.mean(axis=0)
    grow = 1.0 + 30.0 / sc.FIELD_H_MM
    inside = polygon_mask(blue.shape, c + (q - c) * np.array([1.05, grow]), 0)
    caps, rej = detect_caps(cv2.bitwise_and(blue, inside), sc._cap_params(lay, p))
    return caps, rej


def read_measure_file(path):
    out = []
    for ln in Path(path).read_text(encoding='utf-8').splitlines():
        ln = ln.split('#')[0].strip()
        if not ln:
            out.append(None)
            continue
        v = [float(t) for t in ln.replace(',', ' ').split()]
        if len(v) != 2:
            raise SystemExit(f'每行要兩個數字 "front x"：{ln!r}')
        out.append((v[0], v[1]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--image', help='用存好的影像（不訂閱 topic）')
    ap.add_argument('--topic', default='/env_cam/image_raw')
    ap.add_argument('--frames', type=int, default=10, help='平均幾張影格')
    ap.add_argument('--measure', help='量測檔：每行 "front x"（mm），順序同編號；空行 = 跳過')
    ap.add_argument('--calib-dir', default='')
    ap.add_argument('--yes', action='store_true', help='不詢問直接存檔')
    ap.add_argument('--new', action='store_true', help='清掉之前累加的點，從頭開始')
    args = ap.parse_args([a for a in (argv or sys.argv[1:]) if not a.startswith('__')])

    calib_dir = calib_io.resolve_calib_dir(args.calib_dir)
    hsv = calib_io.load_hsv_ranges(calib_dir)
    p = sc.SceneParams()
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f'讀不到 {args.image}')
    else:
        frames = grab_frames(args.topic, args.frames)
        if not frames:
            raise SystemExit(f'收不到 {args.topic} 的影像（usb_cam 有開嗎？）')
        img = np.mean(np.stack(frames), axis=0).astype(np.uint8)
        calib_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(calib_dir / 'env_caps_raw.png'), img)

    caps, rej = detect_all_caps(img, hsv, p)
    caps.sort(key=lambda c: (round(c.cy / 25.0), c.cx))       # 由上到下、由左到右編號
    vis = img.copy()
    for i, c in enumerate(caps):
        cv2.circle(vis, (int(c.cx), int(c.cy)), int(c.radius) + 2, (0, 255, 255), 2)
        for col, th in (((0, 0, 0), 4), ((0, 255, 255), 1)):
            cv2.putText(vis, str(i + 1), (int(c.cx) - 6, int(c.cy) - int(c.radius) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, th, cv2.LINE_AA)
    numbered = calib_dir / 'env_caps_numbered.png'
    cv2.imwrite(str(numbered), vis)
    print(f'找到 {len(caps)} 顆瓶蓋（編號圖：{numbered}）')
    for i, c in enumerate(caps):
        print(f'  #{i + 1}: 像素 ({c.cx:.1f}, {c.cy:.1f})')
    for r in rej:
        print(f'  （排除 @({r.cx:.0f},{r.cy:.0f})：{r.reject_reason}）')
    if not caps:
        raise SystemExit('畫面裡沒有找到瓶蓋')

    # 之前累加的點（同一相機位置）
    pts_path = calib_dir / POINTS_NAME
    old = []
    if pts_path.exists() and not args.new:
        d = calib_io.load_yaml(pts_path)
        if list(d.get('image_size', [])) != [img.shape[1], img.shape[0]]:
            raise SystemExit(f'{pts_path} 的影像大小不同；請加 --new 重新開始')
        old = d.get('points', [])
        print(f'\n已累加 {len(old)} 點（{_zone_summary(old)}）；這次的點會接在後面。要重來請加 --new')

    if args.measure:
        meas = read_measure_file(args.measure)
        if len(meas) != len(caps):
            raise SystemExit(f'量測檔有 {len(meas)} 行，但找到 {len(caps)} 顆')
    else:
        print('\n依編號輸入 "front x"（mm）：front = 到手臂側長邊，x = 到中間接縫（A 側為負）；直接 Enter = 跳過')
        meas = []
        for i in range(len(caps)):
            while True:
                t = input(f'  #{i + 1}: ').strip()
                if not t:
                    meas.append(None)
                    break
                try:
                    a, b = (float(x) for x in t.replace(',', ' ').split())
                    meas.append((a, b))
                    break
                except ValueError:
                    print('    格式：兩個數字，例如 "120 85"')

    new = []
    for c, m in zip(caps, meas):
        if m is not None:
            x, y = m[1], sc.HALF_H - m[0]                    # world x = x, y = 120 - front
            new.append({'pixel': [float(c.cx), float(c.cy)], 'field_mm': [float(x), float(y)],
                        'zone': _zone_of(x)})
    if not new:
        raise SystemExit('這次沒有輸入任何點')
    # 同一位置（< 6 px）重量的點：用新的取代舊的
    old = [o for o in old if all(np.hypot(o['pixel'][0] - n['pixel'][0], o['pixel'][1] - n['pixel'][1]) >= 6
                                 for n in new)]
    allp = old + new
    print(f'\n這次 {len(new)} 點（{_zone_summary(new)}），合計 {len(allp)} 點（{_zone_summary(allp)}）')

    H, rms = None, float('nan')
    if len(allp) >= 4:
        px = [q['pixel'] for q in allp]
        mm = [q['field_mm'] for q in allp]
        H, err = compute_homography(px, mm)
        print('每點誤差（mm）：')
        for q, e in zip(allp, err):
            mark = '  <-- 這次' if q in new else ''
            print(f"  {q['zone'].upper()} 像素 ({q['pixel'][0]:6.1f},{q['pixel'][1]:6.1f}) -> "
                  f"量測 ({q['field_mm'][0]:6.1f},{q['field_mm'][1]:6.1f})  誤差 {e:.1f}{mark}")
        rms = float(np.sqrt(np.mean(err ** 2)))
        print(f'RMS {rms:.2f} mm，最大 {err.max():.2f} mm（目標 < 3 mm；某點特別大多半是量錯或編號對錯）')
        if len(allp) >= 6:   # 留一驗證：每點用其他點算出的 H 預測
            loo = []
            for k in range(len(px)):
                Hk, _ = compute_homography([q for j, q in enumerate(px) if j != k],
                                           [q for j, q in enumerate(mm) if j != k])
                loo.append(np.linalg.norm(apply_homography(Hk, [px[k]])[0] - mm[k]))
            print(f'留一驗證誤差：平均 {np.mean(loo):.2f} mm，最大 {np.max(loo):.2f} mm（這才是新位置的實際精度）')

    if not args.yes and input('存檔？[y/N] ').strip().lower() != 'y':
        print('未存檔（這次的點沒有加入）')
        return 0
    calib_io.save_yaml(pts_path, {'image_size': [img.shape[1], img.shape[0]], 'points': allp},
                       '# calibrate_points 累加的量測點（pixel -> field_mm）\n')
    print(f'已累加到 {pts_path}')
    counts = {z: sum(1 for q in allp if q['zone'] == z) for z in 'abc'}
    missing = [z.upper() for z in 'abc' if counts[z] < MIN_PER_ZONE]
    if missing or H is None:
        print(f'還缺：{"、".join(f"{z} 區（{counts[z.lower()]}/{MIN_PER_ZONE}）" for z in missing)}。'
              '把瓶蓋移到下一區，再執行一次 calibrate_points（不要加 --new）')
        return 0
    out = calib_dir / OUT_NAME
    calib_io.save_yaml(out, {
        'valid': True, 'unit': 'mm', 'image_size': [img.shape[1], img.shape[0]],
        'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'rms_error_mm': rms, 'H': H.tolist(),
        'points': allp,
    }, '# 環境相機：影像像素 -> world mm（標準場地，原點 = 場地中心）。由 calibrate_points 產生。\n')
    print(f'A/B/C 都量好了，已寫出 {out}；重啟 env_scene_node 生效')
    return 0


def _zone_of(x_mm: float) -> str:
    return 'a' if x_mm < -100.0 else ('b' if x_mm < 100.0 else 'c')


def _zone_summary(pts) -> str:
    return ' '.join(f"{z.upper()}={sum(1 for q in pts if q['zone'] == z)}" for z in 'abc')

if __name__ == '__main__':
    sys.exit(main())
