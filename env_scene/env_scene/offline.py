"""離線測試：python3 -m env_scene.offline <影像> [輸出debug圖]（不需要 ROS 執行中）。"""
import sys
from pathlib import Path

import cv2

from vision_pkg import calib_io
from env_scene import scene_core as sc


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    img = cv2.imread(sys.argv[1])
    hsv = calib_io.load_hsv_ranges(calib_io.resolve_calib_dir(''))
    p = sc.SceneParams()
    lay = sc.calibrate([img], hsv, p)
    print('標定:', lay.message)
    res, _ = sc.analyze(img, lay, hsv, p)
    print('結果:', res.message)
    for z in ('a', 'b'):
        objs = getattr(res, f'{z}_objects')
        print(f'{z.upper()} 區 {len(objs)} 個')
        for o in objs:
            print(f'   px=({o.u:6.1f},{o.v:6.1f})  mm=({o.x_mm:6.1f},{o.y_mm:6.1f})  d={o.diameter_mm:.1f}mm')
    if lay.slots_ok:
        print('C 區 8 格')
        for s, occ, r in zip(lay.slots, res.slot_occupied, res.slot_ratio):
            print(f'   {s.index}: px=({s.u:6.1f},{s.v:6.1f})  mm=({s.x_mm:6.1f},{s.y_mm:6.1f})  '
                  f'{"佔用" if occ else "空"} ({r:.2f})')
    if len(sys.argv) > 2:
        cv2.imwrite(sys.argv[2], sc.draw(img, lay, res))
    return 0


if __name__ == '__main__':
    sys.exit(main())
