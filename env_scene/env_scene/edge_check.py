"""即時檢查場地是否完整入鏡（調整相機位置用）。

ros2 run env_scene edge_check     視窗按 q / Esc 結束
每張影格重新找粉紅場地四角，顯示四角離影像邊界的距離；左右各 >= min_margin_px 且上下也沒貼邊時顯示綠色 OK。
"""
import sys

import cv2
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from vision_pkg import calib_io
from vision_pkg.node_common import image_msg_to_bgr
from env_scene import scene_core as sc


def main():
    min_margin = 15
    rclpy.init()
    node = rclpy.create_node('env_edge_check')
    hsv = calib_io.load_hsv_ranges(calib_io.resolve_calib_dir(''))
    p = sc.SceneParams()
    latest = {}
    node.create_subscription(Image, '/env_cam/image_raw', lambda m: latest.__setitem__('img', m),
                             qos_profile_sensor_data)
    last_print = ''
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            msg = latest.pop('img', None)
            if msg is None:
                continue
            img = image_msg_to_bgr(msg)
            h, w = img.shape[:2]
            q, why = sc.find_field_quad(img, hsv['pink'], p, hsv['blue'])
            out = img.copy()
            if q is None:
                text, ok = f'FIELD NOT FOUND: {why}', False
            else:
                left = float(q[:, 0].min())
                right = float(w - 1 - q[:, 0].max())
                top = float(q[:, 1].min())
                bottom = float(h - 1 - q[:, 1].max())
                ok = min(left, right) >= min_margin and min(top, bottom) >= 3
                cv2.polylines(out, [np.round(q).astype(np.int32).reshape(-1, 1, 2)], True,
                              (0, 200, 0) if ok else (0, 0, 255), 2, cv2.LINE_AA)
                for (x, y) in q:
                    cv2.circle(out, (int(x), int(y)), 5, (0, 255, 255), -1)
                text = f'L={left:.0f}px R={right:.0f}px T={top:.0f}px B={bottom:.0f}px  ' + \
                       ('OK' if ok else f'CLIPPED (need L,R >= {min_margin})')
            col = (0, 200, 0) if ok else (0, 0, 255)
            cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
            if text != last_print:
                print(text, flush=True)
                last_print = text
            cv2.imshow('env_edge_check', out)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
