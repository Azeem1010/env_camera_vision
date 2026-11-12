# env_camera_vision — 環境相機場景（A / B / C）

參考 [three-camera-vision-ros2](https://github.com/Azeem1010/three-camera-vision-ros2) 的 `env_vision_node`，
針對「相機移到使用者這側」的新場地另外做一個節點 `env_scene_node`。**不修改** 原本的 `vision_pkg` / `vision_interfaces`，
兩個節點可以同時跑（都訂閱 `/env_cam/image_raw`）。

場地採用 [dual_arm_system](https://github.com/xian1022/dual_arm_system) 的標準定義（`docs/COORDINATE_FRAMES.md`、`scripts/field_layout.py`）：

```
影像左 ──────────────────────────────────────────────────────── 影像右
|  A 區 x=-300..-100  |   B 區 -100..100   |  C 區 100..300（8 格）|   y=+120 手臂側
───────────────────── 使用者這側（影像下方）y=-120 ─────────────────────
      arm1 軸心 (-100, 125)        arm2 軸心 (100, 125)        單位 mm

C 區格位（1、2 最靠手臂，單號靠 B）：x = 135 / 170，y = 50 / 15 / -20 / -55
```

- **world 座標（mm）**：原點 = 場地中心（兩塊板子的接縫中點），x 由 A 往 C，y 遠離使用者；場地 600 × 240 mm
- **armN_base（m）**：原點 = 底座旋轉軸，x 朝使用者（= −y_world），y 由 A 往 C（= +x_world）
- **標定**：y = ±120 取場地上下長邊（逐欄穩健擬合），x = 0 取中間接縫，比例由場地高度 240 mm 換算。
  相機看不到 A 區最左端（約 25% 在畫面外）也沒關係。標定後用 C 區 8 個方格對照標準格位驗證
  （目前相機位置：平均 2.1 mm、最大約 3 mm）；驗證超過 10 mm 或找不到 8 格時 `arm*_valid=false`。

## 輸出

| 介面 | 內容 |
|---|---|
| srv `/vision/env_scene/get_scene`（`GetEnvScene`） | 最新一筆場景（等新結果，含逾時；結果只走 service，不發布 topic） |
| srv `/vision/env_scene/calibrate`（`std_srvs/Trigger`） | 重新找場地、黑框、8 格並存檔 |

- `a_objects[]` / `b_objects[]`：每個瓶蓋的像素 `u,v`、world `x_mm,y_mm`、`arm1_x/y`、`arm2_x/y`
- `c_occupied[8]`、`c_fill_ratio[8]`、`c_u/c_v[8]`、`c_x_mm/c_y_mm[8]`：第 i 項 = 格位 i+1
- `coords_source`、`calib_check_mm`（8 格驗證誤差）、`a_visible_ratio`

## 建置與執行

```bash
# 兩個套件以 symlink 放進 ~/vision_ws/src（已做好）
cd ~/vision_ws && colcon build --symlink-install --packages-select env_scene_interfaces env_scene
source install/setup.bash
ros2 run env_scene env_scene_node --ros-args --params-file install/env_scene/share/env_scene/config/env_scene.yaml

ros2 service call /vision/env_scene/calibrate std_srvs/srv/Trigger          # C 區 8 格淨空、線材移開
ros2 service call /vision/env_scene/get_scene env_scene_interfaces/srv/GetEnvScene

# 離線（單張圖片）
python3 -m env_scene.offline 圖.png 輸出debug.png        # 在 env_scene/ 目錄下
cd env_scene && python3 -m pytest -q
```

## 選用：瓶蓋 + 尺量標定（進一步提高精度）

標準標定（接縫 + 上下邊）已經能用；要更準（尤其 A 區，目前沒有驗證點）時，
`calibrate_points` 直接擬合「像素 → world mm」，量的是瓶蓋頂面（夾取高度），沒有視差誤差。

可以**一區一區量**（相機不能動），每次的點會累加到 `env_caps_points.yaml`：

1. 瓶蓋放 A 區（≥ 4 顆，分散四角與中間，不互碰；手臂移出畫面）→ `ros2 run env_scene calibrate_points --new`
2. 瓶蓋移到 B 區 → `ros2 run env_scene calibrate_points`（不加 `--new`）
3. 瓶蓋移到 C 區（方格之間或方格裡）→ 再執行一次
4. 每次看 `~/.ros/vision_calib/env_caps_numbered.png` 的編號，輸入 `front x`（mm）：
   front = 瓶蓋中心到手臂側長邊；x = 到中間接縫（A 側為負）
5. A/B/C 每區都 ≥ 3 點後才寫出 `env_field_homography.yaml`（只有一區的點外推到別區會不準）→ 重啟 `env_scene_node`

之後 `coords_source="measured"`。

## 做法

1. **場地上下邊**：粉紅（含放在上面的瓶蓋）遮罩，逐欄找上下緣，穩健擬合直線；連在場地邊上的其他粉紅泡棉只影響少數欄，會被排除。
2. **接縫（x=0）**：只在左右兩邊都是粉紅的位置找垂直邊緣，排除黑框膠帶、黑色小方塊等干擾。
3. **C 區 8 格**：非粉紅的亮色方塊（白紙）或藍色，取凸包，依 world 座標排成 4 排 × 2 個；之後格位固定，
   佔用 = 格內（內縮 12%）藍色比例 ≥ 0.25，連續 3 張才切換。方格旁邊有瓶蓋時標定會失敗（不猜）。
4. **瓶蓋**：藍色、圓度與面積篩選（直徑 20 mm），相黏的依面積用 k-means 拆開；只找場地範圍內。

## 已知問題 / 待處理

- A 區最左端約 25% 在畫面外，放在那裡的瓶蓋看不到。
- 標定（開機或呼叫 `calibrate`）時，C 區方格旁邊不要放瓶蓋；方格裡可以有瓶蓋。
- `arm2_base` 軸向採 COORDINATE_FRAMES.md 的建議值（同手臂 1），該文件標為待確認。
- 尚未做實際尺量標定（`calibrate_points`）；目前精度以 8 格驗證（約 3 mm）為準，A 區沒有直接驗證點。
- B 區只偵測藍色瓶蓋；B 區那塊黑色小方塊不算物件。
