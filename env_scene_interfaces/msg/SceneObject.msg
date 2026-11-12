# 環境相機看到的一個物件（藍色瓶蓋）
string zone          # "a" 或 "b"
float64 u            # 影像像素座標（640x480）
float64 v
float64 x_mm         # world 座標 (mm)：原點 = 場地中心，x 由 A 往 C，y 遠離使用者
float64 y_mm
float64 diameter_mm  # 等面積圓直徑
float64 arm1_x       # arm1_base (m)：x 朝使用者、y 由 A 往 C（EnvScene.arm1_valid=false 時無意義）
float64 arm1_y
float64 arm2_x       # arm2_base (m)（EnvScene.arm2_valid=false 時無意義）
float64 arm2_y
