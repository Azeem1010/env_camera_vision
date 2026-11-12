# 環境相機場景，由 service /vision/env_scene/get_scene 回傳（GetEnvScene.scene）。valid=false 時只看 message。
# 座標採 dual_arm_system 標準場地（docs/COORDINATE_FRAMES.md）：
#   world (mm)：原點 = 場地中心（兩塊板子接縫中點），x 由 A 往 C（-300..300），y 遠離使用者（-120..120）
#   A 區 x=-300..-100、B 區 -100..100、C 區 100..300
#   armN_base (m)：原點 = 底座旋轉軸（arm1 world (-100,125) mm，arm2 (+100,125) mm），x 朝使用者、y 由 A 往 C
std_msgs/Header header
bool valid
string message

string coords_source      # "standard" = 接縫+上下邊（場地 240 mm）；"measured" = 瓶蓋+尺量；"estimated" = 估算值試用
float64 calib_check_mm    # C 區 8 個方格中心 vs 標準格位的最大誤差 (mm)；NaN = 沒找到 8 格
bool arm1_valid           # arm1_x/arm1_y 可用（標定成功且 calib_check_mm <= 門檻）
bool arm2_valid
float64 a_visible_ratio   # A 區在畫面內的比例（相機看不到 A 區最左端）

SceneObject[] a_objects   # A 區瓶蓋（依 y 由遠到近、x 由左到右排序）
SceneObject[] b_objects   # B 區瓶蓋

# C 區 8 格，陣列第 i 項 = 格位 i+1（1、2 最靠手臂，單號靠 B 區）
bool c_slots_valid        # false = 沒找到 8 個方格，以下陣列無意義
bool[8] c_occupied
float64[8] c_fill_ratio   # 格內藍色比例
float64[8] c_u            # 格中心像素
float64[8] c_v
float64[8] c_x_mm         # 格中心 world 座標 (mm)
float64[8] c_y_mm
float64[8] c_arm1_x       # 格中心的手臂基座座標 (m)
float64[8] c_arm1_y
float64[8] c_arm2_x
float64[8] c_arm2_y
int32 c_occupied_count
