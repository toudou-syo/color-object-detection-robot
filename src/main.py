import cv2
import numpy as np
import threading
import time
import queue
import math
import sys
import os
import signal
from enum import IntEnum, auto

# --- Dobot / CyberPi 関連インポート ---
from cyberpi import mbot2, display, controller, table, ultrasonic2, cyberpi
import DobotDllType as dType

# --- ユーザー環境モジュール (Dobot用) ---
try:
    import cameraSetting as camset
    import myDobotModule_rpi as dobot
    from TransformationMatrix import MATRIX
    from common import *
    from PyQt5.QtWidgets import QApplication
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")

# ==========================================
# グローバル設定・変数定義
# ==========================================

app = QApplication(sys.argv)

SKIP_CALIBRATION = False
SKIP_PICK = False # Pickフェーズをスキップするかどうか

# --- mBot用設定 ---
CAMERA_ID_MBOT = 0     
CAMERA_ID_PICK = 2    
TARGET_W, TARGET_H = 320, 240
PROCESS_FPS = 20
CONTROL_PERIOD = 0.05

# 障害物検知ライン (画面下部200pxより下で反応)
FORWARD_THRESHOLD = 200  

DEBUG_MODE = True

# mBot 制御用グローバル
frame_queue = queue.Queue(maxsize=1)
display_queue = queue.Queue(maxsize=1) # 表示用
direction_lock = threading.Lock()
latest_direction = "forward"
stop_event = threading.Event()

# デバッグ表示用共有変数
debug_info = {
    "action": "Init", "error_deg": 0.0, "dist_cm": 0.0,
    "us_front": 0.0, "us_left": 0.0, "us_right": 0.0,
    "enc_L_delta": 0.0, "enc_R_delta": 0.0
}

# mBot 位置推定用
WHEEL_DIAMETER = 6.5
WHEEL_BASE = 18.45
prev_left_encoder = 0
prev_right_encoder = 0
current_x = 0
current_y = 0
current_theta = 90 * math.pi / 180

# エンコーダ補正係数
ENCODER_LEFT_CORRECTION_FACTOR = 1.0
ENCODER_RIGHT_CORRECTION_FACTOR = 0.96 

# mBot 目的地 (Phase 2のラフなゴール)
GOAL_X = 70
GOAL_Y = 150
GOAL_TOLERANCE = 2

# mBot カメラ認識値
forward_y = 0; left_y = 0; right_y = 0

# --- Dobot用設定 ---
R_HOME_OFFSET = 0
MATRIX = np.array(MATRIX) 
VIRTUAL_ORIGIN_X = 0.0
VIRTUAL_ORIGIN_Y = 0.0
virtual_origin_set = False

# ジョイント可動域条件
J1_LIMIT_MIN, J1_LIMIT_MAX = -125, 125
J2_LIMIT_MIN, J2_LIMIT_MAX = -5, 95
J3_LIMIT_MIN, J3_LIMIT_MAX = -15, 85
J4_LIMIT_MIN, J4_LIMIT_MAX = -150, 150

# ワークスペース半径制限 (mm)
MIN_RADIUS = 120; MAX_RADIUS = 315 

# --- opencv_setting.py 由来の定数 ---
MIN_AREA_SIZE = 200; MAX_AREA_SIZE = 400      
MIN_PICK_AREA_SIZE = 500; MAX_PICK_AREA_SIZE = 20000 

dobot_red = [150, 0]; dobot_blue = [300, 0]
dobot_yellow = [150, -100]; dobot_green = [300, -100]

class Color(IntEnum):
    RED = 0; BLUE = auto(); GREEN = auto(); YELLOW = auto()

# ==========================================
# placetest3.py 由来の設定・定数
# ==========================================
TARGET_ID = 0          # ArUcoマーカーID
Z_HEIGHT = 173         
RELEASE_Z = -30        
WAIT_Z = 173           

# mbot2 モーター特別仕様設定
MBOT_LEFT_POLARITY  = 1   
MBOT_RIGHT_POLARITY = -1  

# 速度・インチング制御パラメータ
MBOT_MAX_SPEED = 20         
MBOT_INCH_POWER = 85        
MBOT_INCH_POWER_WEAK = 75   

# 秒数制御用の設定
TIME_ALIGN_LONG = 0.10; TIME_ALIGN_MID = 0.05; TIME_ALIGN_TINY = 0.02
TIME_SEARCH_STEP = 0.1; TIME_SEARCH_WAIT = 0.2

# 閾値
INCH_THRESHOLD_LARGE = 300 
INCH_THRESHOLD_MEDIUM = 150 

# 目標エリアと許容範囲
MBOT_TARGET_AREA = 85000       # 80000-90000の中間
MBOT_AREA_TOLERANCE = 5000     # ±5000 (80k-90k)
MBOT_LOST_AS_GOAL_THRESHOLD = 80000 

MBOT_TOLERANCE_X = 80       

MBOT_ALIGN_THRESHOLD_FAR = 80
MBOT_ALIGN_THRESHOLD_NEAR = 180
MBOT_P_GAIN_FORWARD = 0.0008 

# Phase 2 -> 3 移行条件
# ▼▼▼【修正】閾値を 1000 に変更 ▼▼▼
PHASE_TRANSITION_AREA_THRESHOLD = 100 
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
SAFE_TRANSITION_DIST = 15.0

# ArUco設定
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters_create()

# ==========================================
# 共通 / ユーティリティ関数
# ==========================================

def normalize_angle(angle_rad):
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))

def transform_coordinate(pos_x, pos_y):
    pos = np.array([[[pos_x, pos_y]]], dtype='float32')
    tp = cv2.perspectiveTransform(pos, MATRIX)
    return int(tp[0][0][0]), int(tp[0][0][1])

def set_virtual_origin():
    global VIRTUAL_ORIGIN_X, VIRTUAL_ORIGIN_Y, virtual_origin_set
    try:
        pose = dobot.dType.GetPose(dobot.api)
        VIRTUAL_ORIGIN_X = pose[0]
        VIRTUAL_ORIGIN_Y = pose[1]
        virtual_origin_set = True
        print(f"[Virtual Origin] ({VIRTUAL_ORIGIN_X:.1f}, {VIRTUAL_ORIGIN_Y:.1f}) set")
    except Exception as e:
        print(f"[Virtual Origin ERROR] {e}")

def check_safe_target(x, y, z_target=0):
    r_dist = math.sqrt(x**2 + y**2)
    if not (MIN_RADIUS < r_dist < MAX_RADIUS):
        print(f"[LIMIT] Out of Radius Range: R={r_dist:.1f}")
        return False
    j1_angle = math.degrees(math.atan2(y, x))
    if not (J1_LIMIT_MIN <= j1_angle <= J1_LIMIT_MAX):
        print(f"[LIMIT] J1 Angle Out of Range: {j1_angle:.1f}")
        return False
    if r_dist > 300 and z_target < -10:
        print(f"[LIMIT] Too far to go low: R={r_dist:.1f}, Z={z_target}")
        return False
    return True

def find_specific_color(color_name, frame, edframe, low, high, min_area=MIN_AREA_SIZE, max_area=MAX_AREA_SIZE):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ex_img = cv2.inRange(hsv, low, high)
    contours, hierarchy = cv2.findContours(ex_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < min_area or max_area < area: continue
            
        X = np.array(contour, dtype=np.float64).reshape((contour.shape[0], contour.shape[2]))
        mean, eigenvectors = cv2.PCACompute(X, mean=np.array([], dtype=np.float64), maxComponents=1)
        x, y, width, height = cv2.boundingRect(contour)
        cv2.rectangle(edframe, (x, y), (x+width, y+height), (0,0,255), thickness=1)
        mp_x = int(mean[0][0])
        mp_y = int(mean[0][1])
        cv2.drawMarker(edframe, (mp_x, mp_y), (0,0,255), cv2.MARKER_TILTED_CROSS, thickness = 1)
        
        vx, vy = eigenvectors[0][0], eigenvectors[0][1]
        angle_deg = np.degrees(np.arctan2(vy, vx))
        
        label = " Mid : (" + str(mp_x) + ", " + str(mp_y) + ")"
        cv2.putText(edframe, label, (x+width, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        return (mp_x, mp_y, angle_deg)
    return None

# ==========================================
# placetest3.py 由来のヘルパー関数
# ==========================================

def get_aruco_info(frame):
    """ArUcoマーカー検出"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    
    if ids is not None:
        for index, marker_id in enumerate(ids):
            if marker_id[0] == TARGET_ID:
                c = corners[index][0]
                cx = int(np.mean(c[:, 0]))
                cy = int(np.mean(c[:, 1]))
                area = cv2.contourArea(c)
                return (cx, cy, area), frame
    return None, frame

def drive_mbot_placetest(left_val, right_val):
    """placetest3用: 速度指定(RPM)"""
    final_left = left_val * MBOT_LEFT_POLARITY
    final_right = right_val * MBOT_RIGHT_POLARITY
    mbot2.drive_speed(final_left, final_right)
    return final_left, final_right

def drive_mbot_power_placetest(left_val, right_val):
    """placetest3用: パワー指定(%)"""
    final_left = left_val * MBOT_LEFT_POLARITY
    final_right = right_val * MBOT_RIGHT_POLARITY
    mbot2.drive_power(final_left, final_right)
    return final_left, final_right

def clamp_speed(val, min_val, max_val):
    if abs(val) < 1.0: return 0
    if val > 0:
        if val < min_val: return min_val
        if val > max_val: return max_val
        return val
    else:
        if val > -min_val: return -min_val
        if val < -max_val: return -max_val
        return val

# ==========================================
# Calibration Function
# ==========================================
def run_camera_calibration():
    print("\n=== Camera Calibration (opencv_setting.py) ===")
    cap = cv2.VideoCapture(CAMERA_ID_PICK, cv2.CAP_V4L2)
    camset.camera_get(cv2, cap)
    calibrated_matrix = None

    while True:
        ret, frame = cap.read()
        if not ret: continue
        _, edframe = cap.read()
        cv2.imshow('Raw Frame', frame)

        res_red = find_specific_color("RED", frame, edframe, RED_LOW_COLOR, RED_HIGH_COLOR)
        res_blue = find_specific_color("BLUE", frame, edframe, BLUE_LOW_COLOR, BLUE_HIGH_COLOR)
        res_yellow = find_specific_color("YELLOW", frame, edframe, YELLOW_LOW_COLOR, YELLOW_HIGH_COLOR)
        res_green = find_specific_color("GREEN", frame, edframe, GREEN_LOW_COLOR, GREEN_HIGH_COLOR)

        pos_red = res_red[:2] if res_red else None
        pos_blue = res_blue[:2] if res_blue else None
        pos_yellow = res_yellow[:2] if res_yellow else None
        pos_green = res_green[:2] if res_green else None

        M_curr = None
        if pos_red and pos_blue and pos_yellow and pos_green:
            h, w = frame.shape[:2]
            pts_cam = np.float32([pos_red, pos_yellow, pos_blue, pos_green])
            pts_dobot = np.float32([dobot_red, dobot_yellow, dobot_blue, dobot_green])
            M_curr = cv2.getPerspectiveTransform(pts_cam, pts_dobot)
            cv2.warpPerspective(frame, M_curr, (h, w)) 

        cv2.imshow('Edited Frame', edframe)
        k = cv2.waitKey(1)

        if k == 27: 
            cap.release(); cv2.destroyAllWindows(); return None 
        elif k == ord('c'):
            g = input("gain     : "); e = input("exposure : ")
            camset.camera_set(cv2, cap, gain = float(g), exposure = float(e))
            camset.camera_get(cv2, cap)
        elif k == ord('s'):
            if M_curr is not None:
                calibrated_matrix = M_curr; break 
            else:
                print(">> Cannot save: Not all colors detected.")

    cap.release(); cv2.destroyAllWindows()
    return calibrated_matrix

# ==========================================
# Phase 1: Dobot Pick
# ==========================================
def run_pick_phase():
    print("\n=== PHASE 1: DOBOT PICK START ===")
    global MATRIX 
    
    dobot.initialize()
    dType.ClearAllAlarmsState(dobot.api)
    time.sleep(0.5) 
    
    dType.SetPTPJointParams(dobot.api, 200, 200, 200, 200, 200, 200, 200, 200, 0)
    dType.SetPTPCommonParams(dobot.api, 100, 100, 0)
    
    dType.SetHOMECmdEx(dobot.api, 0, True)
    
    dType.SetEndEffectorGripperEx(dobot.api, True, True)
    time.sleep(1) 
    dType.SetEndEffectorGripperEx(dobot.api, False, True)
    time.sleep(1) 

    if not SKIP_CALIBRATION:
        p = dType.GetPose(dobot.api)
        dType.SetPTPCmdEx(dobot.api, 1, p[0], p[1], -85, p[3], 0) 
        dobot.wait(5.0)

    print("Moving to Camera Position (Z=173)...")
    p = dType.GetPose(dobot.api)
    dType.SetPTPCmdEx(dobot.api, 1, p[0], p[1], 173, p[3], 0) 
    dobot.wait(1.0)

    if not SKIP_CALIBRATION:
        new_matrix = run_camera_calibration()
        if new_matrix is not None: MATRIX = new_matrix 
    
    set_virtual_origin()
    cap = cv2.VideoCapture(CAMERA_ID_PICK, cv2.CAP_V4L2)
    camset.camera_get(cv2, cap)
    detected_color_enum = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            _, edframe = cap.read()

            res_red = find_specific_color("RED", frame, edframe, RED_LOW_COLOR, RED_HIGH_COLOR, MIN_PICK_AREA_SIZE, MAX_PICK_AREA_SIZE)
            res_blue = find_specific_color("BLUE", frame, edframe, BLUE_LOW_COLOR, BLUE_HIGH_COLOR, MIN_PICK_AREA_SIZE, MAX_PICK_AREA_SIZE)
            res_green = find_specific_color("GREEN", frame, edframe, GREEN_LOW_COLOR, GREEN_HIGH_COLOR, MIN_PICK_AREA_SIZE, MAX_PICK_AREA_SIZE)
            res_yellow = find_specific_color("YELLOW", frame, edframe, YELLOW_LOW_COLOR, YELLOW_HIGH_COLOR, MIN_PICK_AREA_SIZE, MAX_PICK_AREA_SIZE)

            cv2.imshow('Pick Camera', edframe)
            k = cv2.waitKey(1)

            target_pos = None
            if k == 27: return None
            elif k == ord('r') and res_red: target_pos = res_red; detected_color_enum = Color.RED
            elif k == ord('b') and res_blue: target_pos = res_blue; detected_color_enum = Color.BLUE
            elif k == ord('g') and res_green: target_pos = res_green; detected_color_enum = Color.GREEN
            elif k == ord('y') and res_yellow: target_pos = res_yellow; detected_color_enum = Color.YELLOW

            if target_pos is not None:
                x_cam, y_cam, angle = target_pos
                mx, my = transform_coordinate(x_cam, y_cam)
                PICK_Z = -40 
                if not check_safe_target(mx, my, PICK_Z):
                    print("!!! WARNING !!! Unreachable or Limit Violation.")
                    continue

                mr = np.clip(-angle - R_HOME_OFFSET, J4_LIMIT_MIN, J4_LIMIT_MAX)
                dobot.move(mx, my, 0, mr)
                dobot.gripper(True, False)
                dobot.move(mx, my, -90, mr)
                dobot.wait(0.5)
                dobot.gripper(True, True)
                dobot.wait(1.0)
                dobot.move(mx, my, 0, mr)
                dobot.wait(0.5)

                p = dType.GetPose(dobot.api)
                dType.SetPTPCmdEx(dobot.api, 1, p[0], p[1], p[2] + 10, p[3], 0)
                p = dType.GetPose(dobot.api)
                dType.SetPTPCmdEx(dobot.api, 4, -125, -5, 35, p[7], 0)
                dobot.finalize() 
                break
    except Exception as e:
        print(f"[Pick Error] {e}")
        return None
    finally:
        if cap.isOpened(): cap.release()
        cv2.destroyAllWindows()
    
    return int(detected_color_enum)

# ==========================================
# Phase 2: mBot Transport
# ==========================================
def mbot_update_position():
    global prev_left_encoder, prev_right_encoder
    global current_x, current_y, current_theta

    raw_left = mbot2.EM_get_angle('em1')
    raw_right = mbot2.EM_get_angle('em2')

    if raw_left is None or raw_right is None: return

    left_encoder = raw_left
    right_encoder = -raw_right

    delta_left = left_encoder - prev_left_encoder
    delta_right = right_encoder - prev_right_encoder
    prev_left_encoder = left_encoder
    prev_right_encoder = right_encoder

    debug_info["enc_L_delta"] = delta_left
    debug_info["enc_R_delta"] = delta_right

    delta_left_corrected = delta_left * ENCODER_LEFT_CORRECTION_FACTOR
    delta_right_corrected = delta_right * ENCODER_RIGHT_CORRECTION_FACTOR

    distance_left = delta_left_corrected / 17.55
    distance_right = delta_right_corrected / 17.55
    distance_center = (distance_left + distance_right) / 2.0

    delta_theta_local = (distance_right - distance_left) / WHEEL_BASE
    current_theta += delta_theta_local
    current_theta = normalize_angle(current_theta)

    dx = distance_center * math.cos(current_theta)
    dy = distance_center * math.sin(current_theta)
    current_x += dx
    current_y += dy

def mbot_heading_to_goal():
    dx = GOAL_X - current_x
    dy = GOAL_Y - current_y
    return math.atan2(dy, dx)

def mbot_safe_reset():
    global prev_left_encoder, prev_right_encoder, current_x, current_y, current_theta
    mbot2.drive_power(0, 0)
    mbot2.EM_reset_angle("all")
    display.clear()
    prev_left_encoder = 0; prev_right_encoder = 0
    current_x = 0.0; current_y = 0.0
    current_theta = 90 * math.pi / 180

def mbot_capture_thread():
    cap = cv2.VideoCapture(CAMERA_ID_MBOT, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        stop_event.set()
        return

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1); continue
        
        meta = {"ts": time.time()}
        try:
            frame_queue.put_nowait((frame, meta))
        except queue.Full:
            try: _ = frame_queue.get_nowait(); frame_queue.put_nowait((frame, meta))
            except: pass
        time.sleep(0.005)
    
    print("[Capture Thread] Releasing Camera...")
    cap.release()

def mbot_processing_thread():
    global latest_direction, forward_y, left_y, right_y
    target_period = 1.0 / PROCESS_FPS
    
    while not stop_event.is_set():
        start = time.time()
        try:
            frame, meta = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        
        # --- 画像処理: ArUco検出 & デバッグ描画 ---
        corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
        found_target = False
        target_area = 0 
        
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for i, marker_id in enumerate(ids):
                if marker_id[0] == TARGET_ID:
                    c = corners[i][0]
                    area = cv2.contourArea(c)
                    if area > PHASE_TRANSITION_AREA_THRESHOLD:
                        found_target = True
                        target_area = area
                    else:
                         cv2.putText(frame, f"ID:{TARGET_ID} TOO FAR(A:{int(area)})", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 2. 壁・ライン検出
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 30, 80)
        h, w = edges.shape
        EdgeArray = []
        StepSize = 5
        
        for j in range(0, w, StepSize):
            if stop_event.is_set(): break
            pixel = (j, 0)
            for i in range(h-1, -1, -1):
                if edges[i, j] == 255:
                    pixel = (j, i); break
            EdgeArray.append(pixel)
        
        avg_of_chunk = []
        if len(EdgeArray) > 0:
            size_of_chunk = max(1, len(EdgeArray) // 3)
            for i in range(0, len(EdgeArray), size_of_chunk):
                chunk = EdgeArray[i:i+size_of_chunk]
                if not chunk: continue
                xs, ys = zip(*chunk)
                avg_of_chunk.append((int(np.average(xs)), int(np.average(ys))))
        
        dir_now = "forward"
        if len(avg_of_chunk) >= 3:
            leftEdge = avg_of_chunk[0]
            forwardEdge = avg_of_chunk[1]
            rightEdge = avg_of_chunk[2]
            
            if forwardEdge[1] <= h * 0.9 and forwardEdge[1] >= FORWARD_THRESHOLD:
                dir_now = "left" if leftEdge[1] < rightEdge[1] else "right"
            
            with direction_lock:
                forward_y = forwardEdge[1]; left_y = leftEdge[1]; right_y = rightEdge[1]
        
        with direction_lock:
            latest_direction = dir_now

        # --- 描画処理 ---
        cv2.line(frame, (0, FORWARD_THRESHOLD), (w, FORWARD_THRESHOLD), (0, 255, 255), 1)
        cv2.putText(frame, "OBSTACLE THRESHOLD", (10, FORWARD_THRESHOLD - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        if len(avg_of_chunk) >= 3:
            pts_to_draw = [("L", avg_of_chunk[0]), ("F", avg_of_chunk[1]), ("R", avg_of_chunk[2])]
            for name, (px, py) in pts_to_draw:
                if py >= FORWARD_THRESHOLD:
                    color = (0, 0, 255) # Red
                    cv2.putText(frame, "BLOCK", (px-20, py-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                else:
                    color = (0, 255, 0) # Green
                cv2.circle(frame, (px, py), 6, color, -1)
                cv2.putText(frame, f"{name}:{py}", (px+10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        act = debug_info['action']
        err = debug_info['error_deg']
        dst = debug_info['dist_cm']
        
        cv2.putText(frame, f"{act}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"ERR: {err:.1f} deg", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"DST: {dst:.1f} cm", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # ▼▼▼【修正】Areaサイズを常に表示 ▼▼▼
        if found_target:
             cv2.putText(frame, f"MARKER FOUND!(A:{int(target_area)})", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        # 4. マップ作成
        MAP_H, MAP_W = 400, 300
        map_img = np.zeros((MAP_H, MAP_W, 3), dtype=np.uint8)
        OX, OY = MAP_W // 2, MAP_H - 50 
        SCALE = 1.5
        
        gx = int(OX + GOAL_X * SCALE)
        gy = int(OY - GOAL_Y * SCALE)
        cv2.circle(map_img, (gx, gy), 8, (0, 255, 0), 2)
        cv2.putText(map_img, "GOAL", (gx+10, gy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        rx = int(OX + current_x * SCALE)
        ry = int(OY - current_y * SCALE)
        cv2.circle(map_img, (rx, ry), 6, (0, 0, 255), -1)
        
        tx = int(rx + 20 * math.cos(current_theta))
        ty = int(ry - 20 * math.sin(current_theta))
        cv2.line(map_img, (rx, ry), (tx, ty), (0, 0, 255), 2)

        try:
            display_queue.put_nowait((frame, map_img))
        except queue.Full:
            pass 

        # 6. Phase 2終了時の安全確認
        if found_target:
            us_left = ultrasonic2.get(3)
            us_right = ultrasonic2.get(1)
            safe_left = True if (us_left is None) else (us_left > SAFE_TRANSITION_DIST)
            safe_right = True if (us_right is None) else (us_right > SAFE_TRANSITION_DIST)

            if safe_left and safe_right:
                print(f"\n[PHASE 2] Marker ID={TARGET_ID} DETECTED(A:{int(target_area)}) & CLEAR. Transition to Phase 3.")
                mbot2.drive_power(0, 0)
                stop_event.set()
                return 
            else:
                obs_msg = []
                if not safe_left: obs_msg.append(f"Left({us_left})")
                if not safe_right: obs_msg.append(f"Right({us_right})")
                print(f"[PHASE 2] Marker FOUND but OBSTACLE detected: {', '.join(obs_msg)}. Staying in Phase 2.")
        
        elapsed = time.time() - start
        if target_period - elapsed > 0:
            time.sleep(target_period - elapsed)

def mbot_control_thread():
    global forward_y

    mbot_safe_reset()
    print("[Control] Start Driving")
    
    MAX_FORWARD_SPEED = 30
    MIN_FORWARD_SPEED = 3
    TURN_THRESHOLD = 0.2
    
    SLOWDOWN_RADIUS = 30
    ULTRASONIC_EMERGENCY_DIST = 20
    ULTRASONIC_MAX_VAL_THRESHOLD = 290 
    TURN_ADJ_A = 1.7668
    TURN_ADJ_B = 6.6132
    FINAL_TURN_GAIN = 1.1
    MAX_TURN_DEG = 180 
    MOVE_PWR = 30 
    TURN_PWR = 50 

    STUCK_CHECK_INTERVAL = 3.0
    STUCK_DIST_THRESHOLD = 3.0
    STUCK_THRESHOLD_MULTIPLIER_US = 1.5 
    stuck_last_time = time.time()
    stuck_last_x = current_x
    stuck_last_y = current_y

    def drive_time_with_update(left_spd, right_spd, duration):
        start_t = time.time()
        mbot2.drive_speed(left_spd, right_spd) 
        while time.time() - start_t < duration:
            mbot_update_position() 
            time.sleep(0.05)
        mbot2.drive_speed(0, 0)
        mbot_update_position()

    def turn_with_correction(angle_x):
        corrected = TURN_ADJ_A * angle_x + TURN_ADJ_B if angle_x > 0 else TURN_ADJ_A * angle_x - TURN_ADJ_B
        corrected = max(min(corrected, MAX_TURN_DEG), -MAX_TURN_DEG)
        mbot2.turn(corrected)
        time.sleep(0.2)
        mbot_update_position()

    time.sleep(0.5) 
    mbot_update_position()
    init_goal_angle = mbot_heading_to_goal()
    init_angle_error = normalize_angle(init_goal_angle - current_theta)
    print(f"[Init] Aligning to goal. Error: {math.degrees(init_angle_error):.1f} deg")
    
    if abs(init_angle_error) > 0.05:
        turn_with_correction(-math.degrees(init_angle_error))
        time.sleep(0.5)
        mbot_update_position()

    stuck_last_time = time.time()
    stuck_last_x = current_x
    stuck_last_y = current_y

    while not stop_event.is_set():
        mbot_update_position()
        dx = GOAL_X - current_x
        dy = GOAL_Y - current_y
        dist_to_goal = math.hypot(dx, dy)
        goal_angle = mbot_heading_to_goal()
        angle_error = normalize_angle(goal_angle - current_theta)

        debug_info["error_deg"] = math.degrees(angle_error)
        debug_info["dist_cm"] = dist_to_goal

        if dist_to_goal < GOAL_TOLERANCE:
            print(f"\n[GOAL REACHED] Dist: {dist_to_goal:.1f}cm")
            debug_info["action"] = "GOAL REACHED"
            mbot2.drive_power(0, 0)
            time.sleep(0.3)
            
            target_theta = 90 * math.pi / 180
            final_diff = normalize_angle(target_theta - current_theta)
            deg_to_turn = -math.degrees(final_diff) * FINAL_TURN_GAIN
            print(f"Adjusting Heading: {math.degrees(final_diff):.1f} deg -> Turn command: {deg_to_turn:.1f} deg")
            
            if abs(deg_to_turn) > 5:
                turn_with_correction(deg_to_turn)
                time.sleep(1.5)
                mbot_update_position()
            
            mbot2.drive_power(0, 0)
            display.show_label("GOAL", 24, 2)
            stop_event.set()
            break

        front_dist = ultrasonic2.get(2)
        right_dist = ultrasonic2.get(1) 
        left_dist  = ultrasonic2.get(3) 
        
        debug_info["us_front"] = front_dist if front_dist else 999
        debug_info["us_right"] = right_dist if right_dist else 999
        debug_info["us_left"]  = left_dist  if left_dist  else 999

        with direction_lock:
            fy = forward_y; cam_dir = latest_direction

        # 1. 超音波回避
        if front_dist is not None and front_dist < ULTRASONIC_EMERGENCY_DIST:
            debug_info["action"] = "AVOID: US[FRONT] -> Turn LEFT"
            drive_time_with_update(-MOVE_PWR, MOVE_PWR, 1.0)
            turn_with_correction(30); time.sleep(0.5)
            mbot_update_position()
            drive_time_with_update(MOVE_PWR, -MOVE_PWR, 1.0)
            stuck_last_time = time.time()
            continue
        elif left_dist is not None and left_dist < ULTRASONIC_EMERGENCY_DIST:
            debug_info["action"] = "AVOID: US[LEFT] -> Turn LEFT"
            turn_with_correction(30)
            drive_time_with_update(MOVE_PWR, -MOVE_PWR, 1.0)
            stuck_last_time = time.time()
            continue
        elif right_dist is not None and right_dist < ULTRASONIC_EMERGENCY_DIST:
            debug_info["action"] = "AVOID: US[RIGHT] -> Turn RIGHT"
            turn_with_correction(-30)
            drive_time_with_update(MOVE_PWR, -MOVE_PWR, 1.0)
            stuck_last_time = time.time()
            continue
        
        # 2. カメラ回避
        elif fy > FORWARD_THRESHOLD:
            avoid_angle = -30 if angle_error > 0 else 30
            turn_str = "LEFT" if avoid_angle < 0 else "RIGHT"
            debug_info["action"] = f"AVOID: CAM[FRONT:{fy}] -> Turn {turn_str} (Goal Dir)"
            
            drive_time_with_update(-30, 30, 1.0) 
            turn_with_correction(avoid_angle)                
            time.sleep(0.5); mbot_update_position() 
            drive_time_with_update(30, -30, 1.0) 
            
            with direction_lock: forward_y = 0 
            continue

        # 3. ゴールへ
        # ▼▼▼【修正】バック後に旋回して前を向く処理を追加 ▼▼▼
        if dist_to_goal < 20 and abs(angle_error) > math.radians(60):
            debug_info["action"] = "Overshoot -> Back & Turn"
            
            # 1. バック
            drive_time_with_update(-7, 7, 1.5) 
            # 2. 位置更新
            mbot_update_position()
            # 3. ゴール方向計算
            goal_angle = mbot_heading_to_goal()
            angle_error_new = normalize_angle(goal_angle - current_theta)
            # 4. 旋回 (mbot2.turnは 正=右回転 なので -deg を渡す)
            turn_deg = -math.degrees(angle_error_new)
            turn_with_correction(turn_deg)
            # 5. リセット
            stuck_last_time = time.time()
            continue
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        if abs(angle_error) > TURN_THRESHOLD:
            debug_info["action"] = "Turn Adjust"
            angle_deg = -math.degrees(angle_error)
            mbot2.drive_power(0, 0)
            turn_with_correction(angle_deg)
            stuck_last_time = time.time(); stuck_last_x = current_x; stuck_last_y = current_y
        else:
            debug_info["action"] = "Forward"

            if dist_to_goal < SLOWDOWN_RADIUS:
                speed = 7 
            else:
                speed = MAX_FORWARD_SPEED

            HEADING_P_GAIN = 40.0
            turn_adj = angle_error * HEADING_P_GAIN
            turn_adj = max(min(turn_adj, 20), -20)

            pwr_left = speed - turn_adj
            pwr_right = -(speed + turn_adj)
            mbot2.drive_speed(pwr_left, pwr_right)

            curr_time = time.time()
            if curr_time - stuck_last_time > STUCK_CHECK_INTERVAL:
                dist_moved = math.hypot(current_x - stuck_last_x, current_y - stuck_last_y)
                current_stuck_threshold = STUCK_DIST_THRESHOLD
                is_us_max = front_dist is not None and front_dist > ULTRASONIC_MAX_VAL_THRESHOLD
                if is_us_max: current_stuck_threshold *= STUCK_THRESHOLD_MULTIPLIER_US

                if dist_moved < current_stuck_threshold:
                    debug_info["action"] = "STUCK RECOVERY"
                    drive_time_with_update(-30, 30, 1.5)
                    turn_with_correction(45) 
                    drive_time_with_update(30, -30, 1.0)
                    mbot_update_position()
                stuck_last_time = curr_time; stuck_last_x = current_x; stuck_last_y = current_y

        time.sleep(CONTROL_PERIOD)

def run_drive_phase():
    print("\n=== PHASE 2: MBOT ROUGH TRANSPORT START ===")
    stop_event.clear()
    
    threads = [
        threading.Thread(target=mbot_capture_thread),
        threading.Thread(target=mbot_processing_thread),
        threading.Thread(target=mbot_control_thread)
    ]
    
    for t in threads: 
        t.daemon = True
        t.start()

    try:
        while not stop_event.is_set():
            try:
                frame, map_img = display_queue.get(timeout=0.1)
                cv2.imshow("mBot View", frame)
                cv2.imshow("Map", map_img)
                cv2.waitKey(1)
            except queue.Empty:
                pass 
    except KeyboardInterrupt:
        stop_event.set()
    
    print("Stopping mBot threads (Phase 2)...")
    for t in threads:
        t.join(timeout=2.0)
    
    mbot2.drive_power(0, 0)
    cv2.destroyAllWindows()
    for _ in range(5): cv2.waitKey(1)
    print("=== PHASE 2 COMPLETE ===")

# ==========================================
# Phase 3: Fine Approach & Place (placetest3.py Logic)
# ==========================================

def move_mbot_to_aruco_logic(cap):
    """placetest3.py の move_mbot_to_aruco をリネーム"""
    print(f"--- mbot2: ArUco(ID:{TARGET_ID}) 精密誘導開始 ---")
    
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    center_x = frame_w // 2
    
    max_loops = 2000 
    
    search_step_counter = 0   
    lost_patience_counter = 0 
    LOST_THRESHOLD = 15       
    found_marker_previously = False 
    last_known_area = 0

    for i in range(max_loops):
        ret, frame = cap.read()
        if not ret: continue

        aruco_info, frame_disp = get_aruco_info(frame)
        cv2.line(frame_disp, (center_x, 0), (center_x, frame_h), (255, 255, 0), 2)

        if aruco_info:
            cx, cy, area = aruco_info
            last_known_area = area

            if not found_marker_previously:
                print(f"DEBUG: 発見！")
                found_marker_previously = True

            lost_patience_counter = 0
            search_step_counter = 0

            err_x = cx - center_x
            
            # 距離偏差 = 目標 - 現在 (正なら遠い、負なら近すぎる)
            err_dist = MBOT_TARGET_AREA - area 

            is_arrived = False
            
            # X方向と面積の両方が許容範囲内ならゴール
            if abs(err_x) < MBOT_TOLERANCE_X:
                if abs(err_dist) < MBOT_AREA_TOLERANCE: 
                    is_arrived = True
            
            # 念のため、近すぎてもXが合っていれば一旦OKとする
            if is_arrived:
                print(f"★ mbot2: 目標地点に到着 (Area:{int(area)})")
                drive_mbot_placetest(0, 0)
                time.sleep(0.5)
                return True

            if area > 30000: current_align_threshold = MBOT_ALIGN_THRESHOLD_NEAR
            else: current_align_threshold = MBOT_ALIGN_THRESHOLD_FAR

            if abs(err_x) > current_align_threshold:
                # ALIGNモード (インチング)
                abs_err = abs(err_x)
                if abs_err > INCH_THRESHOLD_LARGE:
                    move_time = TIME_ALIGN_LONG; intensity_msg = "LONG"
                elif abs_err > INCH_THRESHOLD_MEDIUM:
                    move_time = TIME_ALIGN_MID; intensity_msg = "MID"
                else:
                    move_time = TIME_ALIGN_TINY; intensity_msg = "TINY"
                
                status_msg = f"ALIGN {intensity_msg}"
                power = MBOT_INCH_POWER
                if abs_err <= INCH_THRESHOLD_MEDIUM:
                     power = MBOT_INCH_POWER_WEAK

                if err_x > 0: l_val, r_val = power, -power
                else: l_val, r_val = -power, power
                
                drive_mbot_power_placetest(l_val, r_val)
                time.sleep(move_time)
                drive_mbot_placetest(0, 0)
            else:
                # APPROACHモード (速度制御)
                status_msg = "GO"
                
                # 基本P制御
                forward_cmd = err_dist * MBOT_P_GAIN_FORWARD
                turn_correction = -err_x * 0.01 
                
                # 最大速度制限 (通常)
                if forward_cmd > MBOT_MAX_SPEED: forward_cmd = MBOT_MAX_SPEED
                if forward_cmd < -MBOT_MAX_SPEED: forward_cmd = -MBOT_MAX_SPEED

                # 速度制限と最低速度保証
                if area > 75000:   # ゴール直前
                    limit_speed = 8
                    status_msg = "SLOW(8)"
                elif area > 60000: # 接近中
                    limit_speed = 12
                    status_msg = "SLOW(12)"
                else:              # 遠方
                    limit_speed = MBOT_MAX_SPEED 
                
                forward_cmd = np.clip(forward_cmd, -limit_speed, limit_speed)

                MIN_MOVE_SPEED = 7
                if abs(forward_cmd) < MIN_MOVE_SPEED and abs(forward_cmd) > 0.5:
                     forward_cmd = MIN_MOVE_SPEED if forward_cmd > 0 else -MIN_MOVE_SPEED

                raw_left = forward_cmd - turn_correction
                raw_right = forward_cmd + turn_correction
                
                final_left = clamp_speed(raw_left, MIN_MOVE_SPEED, MBOT_MAX_SPEED)
                final_right = clamp_speed(raw_right, MIN_MOVE_SPEED, MBOT_MAX_SPEED)

                drive_mbot_placetest(final_left, final_right)

            print(f"Mode:{status_msg} Area:{int(area)} X:{err_x}")
            cv2.putText(frame_disp, f"{status_msg} A:{int(area)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        else:
            # ロスト時
            if found_marker_previously and last_known_area > MBOT_LOST_AS_GOAL_THRESHOLD:
                print(f"★ mbot2: 目標目前でロスト (LastArea:{int(last_known_area)}) -> 到着と判定")
                drive_mbot_placetest(0, 0)
                time.sleep(0.5)
                return True

            lost_patience_counter += 1
            if lost_patience_counter <= LOST_THRESHOLD:
                drive_mbot_placetest(0, 0)
                status_text = f"WAITING... {lost_patience_counter}"
                cv2.putText(frame_disp, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                # サーチモード
                if found_marker_previously:
                    print("DEBUG: ロスト(遠方) -> 探索開始")
                    found_marker_previously = False
                    last_known_area = 0 

                search_step_counter += 1
                step_limit = 20
                total_cycle = step_limit * 4
                cycle_pos = search_step_counter % total_cycle

                if cycle_pos < step_limit:
                    turn_dir = "LEFT"; p_left, p_right = -MBOT_INCH_POWER, MBOT_INCH_POWER
                elif cycle_pos < step_limit * 3:
                    turn_dir = "RIGHT"; p_left, p_right = MBOT_INCH_POWER, -MBOT_INCH_POWER
                else:
                    turn_dir = "LEFT"; p_left, p_right = -MBOT_INCH_POWER, MBOT_INCH_POWER
                
                drive_mbot_power_placetest(p_left, p_right)
                time.sleep(TIME_SEARCH_STEP)
                
                drive_mbot_placetest(0, 0)
                time.sleep(TIME_SEARCH_WAIT)

                msg = f"SEARCH {turn_dir} ({cycle_pos})"
                cv2.putText(frame_disp, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("mbot2 ArUco View", frame_disp)
        if cv2.waitKey(10) == 27: 
            drive_mbot_placetest(0, 0)
            return False
    
    print("タイムアウト")
    drive_mbot_placetest(0, 0)
    return False

def dobot_place_sequence_new(api, holding_color_idx):
    """placetest3.py の dobot_place_sequence_new を統合"""
    print(f"--- Dobot: 新配置シーケンス開始 (ColorIDX: {holding_color_idx}) ---")
    
    print("1. 指定ポジションへ移動 (安全移動)")
    dType.SetPTPCmdEx(api, 1, 200, 0, 120, 0, 0)

    print("2. ターゲットへアプローチ")
    p = dType.GetPose(api)
    dType.SetPTPCmdEx(api, 1, p[0] + 80, p[1], p[2], p[3], 0) 

    print("4. グリッパー開放")
    dType.SetEndEffectorGripperEx(api, True, False) # Open
    time.sleep(1) 

    
    print("6. グリッパー閉じる（退避後）")
    dType.SetEndEffectorGripperEx(api, True, True) # Close
    time.sleep(1) 
    dType.SetEndEffectorGripperEx(api, False, True) # Off
    time.sleep(1) 

    p = dType.GetPose(api)
    # 初期位置（後方）へ戻る
    dType.SetPTPCmdEx(api, 1, 200, 0, 173, 0, 0) 
    p = dType.GetPose(api)
    dType.SetPTPCmdEx(api, 4, -125, -5, 35, p[7], 0)

def run_phase3_sequence(holding_color_idx):
    print("\n=== PHASE 3: FINE APPROACH & PLACE START ===")
    
    cap = cv2.VideoCapture(CAMERA_ID_MBOT)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera for Phase 3")
        return

    try:
        success = move_mbot_to_aruco_logic(cap)
        if success:
            print(">> ArUco Docking Success. Starting Dobot Place...")
            dobot.initialize() 
            dType.ClearAllAlarmsState(dobot.api)
            dobot_place_sequence_new(dobot.api, holding_color_idx)
        else:
            print(">> ArUco Docking Failed. Skipping Place.")
            
    finally:
        cap.release()
        cv2.destroyAllWindows()
        drive_mbot_placetest(0, 0)

    print("=== PHASE 3 COMPLETE ===")

# ==========================================
# Main Sequencer
# ==========================================
def main():
    print("Starting Integrated Pick-Transport-Place System (0126 + placetest3)")
    
    # --- Phase 1: Pick ---
    if SKIP_PICK:
        print(">>> SKIP_PICK is ON. Moving arm to back position...")
        dobot.initialize()
        dType.ClearAllAlarmsState(dobot.api)
        time.sleep(0.5)
        p = dType.GetPose(dobot.api)
        dType.SetPTPCmdEx(dobot.api, 4, -125, -5, 35, p[7], 0)
        dobot.wait(3.0)
        dobot.finalize()
        picked_color = 0 # ダミー色
    else:
        picked_color = run_pick_phase()
        if picked_color is None:
            print("Pick failed.")
            os._exit(0)

    # --- Phase 2: Rough Transport (0126 original) ---
    run_drive_phase()

    # --- Phase 3: Fine Approach & Place (placetest3 logic) ---
    run_phase3_sequence(picked_color)
    
    print("\nALL TASKS FINISHED.")
    os._exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda s, f: stop_event.set())
    main()
