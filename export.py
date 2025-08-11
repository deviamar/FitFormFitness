import cv2, time, math, csv, os, threading, serial
import numpy as np
from collections import deque
from ultralytics import YOLO
import mediapipe as mp
from datetime import datetime

# ------------------------- Config -------------------------
POSE_MODEL_PATH = "yolov8n-pose.pt"   # or "yolov8s-pose.pt"
CAM_INDEX = 0                         # 1/2 if using iPhone Continuity Camera
CONF_MIN = 0.25
TRAIL_LEN = 90
EMA_ALPHA = 0.7

# Arduino serial (USB or BT). Example: '/dev/cu.usbmodem11201' or '/dev/tty.HC-05-XXXX'
SER_PORT = "/dev/cu.usbmodem11401"
BAUD     = 115200
# CSV output
OUT_DIR = "."
OUT_CSV = os.path.join(OUT_DIR, f"tennis_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

# ------------------------- Models -------------------------
pose_model = YOLO(POSE_MODEL_PATH)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ------------------------- Keypoints -------------------------
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = range(17)
RIGHT = {"SH": R_SH, "EL": R_EL, "WR": R_WR, "HIP": R_HIP}
LEFT  = {"SH": L_SH, "EL": L_EL, "WR": L_WR, "HIP": L_HIP}
side = RIGHT

# ------------------------- State -------------------------
wrist_trail = deque(maxlen=TRAIL_LEN)
elbow_ema = wristdev_ema = abduct_ema = None
fps_t, fps_n = time.time(), 0

# latest IMU sample (updated by serial thread)
latest_imu = {"ax": np.nan, "ay": np.nan, "az": np.nan, "gx": np.nan, "gy": np.nan, "gz": np.nan, "t": 0}
imu_lock = threading.Lock()
stop_event = threading.Event()

# ------------------------- Helpers -------------------------
def angle_abc(a, b, c):
    ax, ay = a; bx, by = b; cx, cy = c
    v1 = np.array([ax - bx, ay - by], float)
    v2 = np.array([cx - bx, cy - by], float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6: return None
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def signed_angle_deg(u, v):
    u = np.asarray(u, float); v = np.asarray(v, float)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-6 or nv < 1e-6: return None
    u /= nu; v /= nv
    dot = np.clip(np.dot(u, v), -1.0, 1.0)
    ang = float(np.degrees(np.arccos(dot)))
    cross_z = u[0]*v[1] - u[1]*v[0]  # image y-down
    return ang if cross_z >= 0 else -ang

def ema(prev, curr, alpha=EMA_ALPHA):
    if curr is None: return prev
    return curr if prev is None else alpha*prev + (1-alpha)*curr

def draw_trail(img, pts, color=(255, 200, 0)):
    for i in range(1, len(pts)):
        cv2.line(img, tuple(pts[i-1]), tuple(pts[i]), color, 2)

def fmt(v, unit="°", prec=1, none_str="—"):
    return f"{v:.{prec}f}{unit}" if v is not None else none_str

# ------------------------- Serial Thread -------------------------
def serial_reader(port, baud):
    try:
        ser = serial.Serial(port, baud, timeout=1)
        # optional: flush first line if it’s a header
        time.sleep(0.2)
        ser.reset_input_buffer()
    except Exception as e:
        print(f"[IMU] Could not open serial port {port} @ {baud}: {e}")
        return

    print(f"[IMU] Reading from {port} @ {baud}")
    while not stop_event.is_set():
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue
            # Expecting "ax,ay,az,gx,gy,gz" (6 numbers). If your Arduino sends a header, ignore non-numeric lines.
            parts = line.split(",")
            if len(parts) >= 6:
                vals = []
                ok = True
                for i in range(6):
                    try:
                        vals.append(float(parts[i]))
                    except:
                        ok = False; break
                if ok:
                    with imu_lock:
                        latest_imu.update({"ax": vals[0], "ay": vals[1], "az": vals[2],
                                           "gx": vals[3], "gy": vals[4], "gz": vals[5],
                                           "t": int(time.time()*1000)})
        except Exception as e:
            # keep going; print occasionally if needed
            pass
    try:
        ser.close()
    except:
        pass
    print("[IMU] Serial thread stopped.")

# ------------------------- Main -------------------------
def main():
    global side, elbow_ema, wristdev_ema, abduct_ema, fps_t, fps_n

    # Start serial thread
    t_ser = threading.Thread(target=serial_reader, args=(SER_PORT, BAUD), daemon=True)
    t_ser.start()

    # Prepare CSV
    os.makedirs(OUT_DIR, exist_ok=True)
    f = open(OUT_CSV, "w", newline="")
    writer = csv.writer(f)
    header = ["time_ms","ax","ay","az","gx","gy","gz","elbow_deg","wrist_dev_deg","abduction_deg","trail_len_px","trail_dY_px"]
    writer.writerow(header); f.flush()
    print(f"[LOG] Writing to {OUT_CSV}")

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Camera not found. Try CAM_INDEX=1 or 2.")
        stop_event.set(); t_ser.join(timeout=1.0)
        return

    print("Keys: ESC/Q=quit  L=toggle left/right  R=reset trail")

    while True:
        ok, frame = cap.read()
        if not ok: break
        h, w = frame.shape[:2]
        t_ms = int(time.time()*1000)

        # Pose (single person)
        res = pose_model.predict(frame, imgsz=640, max_det=1, conf=0.25, verbose=False)[0]

        elbow_deg = wrist_dev_deg = abduct_deg = None
        S = E = W = H = None

        if len(res.keypoints):
            kps = res.keypoints.xy[0].cpu().numpy()
            conf = res.keypoints.conf[0].cpu().numpy()
            SH, EL, WR, HIP = side["SH"], side["EL"], side["WR"], side["HIP"]

            if conf[SH] > CONF_MIN and conf[EL] > CONF_MIN and conf[WR] > CONF_MIN:
                S = tuple(map(int, kps[SH]))
                E = tuple(map(int, kps[EL]))
                W = tuple(map(int, kps[WR]))

                cv2.line(frame, S, E, (0, 255, 0), 3)
                cv2.line(frame, E, W, (0, 255, 0), 3)
                for p in (S, E, W): cv2.circle(frame, p, 4, (0, 165, 255), -1)

                elbow_deg = angle_abc(S, E, W)

                if conf[HIP] > CONF_MIN:
                    H = tuple(map(int, kps[HIP]))
                    upper_arm = (E[0]-S[0], E[1]-S[1])
                    torso_vec = (H[0]-S[0], H[1]-S[1])
                    # signed angle from torso axis to upper arm
                    abduct_deg = signed_angle_deg(torso_vec, upper_arm)
                    cv2.line(frame, S, H, (200, 200, 200), 2)

                wrist_trail.append(W)

        # Wrist deviation via MediaPipe Hands (wrist->middle MCP vs elbow->wrist)
        if W is not None and E is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res_h = hands.process(rgb)
            if res_h.multi_hand_landmarks:
                lm = res_h.multi_hand_landmarks[0].landmark
                WRIST_LM, MID_MCP = 0, 9
                def to_px(pt): return np.array([int(pt.x * w), int(pt.y * h)], dtype=np.int32)
                wrist_px = to_px(lm[WRIST_LM])
                mid_mcp_px = to_px(lm[MID_MCP])

                PALM_SET = [0,1,5,9,13,17]
                palm_pts = np.stack([to_px(lm[i]) for i in PALM_SET], axis=0)
                palm_center = palm_pts.mean(axis=0).astype(np.int32)
                cv2.circle(frame, tuple(palm_center), 4, (255, 0, 200), -1)

                forearm_vec = (np.array(W, float) - np.array(E, float))
                hand_axis   = (mid_mcp_px.astype(float) - wrist_px.astype(float))
                wrist_dev_deg = signed_angle_deg(forearm_vec, hand_axis)
                cv2.line(frame, tuple(wrist_px), tuple(mid_mcp_px), (255, 0, 200), 2)

        # Trail metrics
        path_len = height_dy = None
        if len(wrist_trail) >= 2:
            draw_trail(frame, list(wrist_trail))
            pts = np.array(wrist_trail, int)
            segs = np.diff(pts, axis=0)
            path_len = float(np.sum(np.linalg.norm(segs, axis=1)))
            height_dy = float(pts[-1,1] - pts[0,1])

        # Smooth
        elbow_ema    = ema(elbow_ema,   elbow_deg)
        wristdev_ema = ema(wristdev_ema, wrist_dev_deg)
        abduct_ema   = ema(abduct_ema,  abduct_deg)

        # HUD
        fps_n += 1
        hud = [
            f"Side: {'RIGHT' if side is RIGHT else 'LEFT'}",
            f"Elbow: {fmt(elbow_ema)}",
            f"Wrist dev: {fmt(wristdev_ema, prec=1)}",
            f"Abduction: {fmt(abduct_ema, prec=1)}",
            f"Trail len: {fmt(path_len, unit='px', prec=0)}  dY: {fmt(height_dy, unit='px', prec=0)}"
        ]
        if fps_n % 12 == 0:
            fps = 12 / (time.time() - fps_t); fps_t = time.time()
            hud.append(f"FPS: {fps:4.1f}")
        y0 = 26
        for i, t in enumerate(hud):
            cv2.putText(frame, t, (10, y0 + i*24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        # ---- CSV write: one row per frame using latest IMU sample ----
        with imu_lock:
            ax = latest_imu["ax"]; ay = latest_imu["ay"]; az = latest_imu["az"]
            gx = latest_imu["gx"]; gy = latest_imu["gy"]; gz = latest_imu["gz"]
        row = [
            t_ms,
            float(ax), float(ay), float(az),
            float(gx), float(gy), float(gz),
            float(elbow_ema) if elbow_ema is not None else "",
            float(wristdev_ema) if wristdev_ema is not None else "",
            float(abduct_ema) if abduct_ema is not None else "",
            float(path_len) if path_len is not None else "",
            float(height_dy) if height_dy is not None else ""
        ]
        writer.writerow(row)
        # flush occasionally to ensure data lands on disk
        if fps_n % 30 == 0: f.flush()

        cv2.imshow("Tennis Arm Metrics (YOLOv8 Pose + MediaPipe Hands)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('l'):
            side = LEFT if side is RIGHT else RIGHT
            wrist_trail.clear()
            elbow_ema = wristdev_ema = abduct_ema = None
        elif key == ord('r'):
            wrist_trail.clear()

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    stop_event.set()
    t_ser.join(timeout=1.0)
    f.flush(); f.close()
    print(f"[LOG] Saved: {OUT_CSV}")

if __name__ == "__main__":
    main()

