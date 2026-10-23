import csv
import math
import os
import threading
import time
from datetime import datetime
from math import dist, cos, radians
from time import perf_counter

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify, render_template, send_file

try:
    import winsound
except ImportError:
    winsound = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTION_LOG = os.path.join(BASE_DIR, "dashboard_detection_log.csv")
TIMING_LOG = os.path.join(BASE_DIR, "dashboard_event_timing_log.csv")

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 81, 13, 311, 308, 402, 14, 178]
EAR_THRESHOLD = 0.20
MAR_THRESHOLD = 0.35
EYE_CLOSED_SECONDS = 1.5
YAWN_SECONDS = 1.0
SMOOTHING_FACTOR = 0.4
YAW_CORRECTION_START = 10.0
MAX_RELIABLE_YAW = 50.0

app = Flask(__name__)
lock = threading.RLock()
stop_requested = threading.Event()
alarm_event = threading.Event()

state = {
    "running": False, "status": "STOPPED", "message": "Press Start Monitoring to begin.",
    "ear": None, "left_ear": None, "right_ear": None, "mar": None,
    "yaw": None, "pitch": None, "roll": None,
    "eye_duration": 0.0, "mouth_duration": 0.0, "yawn_count": 0,
    "face_detected": False, "started_at": None, "events": [],
    "blink_count": 0, "prolonged_closure_count": 0,
}
latest_frame = None
worker = None
alarm_worker_thread = None


def get_points(landmarks, indices, width, height):
    return [(int(landmarks[i].x * width), int(landmarks[i].y * height)) for i in indices]


def calculate_ear(points):
    vertical = dist(points[1], points[5]) + dist(points[2], points[4])
    horizontal = 2 * dist(points[0], points[3])
    return vertical / horizontal if horizontal else 0.0


def calculate_mar(points):
    vertical = dist(points[1], points[7]) + dist(points[2], points[6]) + dist(points[3], points[5])
    horizontal = 2 * dist(points[0], points[4])
    return vertical / horizontal if horizontal else 0.0


def estimate_head_pose(landmarks, width, height):
    image_points = np.array([
        (landmarks[1].x * width, landmarks[1].y * height),
        (landmarks[152].x * width, landmarks[152].y * height),
        (landmarks[33].x * width, landmarks[33].y * height),
        (landmarks[263].x * width, landmarks[263].y * height),
        (landmarks[61].x * width, landmarks[61].y * height),
        (landmarks[291].x * width, landmarks[291].y * height),
        (landmarks[10].x * width, landmarks[10].y * height),
    ], dtype=np.float64)
    model_points = np.array([
        (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0), (0.0, 300.0, -100.0),
    ], dtype=np.float64)
    focal = float(width)
    camera_matrix = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)
    success, rotation_vector, _ = cv2.solvePnP(
        model_points, image_points, camera_matrix, np.zeros((4, 1), dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles = cv2.RQDecomp3x3(rotation_matrix)[0]
    return float(angles[0]), float(angles[1]), float(angles[2])


def draw_text(frame, text, position, color=(255, 255, 255), scale=0.62):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def ensure_csvs():
    if not os.path.exists(DETECTION_LOG):
        with open(DETECTION_LOG, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "Timestamp", "EAR", "Left_EAR", "Right_EAR", "MAR", "Yaw_Degrees",
                "Pitch_Degrees", "Roll_Degrees", "Eye_Closure_Seconds", "Mouth_Open_Seconds",
                "Status", "Possible_Yawn_Count",
            ])
    if not os.path.exists(TIMING_LOG):
        with open(TIMING_LOG, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["Timestamp", "Event_Type", "Event_Start_Sec", "Warning_Time_Sec", "Detection_Delay_Sec"])


def add_event(event_type, details=""):
    event = {"timestamp": datetime.now().isoformat(timespec="seconds"), "event": event_type, "details": details}
    with lock:
        state["events"].insert(0, event)
        state["events"] = state["events"][:40]


def alarm_loop():
    while True:
        if alarm_event.is_set():
            if winsound:
                try:
                    winsound.Beep(1000, 500)
                except RuntimeError:
                    time.sleep(0.1)
            else:
                print("\a", end="", flush=True)
                time.sleep(0.5)
        else:
            alarm_event.wait(0.1)


def write_timing(writer, timing_file, session_start, event_type, event_start, warning_time):
    start_s = event_start - session_start
    warning_s = warning_time - session_start
    delay = warning_time - event_start
    writer.writerow([datetime.now().isoformat(timespec="milliseconds"), event_type,
                     f"{start_s:.3f}", f"{warning_s:.3f}", f"{delay:.3f}"])
    timing_file.flush()
    add_event(event_type, f"Warning after {delay:.3f} seconds")


def detection_loop():
    global latest_frame
    stop_requested.clear()
    alarm_event.clear()
    ensure_csvs()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == "nt" else 0)
    if not cap.isOpened():
        cap.release()
        with lock:
                had_error = state.get("status") in {
                    "CAMERA ERROR",
                    "DETECTION ERROR"
                }

                state.update(
                    running=False,
                    face_detected=False
                )

                if not had_error:
                    state.update(
                        status="STOPPED",
                        message="Monitoring stopped."
                    )
        return

    session_start = perf_counter()
    smoothed_ear = None
    smoothed_mar = None
    eye_closed_start = None
    mouth_open_start = None
    eye_warning_logged = False
    yawn_active = False
    yawn_count = 0
    blink_count = 0
    prolonged_count = 0
    last_log_time = 0.0
    with lock:
        state.update(running=True, status="STARTING", message="Camera startingâ€¦", ear=None, left_ear=None,
                     right_ear=None, mar=None, yaw=None, pitch=None, roll=None, eye_duration=0.0,
                     mouth_duration=0.0, yawn_count=0, blink_count=0, prolonged_closure_count=0,
                     face_detected=False, started_at=datetime.now().isoformat(timespec="seconds"), events=[])
    add_event("SESSION_STARTED", "Monitoring session started")


    log_file = open(DETECTION_LOG, "a", newline="",    encoding="utf-8")
    writer = csv.writer(log_file)

    timing_file = open(TIMING_LOG, "a", newline="",   encoding="utf-8")
    timing_writer = csv.writer(timing_file)


    try:
        with mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            while not stop_requested.is_set():
                ret, frame = cap.read()
                if not ret:
                    with lock:
                        state.update(status="CAMERA ERROR", message="Could not read a camera frame.")
                    time.sleep(0.05)
                    continue
                # Mirror preview for a natural webcam experience.
                frame = cv2.flip(frame, 1)
                now = perf_counter()
                height, width = frame.shape[:2]
                results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                status, status_color = "NORMAL", (0, 200, 0)
                ear = left_ear = right_ear = mar = None
                yaw = pitch = roll = None
                eye_duration = mouth_duration = 0.0
                pose_unreliable = False

                if results.multi_face_landmarks:
                    face = results.multi_face_landmarks[0]
                    landmarks = face.landmark
                    mp.solutions.drawing_utils.draw_landmarks(
                        image=frame, landmark_list=face,
                        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))
                    left = get_points(landmarks, LEFT_EYE, width, height)
                    right = get_points(landmarks, RIGHT_EYE, width, height)
                    mouth = get_points(landmarks, MOUTH, width, height)
                    left_ear, right_ear = calculate_ear(left), calculate_ear(right)
                    pose = estimate_head_pose(landmarks, width, height)
                    if pose is not None:
                        pitch, yaw, roll = pose
                    abs_yaw = abs(yaw) if yaw is not None else 0.0
                    left_eye_width, right_eye_width = dist(left[0], left[3]), dist(right[0], right[3])
                    ear = left_ear if abs_yaw >= YAW_CORRECTION_START and left_eye_width >= right_eye_width else (
                        right_ear if abs_yaw >= YAW_CORRECTION_START else (left_ear + right_ear) / 2)
                    raw_mar = calculate_mar(mouth)
                    mar = raw_mar * cos(radians(abs_yaw)) if yaw is not None and abs_yaw < MAX_RELIABLE_YAW else raw_mar
                    smoothed_ear = ear if smoothed_ear is None else SMOOTHING_FACTOR * ear + (1 - SMOOTHING_FACTOR) * smoothed_ear
                    smoothed_mar = mar if smoothed_mar is None else SMOOTHING_FACTOR * mar + (1 - SMOOTHING_FACTOR) * smoothed_mar
                    ear, mar = smoothed_ear, smoothed_mar
                    pose_unreliable = yaw is not None and abs_yaw > MAX_RELIABLE_YAW

                    if pose_unreliable:
                        eye_closed_start = mouth_open_start = None
                        eye_warning_logged = yawn_active = False
                        status, status_color = "HEAD TURNED - MEASUREMENTS UNRELIABLE", (0, 165, 255)
                    else:
                        if ear < EAR_THRESHOLD:
                            if eye_closed_start is None:
                                eye_closed_start = now
                            eye_duration = now - eye_closed_start
                            if eye_duration >= EYE_CLOSED_SECONDS:
                                status, status_color = "WARNING: PROLONGED EYE CLOSURE", (0, 0, 255)
                                if not eye_warning_logged:
                                    write_timing(timing_writer, timing_file, session_start, "EYE_CLOSURE", eye_closed_start, now)
                                    eye_warning_logged = True
                                    prolonged_count += 1
                                    with lock: state["prolonged_closure_count"] = prolonged_count
                            else:
                                status, status_color = "EYES CLOSING", (0, 165, 255)
                        else:
                            if eye_closed_start is not None:
                                duration = now - eye_closed_start
                                if 0.08 <= duration < EYE_CLOSED_SECONDS:
                                    blink_count += 1
                                    with lock: state["blink_count"] = blink_count
                                    add_event("BLINK", f"Eye closure lasted {duration:.2f} seconds")
                            eye_closed_start = None
                            eye_warning_logged = False
                            eye_duration = 0.0

                        if mar > MAR_THRESHOLD:
                            if mouth_open_start is None:
                                mouth_open_start = now
                            mouth_duration = now - mouth_open_start
                            if mouth_duration >= YAWN_SECONDS:
                                if not yawn_active:
                                    yawn_count += 1
                                    yawn_active = True
                                    write_timing(timing_writer, timing_file, session_start, "POSSIBLE_YAWN", mouth_open_start, now)
                                if status == "NORMAL":
                                    status, status_color = "POSSIBLE YAWN", (0, 165, 255)
                        else:
                            mouth_open_start = None
                            yawn_active = False
                            mouth_duration = 0.0

                    draw_text(frame, f"EAR: {ear:.3f}", (20, 35))
                    draw_text(frame, f"Left EAR: {left_ear:.3f}  Right EAR: {right_ear:.3f}", (20, 65), scale=0.52)
                    draw_text(frame, f"MAR: {mar:.3f}", (20, 95))
                    draw_text(frame, f"Eye closure: {eye_duration:.1f}s  Mouth open: {mouth_duration:.1f}s", (20, 125), scale=0.52)
                    draw_text(frame, f"Possible yawns: {yawn_count}", (20, 155))
                    if yaw is not None:
                        draw_text(frame, f"Yaw: {yaw:.1f} Pitch: {pitch:.1f} Roll: {roll:.1f}", (20, 185), scale=0.52)
                else:
                    eye_closed_start = mouth_open_start = None
                    eye_warning_logged = yawn_active = False
                    status, status_color = "FACE NOT DETECTED", (0, 165, 255)

                alarm_event.set() if status == "WARNING: PROLONGED EYE CLOSURE" else alarm_event.clear()
                draw_text(frame, status, (20, height - 25), status_color, scale=0.58)
                if now - last_log_time >= 1.0:
                    writer.writerow([datetime.now().isoformat(timespec="seconds"), f"{ear:.3f}" if ear is not None else "",
                        f"{left_ear:.3f}" if left_ear is not None else "", f"{right_ear:.3f}" if right_ear is not None else "",
                        f"{mar:.3f}" if mar is not None else "", f"{yaw:.2f}" if yaw is not None else "",
                        f"{pitch:.2f}" if pitch is not None else "", f"{roll:.2f}" if roll is not None else "",
                        f"{eye_duration:.2f}", f"{mouth_duration:.2f}", status, yawn_count])
                    log_file.flush()
                    last_log_time = now
                ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if ok:
                    with lock:
                        latest_frame = encoded.tobytes()
                        state.update(status=status, message=status.replace("WARNING:", "Alert:").replace("_", " "),
                            ear=round(ear, 3) if ear is not None else None,
                            left_ear=round(left_ear, 3) if left_ear is not None else None,
                            right_ear=round(right_ear, 3) if right_ear is not None else None,
                            mar=round(mar, 3) if mar is not None else None,
                            yaw=round(yaw, 1) if yaw is not None else None,
                            pitch=round(pitch, 1) if pitch is not None else None,
                            roll=round(roll, 1) if roll is not None else None,
                            eye_duration=round(eye_duration, 2), mouth_duration=round(mouth_duration, 2),
                            yawn_count=yawn_count, face_detected=True)
    except Exception as exc:
        with lock:
            state.update(status="DETECTION ERROR", message=f"{type(exc).__name__}: {exc}")
        add_event("DETECTION_ERROR", str(exc))
    finally:
        alarm_event.clear()
        cap.release()
        log_file.close()
        timing_file.close()
        with lock:
            state.update(running=False, status="STOPPED", message="Monitoring stopped.", face_detected=False)
        add_event("SESSION_STOPPED", "Monitoring session stopped")


def frame_generator():
    while True:
        with lock:
            frame = latest_frame
        if frame is not None:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.04)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(frame_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/status")
def api_status():
    with lock:
        return jsonify(dict(state))


@app.post("/api/start")
def api_start():
    global worker
    with lock:
        if state["running"] or (worker and worker.is_alive()):
            return jsonify(ok=True, message="Monitoring is already running.")
        worker = threading.Thread(target=detection_loop, daemon=True)
        worker.start()
    return jsonify(ok=True, message="Monitoring is starting.")


@app.post("/api/stop")
def api_stop():
    stop_requested.set()
    alarm_event.clear()
    return jsonify(ok=True, message="Stop requested.")


@app.get("/api/report")
def api_report():
    filename = request_file = DETECTION_LOG
    ensure_csvs()
    return send_file(filename, as_attachment=True, download_name="dashboard_detection_log.csv", mimetype="text/csv")


@app.get("/api/timing-report")
def timing_report():
    ensure_csvs()
    return send_file(TIMING_LOG, as_attachment=True, download_name="dashboard_event_timing_log.csv", mimetype="text/csv")


if __name__ == "__main__":
    ensure_csvs()
    alarm_worker_thread = threading.Thread(target=alarm_loop, daemon=True)
    alarm_worker_thread.start()
    print("Open http://127.0.0.1:5000 in your browser.")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True, use_reloader=False)
