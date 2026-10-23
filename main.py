
import csv
import cv2
import mediapipe as mp
import winsound
import threading
import numpy as np

from datetime import datetime
from math import dist, cos, radians
from time import perf_counter


# ---------------- ALARM ----------------

alarm_event = threading.Event()
stop_alarm = threading.Event()


def alarm_worker():
    while not stop_alarm.is_set():
        if alarm_event.is_set():
            winsound.Beep(1000, 500)
            stop_alarm.wait(0.2)
        else:
            alarm_event.wait(0.1)


alarm_thread = threading.Thread(target=alarm_worker, daemon=True)
alarm_thread.start()


# ---------------- MEDIAPIPE ----------------

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 81, 13, 311, 308, 402, 14, 178]

EAR_THRESHOLD = 0.20
MAR_THRESHOLD = 0.35

EYE_CLOSED_SECONDS = 1.5
YAWN_SECONDS = 1.0

SMOOTHING_FACTOR = 0.4

smoothed_ear = None
smoothed_mar = None

YAW_CORRECTION_START = 10.0
MAX_RELIABLE_YAW = 50.0


# ---------------- FUNCTIONS ----------------

def get_points(landmarks, indices, width, height):
    return [
        (int(landmarks[i].x * width), int(landmarks[i].y * height))
        for i in indices
    ]


def calculate_ear(points):
    vertical = (
        dist(points[1], points[5])
        + dist(points[2], points[4])
    )
    horizontal = 2 * dist(points[0], points[3])
    return vertical / horizontal if horizontal else 0.0


def calculate_mar(points):
    vertical = (
        dist(points[1], points[7])
        + dist(points[2], points[6])
        + dist(points[3], points[5])
    )
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
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
        (0.0, 300.0, -100.0),
    ], dtype=np.float64)

    focal_length = float(width)
    camera_matrix = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1],
    ], dtype=np.float64)

    distortion = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vector, _ = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles = cv2.RQDecomp3x3(rotation_matrix)[0]

    pitch, yaw, roll = (
        float(angles[0]),
        float(angles[1]),
        float(angles[2]),
    )

    return pitch, yaw, roll


def draw_text(frame, text, position, color=(255, 255, 255), scale=0.65):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
    )


# ---------------- CAMERA ----------------

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Could not access camera.")
    stop_alarm.set()
    alarm_thread.join(timeout=1)
    raise SystemExit


# Start a timer for this camera session.
session_start = perf_counter()


# ---------------- CSV LOGGING ----------------

log_file = open(
    "detection_log.csv",
    "w",
    newline="",
    encoding="utf-8",
)

writer = csv.writer(log_file)

writer.writerow([
    "Timestamp",
    "EAR",
    "Left_EAR",
    "Right_EAR",
    "MAR",
    "Yaw_Degrees",
    "Pitch_Degrees",
    "Roll_Degrees",
    "Eye_Closure_Seconds",
    "Mouth_Open_Seconds",
    "Status",
    "Possible_Yawn_Count",
])


# Separate log for event timing.
timing_file = open(
    "event_timing_log.csv",
    "w",
    newline="",
    encoding="utf-8",
)

timing_writer = csv.writer(timing_file)

timing_writer.writerow([
    "Timestamp",
    "Event_Type",
    "Event_Start_Sec",
    "Warning_Time_Sec",
    "Detection_Delay_Sec",
])

last_log_time = 0.0


def log_event_timing(event_type, event_start, warning_time):
    """Save event and warning times relative to camera-session start."""
    event_start_sec = event_start - session_start
    warning_time_sec = warning_time - session_start
    delay = warning_time - event_start

    timing_writer.writerow([
        datetime.now().isoformat(timespec="milliseconds"),
        event_type,
        f"{event_start_sec:.3f}",
        f"{warning_time_sec:.3f}",
        f"{delay:.3f}",
    ])

    timing_file.flush()

    print(
        f"{event_type}: warning after "
        f"{delay:.3f} seconds"
    )


# ---------------- DETECTION STATE ----------------

eye_closed_start = None
mouth_open_start = None

eye_warning_logged = False
yawn_active = False
yawn_count = 0


try:
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("Could not read camera frame.")
                break

            now = perf_counter()
            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            status = "NORMAL"
            status_color = (0, 200, 0)

            ear = None
            left_ear = None
            right_ear = None
            mar = None

            yaw = pitch = roll = None
            eye_duration = 0.0
            mouth_duration = 0.0
            pose_unreliable = False

            if results.multi_face_landmarks:
                face = results.multi_face_landmarks[0]
                landmarks = face.landmark

                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 255, 0),
                        thickness=1,
                        circle_radius=1,
                    ),
                )

                left = get_points(landmarks, LEFT_EYE, width, height)
                right = get_points(landmarks, RIGHT_EYE, width, height)
                mouth = get_points(landmarks, MOUTH, width, height)

                left_ear = calculate_ear(left)
                right_ear = calculate_ear(right)

                pose = estimate_head_pose(landmarks, width, height)

                if pose is not None:
                    pitch, yaw, roll = pose

                abs_yaw = abs(yaw) if yaw is not None else 0.0

                left_eye_width = dist(left[0], left[3])
                right_eye_width = dist(right[0], right[3])

                if abs_yaw >= YAW_CORRECTION_START:
                    ear = (
                        left_ear
                        if left_eye_width >= right_eye_width
                        else right_ear
                    )
                else:
                    ear = (left_ear + right_ear) / 2

                raw_mar = calculate_mar(mouth)

                if yaw is not None and abs_yaw < MAX_RELIABLE_YAW:
                    mar = raw_mar * cos(radians(abs_yaw))
                else:
                    mar = raw_mar

                if smoothed_ear is None:
                    smoothed_ear = ear
                else:
                    smoothed_ear = (
                        SMOOTHING_FACTOR * ear
                        + (1 - SMOOTHING_FACTOR) * smoothed_ear
                    )

                if smoothed_mar is None:
                    smoothed_mar = mar
                else:
                    smoothed_mar = (
                        SMOOTHING_FACTOR * mar
                        + (1 - SMOOTHING_FACTOR) * smoothed_mar
                    )

                ear = smoothed_ear
                mar = smoothed_mar

                pose_unreliable = (
                    yaw is not None and abs_yaw > MAX_RELIABLE_YAW
                )

                if pose_unreliable:
                    eye_closed_start = None
                    mouth_open_start = None
                    eye_warning_logged = False
                    yawn_active = False

                    status = "HEAD TURNED - MEASUREMENTS UNRELIABLE"
                    status_color = (0, 165, 255)

                else:
                    # Confirm closure using both eyes for a mostly front-facing head.
                    # Keep the existing single-eye correction for turned faces.
                    if abs_yaw < YAW_CORRECTION_START:
                        eye_closure_detected = (
                            left_ear < EAR_THRESHOLD
                            and right_ear < EAR_THRESHOLD
                            and ear < EAR_THRESHOLD
                        )
                    else:
                        eye_closure_detected = ear < EAR_THRESHOLD
                    # -------- EYE-CLOSURE TIMING --------

                    if eye_closure_detected:
                        if eye_closed_start is None:
                            eye_closed_start = now

                        eye_duration = now - eye_closed_start

                        if eye_duration >= EYE_CLOSED_SECONDS:
                            status = "WARNING: PROLONGED EYE CLOSURE"
                            status_color = (0, 0, 255)

                            # Record only once during this closure.
                            if not eye_warning_logged:
                                log_event_timing(
                                    "EYE_CLOSURE",
                                    eye_closed_start,
                                    now,
                                )
                                eye_warning_logged = True
                        else:
                            status = "EYES CLOSING"
                            status_color = (0, 165, 255)

                    else:
                        eye_closed_start = None
                        eye_warning_logged = False
                        eye_duration = 0.0

                    # -------- POSSIBLE-YAWN TIMING --------

                    if mar > MAR_THRESHOLD:
                      if mouth_open_start is None:
                          mouth_open_start = now

                      mouth_duration = now - mouth_open_start

                      if mouth_duration >= YAWN_SECONDS:
                          if not yawn_active:
                              yawn_count += 1
                              yawn_active = True

                              log_event_timing(
                                  "POSSIBLE_YAWN",
                                  mouth_open_start,
                                  now,
                              )

                          if status == "NORMAL":
                              status = "POSSIBLE YAWN"
                              status_color = (0, 165, 255)
                    else:
                        mouth_open_start = None
                        yawn_active = False
                        mouth_duration = 0.0



                draw_text(frame, f"EAR: {ear:.3f}", (20, 35))
                draw_text(frame, f"Left EAR: {left_ear:.3f}", (20, 65))
                draw_text(frame, f"Right EAR: {right_ear:.3f}", (20, 95))
                draw_text(
                    frame,
                    f"MAR: {mar:.3f}" if mar is not None else "MAR: --",
                    (20, 125),
                )
                draw_text(
                    frame,
                    f"Eye closure: {eye_duration:.1f} s",
                    (20, 155),
                )
                draw_text(
                    frame,
                    f"Mouth open: {mouth_duration:.1f} s",
                    (20, 185),
                )
                draw_text(
                    frame,
                    f"Possible yawns: {yawn_count}",
                    (20, 215),
                )

                if yaw is not None:
                    draw_text(
                        frame,
                        f"Yaw: {yaw:.1f}  Pitch: {pitch:.1f}  Roll: {roll:.1f}",
                        (20, 245),
                        scale=0.55,
                    )

            else:
                eye_closed_start = None
                mouth_open_start = None
                eye_warning_logged = False
                yawn_active = False

                status = "FACE NOT DETECTED"
                status_color = (0, 165, 255)

            # Alarm follows prolonged eye closure.
            if status == "WARNING: PROLONGED EYE CLOSURE":
                alarm_event.set()
            else:
                alarm_event.clear()

            # Save regular measurements approximately once per second.
            if now - last_log_time >= 1.0:
                writer.writerow([
                    datetime.now().isoformat(timespec="seconds"),
                    f"{ear:.3f}" if ear is not None else "",
                    f"{left_ear:.3f}" if left_ear is not None else "",
                    f"{right_ear:.3f}" if right_ear is not None else "",
                    f"{mar:.3f}" if mar is not None else "",
                    f"{yaw:.2f}" if yaw is not None else "",
                    f"{pitch:.2f}" if pitch is not None else "",
                    f"{roll:.2f}" if roll is not None else "",
                    f"{eye_duration:.2f}",
                    f"{mouth_duration:.2f}",
                    status,
                    yawn_count,
                ])

                log_file.flush()
                last_log_time = now

            draw_text(
                frame,
                status,
                (20, height - 25),
                status_color,
                scale=0.6,
            )

            cv2.imshow("Driver Dizziness Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    alarm_event.clear()
    stop_alarm.set()

    cap.release()
    cv2.destroyAllWindows()

    log_file.close()
    timing_file.close()

    alarm_thread.join(timeout=1)

    print("Driver Dizziness Detection stopped.")
    print("Measurement log: detection_log.csv")
    print("Event timing log: event_timing_log.csv")
