# Driver Dizziness Detection System

## 1. Project Overview

The **Driver Dizziness Detection System** is a computer vision-based application designed to monitor driver facial behavior and identify potential signs of drowsiness, including prolonged eye closure and yawning.

The system uses a webcam to capture facial information, processes facial landmarks in real time, and generates visual warnings and an alarm when configured detection conditions are met. A Flask-based web dashboard provides a user interface for monitoring alerts, viewing analytics, and reviewing session reports.

This project is intended as a prototype to support driver alertness monitoring. It is not a medical diagnostic tool or a replacement for safe driving practices.

## 2. Objectives

- Detect prolonged eye closure using facial landmarks.
- Identify possible yawning events.
- Generate an alarm when the configured eye-closure condition is triggered.
- Support detection when the face is tilted or turned within the system's reliable operating range.
- Provide a web-based dashboard for live monitoring.
- Record detection events and timing information for evaluation.
- Analyze system performance using classification metrics and detection-delay measurements.

## 3. Technologies Used

- **Python** — application logic and data processing
- **OpenCV** — webcam capture and image processing
- **MediaPipe 0.10.21** — facial landmark detection
- **NumPy** — numerical calculations
- **Flask** — web application backend
- **HTML, CSS and JavaScript** — dashboard interface
- **Matplotlib** — evaluation charts
- **Pandas and scikit-learn** — evaluation and classification metrics, where used
- **CSV** — event logging, test datasets and evaluation reports

## 4. System Features

### Eye-Closure Detection
The system calculates the Eye Aspect Ratio (EAR) from facial landmarks to estimate whether the eyes are closed. A configurable duration threshold helps distinguish prolonged closure from ordinary blinking.

### Yawning Detection
The system calculates the Mouth Aspect Ratio (MAR) from mouth landmarks to identify possible yawning. A duration threshold helps reduce detections caused by brief mouth movements.

### Head-Pose Handling
Facial landmark and head-pose measurements help the system continue operating when the face is tilted or turned. Measurements are treated as unreliable when head rotation exceeds the configured limit.

### Alarm and Alerts
An audio alarm is triggered by the configured prolonged eye-closure condition. Visual alerts can indicate detected eye-closure and possible-yawning events.

### Flask Dashboard
The dashboard provides a browser-based interface for live monitoring, real-time alerts, analytics and session reports.

### Evaluation and Logging
CSV files store detection events and timing measurements. Evaluation scripts calculate classification metrics, analyze detection delays and generate visual reports.

## 5. Project Structure

```text
Driver-Dizziness-Detection/
├── main.py
├── web_app.py
├── templates/
├── static/
├── evaluation/
│   ├── logs/
│   ├── reports/
│   │   └── evaluation_charts/
│   ├── scripts/
│   │   ├── evaluate.py
│   │   ├── evaluate_complete.py
│   │   ├── evaluate_labelled_tests.py
│   │   ├── evaluate_real_tests.py
│   │   ├── evaluate_timing.py
│   │   ├── evaluate_yawn_tests.py
│   │   └── plot_evaluation_results.py
│   └── test_data/
├── backup/
│   └── before_cleanup/
├── detection_log.csv
├── event_timing_log.csv
├── dashboard_detection_log.csv
├── dashboard_event_timing_log.csv
└── README.md
```

The `templates/` and `static/` directories contain the dashboard files if those are their current locations in your project. Keep any additional existing project files in place.

## 6. Installation and Execution

### Prerequisites
- Python installed on the computer
- A working webcam
- Windows or another compatible operating system
- The project dependencies installed in the virtual environment

### Activate the virtual environment

Open PowerShell in the project directory:

```powershell
cd D:\Driver-Dizziness-Detection
.\.venv\Scripts\Activate.ps1
```

### Run the detection program

```powershell
python .\main.py
```

### Run the web dashboard

Stop the standalone detection program first if it is using the webcam. Then run:

```powershell
python .\web_app.py
```

Open the local address displayed in the terminal, typically `http://127.0.0.1:5000`.

### Run the evaluation scripts

For example:

```powershell
python .\evaluation\scripts\evaluate_real_tests.py
python .\evaluation\scripts\evaluate_timing.py
python .\evaluation\scripts\evaluate_complete.py
python .\evaluation\scripts\plot_evaluation_results.py
```

## 7. Evaluation Results

The current test runs produced the following results.

### Real eye-closure and yawning test datasets

Each dataset contained five test cases.

| Metric | Eye closure | Yawning |
|---|---:|---:|
| True positives | 2 | 2 |
| False positives | 0 | 0 |
| False negatives | 0 | 0 |
| True negatives | 3 | 3 |
| Precision | 100% | 100% |
| Recall | 100% | 100% |
| F1-score | 100% | 100% |
| False alarm rate | 0% | 0% |

These results describe the five cases in each dataset only. Detection-delay measurements were unavailable for these particular tests because timestamps were missing.

### Labelled test datasets

| Metric | Eye closure | Yawning |
|---|---:|---:|
| True positives | 5 | 5 |
| False positives | 1 | 1 |
| False negatives | 1 | 1 |
| True negatives | 3 | 3 |
| Precision | 83.33% | 83.33% |
| Recall | 83.33% | 83.33% |
| F1-score | 83.33% | 83.33% |
| False alarm rate | 25% | 25% |
| Mean detection delay | 1.64 s | 1.24 s |

### Timing evaluation

| Event | Number of events | Mean delay |
|---|---:|---:|
| Eye closure | 3 | 1.542 s |
| Possible yawning | 4 | 1.007 s |

The timing results are based on the events recorded in the current timing log. The different datasets and evaluation procedures should not be treated as interchangeable.

## 8. Testing Performed

The following checks have been completed successfully during project verification:

- Evaluation scripts execute after the folder reorganization.
- Evaluation reports and charts are generated in the reports directory.
- The main webcam detection application runs.
- Eye-closure detection and the configured alarm work.
- Possible-yawning detection works.
- Detection continues during tested head tilts and left/right turns.
- The Flask dashboard opens and its tested features work.

These are functional checks, not a comprehensive safety validation.

## 9. Limitations

- Small test datasets are insufficient to establish general real-world accuracy.
- Lighting, camera placement, face occlusion, glasses and individual facial characteristics may affect detection.
- Speaking, eating and other mouth movements may cause false yawning detections.
- Head poses outside the reliable measurement range may reduce detection quality.
- Detection delays depend on configured thresholds and processing performance.
- The application should not be relied upon as the sole driver-safety mechanism.

## 10. Future Improvements

- Collect larger, more diverse and independently labelled test datasets.
- Evaluate performance under different lighting conditions and camera angles.
- Improve false-positive handling for yawning and eye closure.
- Add configurable user settings and session management.
- Improve historical analytics and report export.
- Measure processing speed and resource usage.
- Conduct systematic testing before any real-world deployment.

## 11. Conclusion

The Driver Dizziness Detection System demonstrates the use of computer vision and facial landmarks to monitor potential signs of driver drowsiness. By combining eye-closure detection, possible-yawning detection, audio alerts, CSV-based evaluation and a Flask dashboard, the project provides a foundation for further development and testing of driver alertness monitoring systems.