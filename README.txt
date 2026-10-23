DRIVER DIZZINESS DETECTION — FLASK FRONTEND

This package adds a Flask dashboard and does NOT replace your existing main.py.
It ports the latest known detection logic from your project conversation into web_app.py
so Flask can serve the camera stream and API status while the detector runs in a worker thread.

FILES TO COPY into D:\Driver-Dizziness-Detection:
  web_app.py
  templates\index.html
  static\app.js
  static\style.css
  requirements-frontend.txt (optional)

SETUP (PowerShell):
  cd D:\Driver-Dizziness-Detection
  .\.venv\Scripts\Activate.ps1
  python -m pip install "Flask>=3.0,<4.0"
  python web_app.py
Then open http://127.0.0.1:5000 in your browser.

Do not run main.py and web_app.py at the same time: only one process can normally use the webcam.
Your original main.py, evaluation scripts, and existing logs are not overwritten. The web app writes
its own dashboard_detection_log.csv and dashboard_event_timing_log.csv in the current project directory, using the
same column formats as the latest known script. Dashboard CSV files are recreated at the start of a web session. Your original detection_log.csv and event_timing_log.csv are not overwritten.

IMPORTANT VALIDATION:
The exact latest logic was reconstructed from the saved code available in the prior conversation,
not executed on your local machine. After copying, first test normal open eyes, a blink, 2-3 second
eye closure, mouth-open/yawn, head tilt, left/right head turn, alarm, Stop, and CSV downloads.
If you need strict line-by-line integration with your current local main.py, compare this web_app.py
with that file before relying on it. This is a research prototype, not a certified safety system.
