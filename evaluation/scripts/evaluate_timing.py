
import csv
import statistics
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
file_path = PROJECT_DIR / "event_timing_log.csv"

if not file_path.exists():
    print("ERROR: event_timing_log.csv not found.")
    raise SystemExit

delays = {
    "EYE_CLOSURE": [],
    "POSSIBLE_YAWN": [],
}

with open(file_path, newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    required = {"Event_Type", "Detection_Delay_Sec"}

    if not reader.fieldnames or not required.issubset(reader.fieldnames):
        print("ERROR: CSV columns are missing.")
        raise SystemExit

    for row in reader:
        event = row["Event_Type"].strip()

        try:
            delay = float(row["Detection_Delay_Sec"])
        except (ValueError, TypeError):
            continue

        if event in delays and delay >= 0:
            delays[event].append(delay)

for event, values in delays.items():
    print(f"\n===== {event} =====")

    if not values:
        print("No valid timing records.")
        continue

    print(f"Number of events: {len(values)}")
    print(f"Mean delay: {statistics.mean(values):.3f} seconds")
    print(f"Median delay: {statistics.median(values):.3f} seconds")
    print(f"Minimum delay: {min(values):.3f} seconds")
    print(f"Maximum delay: {max(values):.3f} seconds")
