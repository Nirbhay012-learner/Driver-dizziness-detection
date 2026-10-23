
import csv
from statistics import mean

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent
FILE_NAME = PROJECT_DIR / "evaluation" / "test_data" / "labelled_yawn_tests.csv"

tp = fp = fn = tn = 0
delays = []

with open(FILE_NAME, "r", newline="", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)

    required = {
        "test_id",
        "actual_event",
        "warning_detected",
        "event_start_sec",
        "warning_time_sec"
    }

    if not required.issubset(reader.fieldnames or []):
        raise ValueError("Check the CSV column names.")

    for row in reader:
        actual = int(row["actual_event"])
        detected = int(row["warning_detected"])

        if actual not in (0, 1) or detected not in (0, 1):
            raise ValueError(f"Invalid labels in {row['test_id']}")

        if actual == 1 and detected == 1:
            tp += 1

            if row["event_start_sec"] and row["warning_time_sec"]:
                delay = (
                    float(row["warning_time_sec"])
                    - float(row["event_start_sec"])
                )

                if delay < 0:
                    raise ValueError(
                        f"Warning precedes event in {row['test_id']}"
                    )

                delays.append(delay)

        elif actual == 0 and detected == 1:
            fp += 1
        elif actual == 1 and detected == 0:
            fn += 1
        else:
            tn += 1

precision = tp / (tp + fp) if tp + fp else 0
recall = tp / (tp + fn) if tp + fn else 0
f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0
)
false_alarm_rate = fp / (fp + tn) if fp + tn else 0

print("\n===== YAWNING EVALUATION =====")
print(f"True positives (TP):  {tp}")
print(f"False positives (FP): {fp}")
print(f"False negatives (FN): {fn}")
print(f"True negatives (TN):  {tn}")
print(f"\nPrecision:        {precision:.2%}")
print(f"Recall:           {recall:.2%}")
print(f"F1-score:         {f1:.2%}")
print(f"False alarms:     {fp}")
print(f"False alarm rate: {false_alarm_rate:.2%}")

if delays:
    print(f"Mean detection delay: {mean(delays):.2f} seconds")
else:
    print("Mean detection delay: unavailable")
