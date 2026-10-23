import csv
import statistics
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
TEST_DATA_DIR = PROJECT_DIR / "evaluation" / "test_data"
OUTPUT_DIR = PROJECT_DIR / "evaluation" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate(file_name, title, start_column):
    path = TEST_DATA_DIR / file_name

    if not path.exists():
        print(f"\nERROR: File not found: {file_name}")
        return None

    required = {
        "test_id", "actual_event", "warning_detected",
        start_column, "warning_time_sec"
    }
    counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    delays = []
    errors = []
    total = 0

    with open(path, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            print(f"\nERROR: {file_name} has no header.")
            return None

        reader.fieldnames = [
            name.strip() if name else name
            for name in reader.fieldnames
        ]
        missing = required - set(reader.fieldnames)

        if missing:
            print(f"\nERROR: {file_name} is missing: {sorted(missing)}")
            return None

        for line, row in enumerate(reader, start=2):
            actual_text = (row.get("actual_event") or "").strip()
            detected_text = (row.get("warning_detected") or "").strip()

            # Skip unused template rows.
            if not actual_text and not detected_text:
                continue

            if actual_text not in ("0", "1") or detected_text not in ("0", "1"):
                errors.append(f"Line {line}: incomplete or invalid labels")
                continue

            actual = int(actual_text)
            detected = int(detected_text)
            total += 1

            if actual == 1 and detected == 1:
                counts["TP"] += 1
                start = (row.get(start_column) or "").strip()
                warning = (row.get("warning_time_sec") or "").strip()

                if start and warning:
                    try:
                        delay = float(warning) - float(start)
                        if delay >= 0:
                            delays.append(delay)
                        else:
                            errors.append(f"Line {line}: warning precedes event")
                    except ValueError:
                        errors.append(f"Line {line}: invalid timestamps")

            elif actual == 0 and detected == 1:
                counts["FP"] += 1
            elif actual == 1 and detected == 0:
                counts["FN"] += 1
            else:
                counts["TN"] += 1

    print(f"\n===== {title} =====")
    print(f"File: {file_name}")
    print(f"Tests evaluated: {total}")

    if errors:
        print(f"Data issues: {len(errors)}")
        for error in errors[:10]:
            print(f"  - {error}")

    if total == 0:
        print("No completed test records to evaluate.")
        return None

    tp, fp, fn, tn = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    far = fp / (fp + tn) if fp + tn else 0

    print(f"True positives:  {tp}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")
    print(f"True negatives:  {tn}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1-score: {f1:.2%}")
    print(f"False alarm rate: {far:.2%}")

    if delays:
        print(f"Timed detections: {len(delays)}")
        print(f"Mean delay: {statistics.mean(delays):.2f} seconds")
        print(f"Median delay: {statistics.median(delays):.2f} seconds")
    else:
        print("Detection delay: unavailable (no valid timed detections)")

    return {
        "test_file": file_name,
        "tests": total,
        **counts,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "false_alarm_rate": round(far, 4),
        "timed_detections": len(delays),
        "mean_delay_sec": round(statistics.mean(delays), 3) if delays else "",
        "median_delay_sec": round(statistics.median(delays), 3) if delays else "",
    }


datasets = [
    ("real_eye_tests.csv", "REAL EYE-CLOSURE TESTS", "closure_start_sec"),
    ("real_yawn_tests.csv", "REAL YAWNING TESTS", "event_start_sec"),
    ("real_eye_tests_extended.csv", "EXTENDED REAL EYE-CLOSURE TESTS", "event_start_sec"),
    ("real_yawn_tests_extended.csv", "EXTENDED REAL YAWNING TESTS", "event_start_sec"),
]

results = []
for file_name, title, start_column in datasets:
    result = evaluate(file_name, title, start_column)
    if result:
        results.append(result)

if results:
    report_path = OUTPUT_DIR / "evaluation_report.csv"
    with open(report_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nEvaluation report saved to: {report_path}")
