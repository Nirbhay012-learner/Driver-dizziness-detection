
import csv
import statistics
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
TEST_DATA_DIR = PROJECT_DIR / "evaluation" / "test_data"
LOG_DIR = PROJECT_DIR
OUTPUT_DIR = PROJECT_DIR / "evaluation" / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "complete_evaluation_report.csv"


def read_csv(filename):
    if filename == "event_timing_log.csv":
        path = LOG_DIR / filename
    else:
        path = TEST_DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    with open(path, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError(f"{filename} is empty.")

        reader.fieldnames = [
            name.strip() if name else name
            for name in reader.fieldnames
        ]

        return list(reader), set(reader.fieldnames)


def evaluate_classification(filename, title, start_column):
    rows, columns = read_csv(filename)

    required = {
        "actual_event",
        "warning_detected",
        start_column,
        "warning_time_sec",
    }

    missing = required - columns
    if missing:
        raise ValueError(
            f"{filename} is missing columns: {', '.join(sorted(missing))}"
        )

    tp = fp = fn = tn = 0
    valid_rows = 0

    for line_number, row in enumerate(rows, start=2):
        if not any(str(v or "").strip() for v in row.values()):
            continue

        try:
            actual = int(row["actual_event"])
            detected = int(row["warning_detected"])

            if actual not in (0, 1) or detected not in (0, 1):
                raise ValueError
        except (ValueError, TypeError):
            raise ValueError(
                f"Invalid labels in {filename}, line {line_number}. "
                "Labels must be 0 or 1."
            )

        valid_rows += 1

        if actual == 1 and detected == 1:
            tp += 1
        elif actual == 0 and detected == 1:
            fp += 1
        elif actual == 1 and detected == 0:
            fn += 1
        else:
            tn += 1

    if valid_rows == 0:
        raise ValueError(f"No test records found in {filename}.")

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    false_alarm_rate = fp / (fp + tn) if fp + tn else 0.0
    accuracy = (tp + tn) / valid_rows

    results = {
        "Tests evaluated": valid_rows,
        "True positives": tp,
        "False positives": fp,
        "False negatives": fn,
        "True negatives": tn,
        "Accuracy (%)": round(accuracy * 100, 2),
        "Precision (%)": round(precision * 100, 2),
        "Recall (%)": round(recall * 100, 2),
        "F1-score (%)": round(f1 * 100, 2),
        "False alarm rate (%)": round(false_alarm_rate * 100, 2),
    }

    print(f"\n===== {title}: CLASSIFICATION =====")
    for metric, value in results.items():
        print(f"{metric}: {value}")

    return results


def evaluate_timing(event_type, title):
    rows, columns = read_csv("event_timing_log.csv")

    required = {"Event_Type", "Detection_Delay_Sec"}
    missing = required - columns

    if missing:
        raise ValueError(
            "event_timing_log.csv is missing columns: "
            + ", ".join(sorted(missing))
        )

    delays = []

    for row in rows:
        if row.get("Event_Type", "").strip() != event_type:
            continue

        try:
            delay = float(row["Detection_Delay_Sec"])
        except (ValueError, TypeError):
            continue

        if delay >= 0:
            delays.append(delay)

    print(f"\n===== {title}: DETECTION TIMING =====")

    if not delays:
        print("No valid timing records found.")
        return {
            "Timing events": 0,
            "Mean delay (seconds)": "",
            "Median delay (seconds)": "",
            "Minimum delay (seconds)": "",
            "Maximum delay (seconds)": "",
        }

    results = {
        "Timing events": len(delays),
        "Mean delay (seconds)": round(statistics.mean(delays), 3),
        "Median delay (seconds)": round(statistics.median(delays), 3),
        "Minimum delay (seconds)": round(min(delays), 3),
        "Maximum delay (seconds)": round(max(delays), 3),
    }

    for metric, value in results.items():
        print(f"{metric}: {value}")

    return results


def main():
    eye_classification = evaluate_classification(
        "real_eye_tests.csv",
        "EYE CLOSURE",
        "closure_start_sec",
    )

    yawn_classification = evaluate_classification(
        "real_yawn_tests.csv",
        "POSSIBLE YAWNING",
        "event_start_sec",
    )

    eye_timing = evaluate_timing(
        "EYE_CLOSURE",
        "EYE CLOSURE",
    )

    yawn_timing = evaluate_timing(
        "POSSIBLE_YAWN",
        "POSSIBLE YAWNING",
    )

    report_rows = []

    reports = [
        ("Eye closure", "Classification", eye_classification),
        ("Possible yawning", "Classification", yawn_classification),
        ("Eye closure", "Timing", eye_timing),
        ("Possible yawning", "Timing", yawn_timing),
    ]

    for category, section, metrics in reports:
        for metric, value in metrics.items():
            report_rows.append({
                "Category": category,
                "Section": section,
                "Metric": metric,
                "Value": value,
            })

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = ["Category", "Section", "Metric", "Value"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    print("\n====================================")
    print("COMBINED EVALUATION COMPLETED")
    print(f"Report saved to: {OUTPUT_FILE}")
    print("====================================")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"\nERROR: {error}")
