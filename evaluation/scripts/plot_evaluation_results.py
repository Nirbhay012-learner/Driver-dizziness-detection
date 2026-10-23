import csv
import os
from pathlib import Path
import matplotlib.pyplot as plt
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent
TEST_DATA_DIR = PROJECT_DIR / "evaluation" / "test_data"
OUTPUT_DIR = PROJECT_DIR / "evaluation" / "reports" / "evaluation_charts"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
def evaluate(file_name, start_column):
    tp = fp = fn = tn = 0
    delays = []

    with open(file_name, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        for row in reader:
            actual = int(row["actual_event"])
            detected = int(row["warning_detected"])

            if actual == 1 and detected == 1:
                tp += 1

                if row[start_column] and row["warning_time_sec"]:
                    delay = (
                        float(row["warning_time_sec"])
                        - float(row[start_column])
                    )
                    if delay >= 0:
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

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "False alarm rate": false_alarm_rate,
        "Mean delay": sum(delays) / len(delays) if delays else None
    }


def plot_confusion_matrix(result, title, filename):
    matrix = [
        [result["TN"], result["FP"]],
        [result["FN"], result["TP"]]
    ]

    fig, ax = plt.subplots()
    ax.imshow(matrix)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No warning", "Warning"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No event", "Event"])

    ax.set_xlabel("System prediction")
    ax.set_ylabel("Actual label")
    ax.set_title(title)

    for row in range(2):
        for col in range(2):
            ax.text(
                col, row, str(matrix[row][col]),
                ha="center", va="center"
            )

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200)
    plt.close(fig)


eye = evaluate(
    TEST_DATA_DIR / "labelled_eye_tests.csv",
    "closure_start_sec"
)
yawn = evaluate(
    TEST_DATA_DIR / "labelled_yawn_tests.csv",
    "event_start_sec"
)
plot_confusion_matrix(
    eye, "Eye-Closure Confusion Matrix", "eye_confusion_matrix.png"
)
plot_confusion_matrix(
    yawn, "Yawning Confusion Matrix", "yawn_confusion_matrix.png"
)

metrics = ["Precision", "Recall", "F1-score", "False alarm rate"]
x = list(range(len(metrics)))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(
    [i - width / 2 for i in x],
    [eye[m] * 100 for m in metrics],
    width, label="Eye closure"
)
ax.bar(
    [i + width / 2 for i in x],
    [yawn[m] * 100 for m in metrics],
    width, label="Yawning"
)

ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=10)
ax.set_ylabel("Percentage (%)")
ax.set_ylim(0, 100)
ax.set_title("Driver Dizziness Detection - Evaluation Metrics")
ax.legend()

fig.tight_layout()
fig.savefig(
    os.path.join(OUTPUT_DIR, "performance_comparison.png"),
    dpi=200
)
plt.close(fig)

print("Charts saved in:", OUTPUT_DIR)
print("\nEye-closure metrics:", eye)
print("Yawning metrics:", yawn)

print("\nMean eye-closure delay:", eye["Mean delay"])
print("Mean yawning delay:", yawn["Mean delay"])
