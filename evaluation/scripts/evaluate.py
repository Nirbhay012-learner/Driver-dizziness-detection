
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

# Load manually labeled test results
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent
TEST_DATA_DIR = PROJECT_DIR / "evaluation" / "test_data"

df = pd.read_csv(TEST_DATA_DIR / "test_results.csv")

actual = df["Actual"]
predicted = df["Predicted"]

# Compare actual and predicted results
print("Confusion Matrix:")
print(confusion_matrix(actual, predicted, labels=[0, 1]))

print("\nEvaluation Report:")
print(classification_report(
    actual,
    predicted,
    labels=[0, 1],
    target_names=["No eye closure event", "Eye closure event"],
    zero_division=0
))
