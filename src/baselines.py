from src.data import return_split
from src.evaluate import return_balanced_accuracy_score, return_confusion_matrix
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

"""
First baseline, dummy classifier


"""

train_df, val_df, test_df = return_split()

biggest_class = train_df["label"].value_counts().idxmax()
predicted_val_dummy = pd.Series([biggest_class] * len(val_df), index=val_df.index)
predicted_test_dummy = pd.Series([biggest_class] * len(test_df), index=test_df.index)

bal_acc_score_val_dummy = return_balanced_accuracy_score(val_df["label"], predicted_val_dummy)
bal_acc_score_test_dummy = return_balanced_accuracy_score(test_df["label"], predicted_test_dummy)

confusion_matrix_val = return_confusion_matrix(val_df["label"], predicted_val_dummy)
confusion_matrix_test = return_confusion_matrix(test_df["label"], predicted_test_dummy)

print(f"Baseline class (majority): {biggest_class}")
print(f"Val balanced accuracy: {bal_acc_score_val_dummy:.4f}")
print(f"Test balanced accuracy:  {bal_acc_score_test_dummy:.4f}")
print("Val confusion matrix:")
print(confusion_matrix_val)
print("Test confusion matrix:")
print(confusion_matrix_test)
print("-----------------------------------------------------------------")
"""
Second baseline, logistic regression on acquisition metadata only.

The model is fitted on width, height and aspect ratio. No pixel is ever read.
Its purpose is not to classify well, but to measure how much of the task is
solvable from acquisition provenance rather than from anatomy.

"""


X_train = train_df[["width", "height", "aspect_ratio"]]
y_train = train_df["label"]

X_val = val_df[["width", "height", "aspect_ratio"]]
y_val = val_df["label"]

X_test = test_df[["width", "height", "aspect_ratio"]]
y_test = test_df["label"]

class_weights = [None, "balanced"]

results = []

for cw in class_weights:
    pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42, class_weight= cw))])
    pipe.fit(X_train, y_train)
    predicted_val = pipe.predict(X_val)
    predicted_test = pipe.predict(X_test)
    results.append({
        "class_weight": cw,
        "bal_acc_val": return_balanced_accuracy_score(y_val, predicted_val),
        "bal_acc_test": return_balanced_accuracy_score(y_test, predicted_test),
        "confusion_matrix_val": return_confusion_matrix(y_val, predicted_val),
        "confusion_matrix_test": return_confusion_matrix(y_test, predicted_test)
    })

results_df = pd.DataFrame(results)
print(results_df)

