from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from src.config import LABELS


def return_balanced_accuracy_score(y_true, y_pred):
    """
    Compute the balanced accuracy score.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.

    Returns:
    float: Balanced accuracy score.
    """
    return balanced_accuracy_score(y_true, y_pred)

def return_confusion_matrix(y_true, y_pred):
    """
    Compute the confusion matrix.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.

    matrix schema:
                     pred NORMAL   pred PNEUMONIA
    true NORMAL          [0,0]          [0,1]
    true PNEUMONIA       [1,0]          [1,1]

    Returns:
    ndarray: Confusion matrix.
    """
    return confusion_matrix(y_true, y_pred, labels=LABELS)

