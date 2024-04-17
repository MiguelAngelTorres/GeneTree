from math import log
from sklearn.metrics import accuracy_score, roc_auc_score


def entropy(v):           # v es la proporcion de la clase (frec/total)
    if v == 0 or v == 1:
        return 0
    return v * log(v, 2)


def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def auc(y_true, proba_pred, labels):
    if len(labels) == 2:
        return roc_auc_score(y_true, proba_pred[:, 1])
    return roc_auc_score(y_true, proba_pred)
