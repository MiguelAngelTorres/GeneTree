from math import log
from sklearn.metrics import roc_auc_score
from numpy import equal

def entropy(v):           # v es la proporcion de la clase (frec/total)
    if v == 0 or v == 1:
        return 0
    return v * log(v, 2)


def accuracy(y_true, y_pred):
    score = equal(y_true, y_pred)
    score = sum(score)/len(score)
    return score  # Faster than accuracy_score(y_true, y_pred) from sklearn


def auc(y_true, proba_pred, labels):
    if len(labels) == 2:
        return roc_auc_score(y_true, proba_pred[:, 1])
    return roc_auc_score(y_true, proba_pred)
