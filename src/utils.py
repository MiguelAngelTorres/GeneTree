from math import log
from sklearn.metrics import accuracy_score

def entropy(v):           # v es la proporcion de la clase (frec/total)
    if v == 0 or v == 1:
        return 0
    return v * log(v, 2)


def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
