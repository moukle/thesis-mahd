from tensorflow.keras.metrics import Recall, Precision


def recall(y_true, y_pred):
    re = Recall()
    re.update_state(y_true, y_pred)
    r = re.result()

    return r


def precision(y_true, y_pred):
    pr = Precision()
    pr.update_state(y_true, y_pred)
    p = pr.result()

    return p


def f1(y_true, y_pred):
    r = recall(y_true, y_pred)
    p = precision(y_true, y_pred)
    eps = 1e-6

    return 2 * (p * r) / (p + r + eps)
