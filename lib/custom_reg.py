import numpy as np

ASSYM_COEF = 10.


def assym_obj_fn(y_true, y_pred):
    residual = (y_true - y_pred).astype("float")
    grad = np.where(residual > 0, -2.0 * ASSYM_COEF * residual, -2.0 * residual)
    hess = np.where(residual > 0, 2.0 * ASSYM_COEF, 2.0)
    return grad, hess


def assym_valid_fn(y_true, y_pred):
    residual = (y_true - y_pred).astype("float")
    loss = np.where(residual > 0, (residual ** 2.0) * ASSYM_COEF, residual ** 2.0)
    return "custom_asymmetric_eval", np.mean(loss), False
