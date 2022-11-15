import numpy as np
from sklearn.metrics import r2_score


class GbmSymLosses:
    def __init__(self):
        pass

    def train_fn(self, y_true, y_pred):
        residual = (y_true - y_pred).astype("float")
        grad = -2 * residual
        hess = [2.0] * len(residual)
        return grad, hess

    def valid_fn(self, y_true, y_pred):
        residual = (y_true - y_pred).astype("float")
        loss = residual ** 2
        return "custom_symmetric_eval", np.mean(loss), False


class GbmAsymLosses:
    def __init__(self, assym_coef: float):
        self._assym_coef = assym_coef

    def train_fn(self, y_true, y_pred):
        residual = (y_true - y_pred).astype("float")
        grad = np.where(residual < 0, -2 * self._assym_coef * residual, -2 * residual)
        hess = np.where(residual < 0, 2 * self._assym_coef, 2.0)
        return grad, hess

    def valid_fn(self, y_true, y_pred):
        residual = (y_true - y_pred).astype("float")
        loss = np.where(residual < 0, (residual ** 2) * self._assym_coef, residual ** 2)
        return "custom_asymmetric_eval", np.mean(loss), False


class Losses:
    @staticmethod
    def mae(arr1, arr2) -> float:
        return np.abs((arr1 - arr2)).mean()

    @staticmethod
    def rmse(arr1, arr2) -> float:
        return np.sqrt(Losses.mse(arr1, arr2))

    @staticmethod
    def mse(arr1, arr2) -> float:
        return ((arr1 - arr2) ** 2).mean()

    @staticmethod
    def r(pred, y):
        return np.corrcoef(x=pred, y=y)[0][1]

    @staticmethod
    def r_my(pred, y):
        pred_resid = pred - np.mean(pred)
        y_resid = y - np.mean(y)
        num = np.sum(pred_resid * y_resid)
        den = np.sqrt(np.sum(pred_resid ** 2) * (np.sum(y_resid ** 2)))
        return num / den

    @staticmethod
    def r2(pred, y):
        return r2_score(y_true=y, y_pred=pred)

    @staticmethod
    def r2_my(pred, y):
        num = np.sum((y - pred) ** 2)
        den = np.sum((y - np.mean(y)) ** 2)
        return 1 - num / den
