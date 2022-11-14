from lifelines import datasets
import numpy as np
from sklearn.model_selection import train_test_split
import time
from random_survival_forest.models import RandomSurvivalForest

rossi = datasets.load_rossi()

# Attention: duration column must be index 0, event column index 1 in y
y = rossi.loc[:, ["arrest", "week"]]
X = rossi.drop(["arrest", "week"], axis=1)
X, X_test, y, y_test = train_test_split(X, y, test_size=0.33, random_state=10)

print("Start training...")
start_time = time.time()
rsf = RandomSurvivalForest(n_estimators=10, n_jobs=-1, random_state=10)
rsf = rsf.fit(X, y)
print(f'--- {round(time.time() - start_time, 3)} seconds ---')

# ----------------------------------------------------------------
from random_survival_forest.scoring import concordance_index as concordance_index_rsf

y_pred = rsf.predict(X_test)
c_val = concordance_index_rsf(y_time=y_test["week"], y_pred=y_pred, y_event=y_test["arrest"])
print(f'C-index rsf metric {round(c_val, 3)}')

# ----------------------------------------------------------------
# from lifelines.utils import concordance_index as concordance_index_lf
#
# y_pred_form = [list(pred.index)[list(pred).index(max(pred))] for pred in y_pred]
# c_val = concordance_index_lf(y_test["week"], y_pred_form)
# print(f'C-index lifelines metric {round(c_val, 3)}')
# residuals = np.array(y_pred_form) - y_test["week"]
# print(f'Mean week error = {round(np.abs(residuals).mean(), 2)}')
