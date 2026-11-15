import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


def ridge_kfold_RMSE_avg(y, X, n_folds, lambda_val) -> int:
    n_repeats = 100
    rkf = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats)
    RMSE_tot = 0
    it = 0
    for (train_index, test_index) in rkf.split(X):
        it += 1
        model = Ridge(alpha=lambda_val, fit_intercept=False)
        model.fit(X[train_index], y[train_index])
        #w = model.coef_
        #print(w.shape)
        y_pred = model.predict(X[test_index])

        RMSE = np.sqrt(mean_squared_error(y[test_index], y_pred))
        RMSE_tot += RMSE
    
    return RMSE_tot/it

if __name__ == "__main__":
    data = pd.read_csv('./task1a_do4bq81me/train.csv')

    # Retrieve only column y
    y = data["y"].to_numpy()

    # Drop column y so that only x_i are remaining and retrieve the x_i
    data = data.drop(columns="y")
    print(data.head())
    X = data.to_numpy()

    lambdas = [0.1, 1, 10, 100, 200]
    n_folds = 10
    RMSE_mat = np.zeros(len(lambdas))
    for index_lambda, elem in enumerate(lambdas):
        RMSE_avg = ridge_kfold_RMSE_avg(y, X, n_folds, elem)
        RMSE_mat[index_lambda] = RMSE_avg
        print(f"lambda {elem} is RMSE_avg: {RMSE_avg}")

    print(RMSE_mat)

    np.savetxt("./results.csv", RMSE_mat, fmt="%.12f")