import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, LinearRegression, Lasso, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    data = pd.read_csv('./task1b_ql4jfi6af0/train.csv')
    y = data["y"].to_numpy()

    data = data.drop(columns=["Id", "y"])
    X = data.to_numpy()

    X_new = np.concatenate((X, np.power(X,2), np.exp(X), np.cos(X)), axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X_new, y, train_size=0.8, shuffle=False)


    alphas = [i/10 for i in range(1, 1000)]
    ridge_reg = RidgeCV(alphas=alphas)
    ridge_reg.fit(X_new, y)
    y_pred_ridge = ridge_reg.predict(X_test)
    print(ridge_reg.alpha_)

    w_res = np.array(ridge_reg.coef_).T
    intercept = np.array([ridge_reg.intercept_])
    print(w_res.shape)
    print(intercept.shape)
    w_res = np.hstack([w_res, np.array(ridge_reg.intercept_)])
    print(w_res.shape)
    np.savetxt("./results.csv", w_res, fmt="%.12f")
