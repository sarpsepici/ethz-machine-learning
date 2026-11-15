import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, ExpSineSquared, Matern, RationalQuadratic, WhiteKernel
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold



if __name__ == "__main__":

# Load the training data and preprocess

    train_df = pd.read_csv("./train.csv")

    # Encode the season with an integer
    le = LabelEncoder()
    le.fit(train_df['season'])
    train_df['season'] = le.transform(train_df['season'])

    imputer = KNNImputer(n_neighbors=5)

    # Now impute the missing data
    train_df = pd.DataFrame(imputer.fit_transform(train_df), columns=train_df.columns)
    # print(train_df.head(2))

    y_train = train_df['price_CHF'].to_numpy()

    train_df = train_df.drop(columns="price_CHF")
    X_train = train_df.to_numpy()

# Load the test data and preprocess

    test_df = pd.read_csv("./test.csv")

    test_df['season'] = le.transform(test_df['season'])

    # test_df = pd.DataFrame(SimpleImputer().fit_transform(test_df), columns=test_df.columns)

    X_test = imputer.fit_transform(test_df)

# Use cross validation to select the best kernel

    n_folds = 10
    n_repeats = 10
    list_kernel = [RationalQuadratic() + Matern(), RationalQuadratic() + WhiteKernel()]
    rkf = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats)
    best_kernel = [np.infty, None]
    
    for kernel in list_kernel:
        print(kernel)
        RMSE_tot = 0
        gpr = GaussianProcessRegressor(kernel=kernel)
        for i,(train_index, test_index) in enumerate(rkf.split(X_train)):
            gpr.fit(X_train[train_index], y_train[train_index])
            y_pred = gpr.predict(X_train[test_index])
            
            RMSE = np.sqrt(mean_squared_error(y_train[test_index], y_pred))
            RMSE_tot += RMSE
        RMSE_avg = 1/(n_folds*n_repeats) * RMSE_tot
        if RMSE_avg < best_kernel[0]:
            best_kernel[0] = RMSE_avg
            best_kernel[1] = kernel
        print(f"kernel {kernel} is RMSE_avg: {RMSE_avg}")
    print(best_kernel)

# Fit and predict with the best kernel

    gpr = GaussianProcessRegressor(kernel=best_kernel[1])
    gpr.fit(X_train, y_train)
    y_test = gpr.predict(X_test)
    np.savetxt("./results.csv", y_test, fmt="%.12f", header='price_CHF', comments='')
