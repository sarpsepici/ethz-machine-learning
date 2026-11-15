# Introduction to Machine Learning

Course projects for *Introduction to Machine Learning*, ETH Zurich, spring 2023. Group projects in
Python (scikit-learn, PyTorch).

| Task | Problem | Approach |
| --- | --- | --- |
| [1a](task1a-ridge-regression) | Cross-validated error of ridge regression for several regularization strengths | Repeated K-fold cross-validation |
| [1b](task1b-feature-transformations) | Linear regression on 21 feature transformations (linear, quadratic, exponential, cosine, constant) | Ridge regression with built-in cross-validation (`RidgeCV`) |
| [2](task2-price-prediction) | Predict prices in Switzerland from incomplete data from other countries | KNN imputation of missing values, then Gaussian process regression with the kernel chosen by repeated K-fold cross-validation |
| [3](task3-food-similarity) | Decide which of two dishes tastes more like a third, from food images | ResNet50 image embeddings fed into a small fully connected network |
| [4](task4-transfer-learning) | Predict a molecular property from only a few labelled examples | A network pretrained on a large related dataset extracts features, then support vector regression tuned with grid search |

Each folder has one `solution.py`. The datasets are not included.
