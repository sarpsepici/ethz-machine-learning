# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, ElasticNetCV, ElasticNet, BayesianRidge, SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingRegressor

# Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
print(f"Is MPS available? {torch.backends.mps.is_available()}")

# Set the device
device = "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device)
print(f"Using device: {device}")

def load_data():
    """
    This function loads the data from the csv files and returns it as numpy arrays.

    input: None
    
    output: x_pretrain: np.ndarray, the features of the pretraining set
            y_pretrain: np.ndarray, the labels of the pretraining set
            x_train: np.ndarray, the features of the training set
            y_train: np.ndarray, the labels of the training set
            x_test: np.ndarray, the features of the test set
    """
    x_pretrain = pd.read_csv("pretrain_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_pretrain = pd.read_csv("pretrain_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_train = pd.read_csv("train_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_train = pd.read_csv("train_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_test = pd.read_csv("test_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1)
    return x_pretrain, y_pretrain, x_train, y_train, x_test

class Net(nn.Module):
    """
    The model class, which defines our feature extractor used in pretraining.
    """
    def __init__(self, in_features):
        """
        The constructor of the model.
        """
        super().__init__()

        self.act = nn.Sigmoid()
        
        self.fc1 = nn.Linear(in_features,1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.d1 = nn.Dropout(p=0.3)

        self.fc2 = nn.Linear(1024,768)
        self.bn2 = nn.BatchNorm1d(768)
        self.d2 = nn.Dropout(p=0.2)

        self.fc3 = nn.Linear(768,512)
        # self.bn3 = nn.BatchNorm1d(512)
        # self.d3 = nn.Dropout(p=0.05)

        # self.fc4 = nn.Linear(512,256)
        # self.bn4 = nn.BatchNorm1d(32)
        # self.d4 = nn.Dropout(p=0.2)

        # self.fc5 = nn.Linear(64,32)
        # self.bn5 = nn.BatchNorm1d(32)

        self.out = nn.Linear(512, 1)

    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        # TODO: Implement the forward pass of the model, in accordance with the architecture 
        # defined in the constructor.
        # x = F.relu(self.d1(self.bn1(self.fc1(x))))
        # x = F.relu(self.d2(self.bn2(self.fc2(x))))
        # x = F.relu(self.d3(self.bn3(self.fc3(x))))
        # x = F.relu(self.d4(self.bn4(self.fc4(x))))
        # x = F.relu(self.bn5(self.fc5(x)))
        # x = (self.out(x))

        # x = F.relu(self.d1((self.fc1(x))))
        # x = F.relu(self.d2((self.fc2(x))))
        # x = F.relu(self.d3((self.fc3(x))))
        # x = F.relu(self.d4((self.fc4(x))))
        # x = F.relu((self.fc5(x)))
        # x = (self.out(x))

        x = self.d1(self.act(self.bn1(self.fc1(x))))
        x = self.d2(self.act(self.bn2(self.fc2(x))))
        # x = self.d3(self.act(self.bn3(self.fc3(x))))
        # x = self.d3(self.fc3(x))
        x = (self.fc3(x))
        x = (self.out(x))
        return x
    
def make_feature_extractor(x, y, batch_size=256, eval_size=1000):
    """
    This function trains the feature extractor on the pretraining data and returns a function which
    can be used to extract features from the training and test data.

    input: x: np.ndarray, the features of the pretraining set
              y: np.ndarray, the labels of the pretraining set
                batch_size: int, the batch size used for training
                eval_size: int, the size of the validation set
            
    output: make_features: function, a function which can be used to extract features from the training and test data
    """

    # Pretraining data loading
    in_features = x.shape[-1]
    x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=eval_size, random_state=0, shuffle=True)
    x_tr, x_val = torch.tensor(x_tr, dtype=torch.float, device=device), torch.tensor(x_val, dtype=torch.float, device=device)
    y_tr, y_val = torch.tensor(y_tr, dtype=torch.float, device=device), torch.tensor(y_val, dtype=torch.float, device=device)

    # model declaration
    model = Net(in_features)
    model.to(device)
    model.train()

    n_epochs = 10

    loss_function = nn.HuberLoss()
    # loss_function = nn.MSELoss()
    # loss_function = nn.L1Loss()
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    losses = []
    
    for epoch in range(n_epochs):
        perm = torch.randperm(x_tr.size()[0])

        for i in range(0,x_tr.size()[0], batch_size):
            optimizer.zero_grad()
            indices = perm[i:i+batch_size] #range(i,i+batch_size,1)
            batch_x, batch_y = x_tr[indices], y_tr[indices]
            output = model(batch_x)
            loss = loss_function(output.squeeze().float(), batch_y)
            losses.append(loss)
            loss.backward()
            optimizer.step()

            # if i % 100 == 0:
            #     print('Epoch {}, Batch idx {}, loss {}'.format(
            #         epoch, i, loss.item()))
                
        model.eval()
        # Iterate over the test data
        with torch.no_grad(): # We don't need to compute gradients for testing
            predicted = model(x_val)
            print('validation loss: {}'.format(loss_function(predicted.squeeze().float(), y_val).item()))
        # scheduler.step()
                
    model.eval()
    # Iterate over the test data
    with torch.no_grad(): # We don't need to compute gradients for testing
        predicted = model(x_val)
        print('validation loss: {}'.format(loss_function(predicted.squeeze().float(), y_val).item()))

    model = torch.nn.Sequential(*(list(model.children())[:-2])) # removes last layer, we should have 32 features

    def make_features(x):
        """
        This function extracts features from the training and test data, used in the actual pipeline 
        after the pretraining.

        input: x: np.ndarray, the features of the training or test set

        output: features: np.ndarray, the features extracted from the training or test set, propagated
        further in the pipeline
        """
        nonlocal model
        model.eval()
        # embeddings = []
        # feature_size = 2048 # Dummy variable, replace with the actual embedding
        # num_images = len(train_dataset)
        # embeddings = np.zeros((num_images, embedding_size))
        with torch.no_grad():
            features = model(x).cpu().numpy()
        return features

    return make_features

class PretrainedFeatures(BaseEstimator, TransformerMixin):
    """
    The wrapper class for Pretraining pipeline.
    """
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        assert self.feature_extractor is not None
        X = torch.tensor(X, dtype=torch.float, device=device)
        return self.feature_extractor(X)

def get_regression_model():
    """
    This function returns the regression model used in the pipeline.

    input: None

    output: model: sklearn compatible model, the regression model
    """
    # TODO: Implement the regression model. It should be able to be trained on the features extracted
    # by the feature extractor.
    # model = Ridge(alpha=1.0)
    # model = ElasticNetCV(random_state=0)
    # model = ElasticNet(random_state=0)
    model = SVR(kernel="rbf", C = 47, gamma = 0.0001)
    # k=['rbf','linear','poly','sigmoid']
    # c= range(1,100)
    # g=np.arange(1e-4,1e-2,0.0001)
    # g=g.tolist()
    # param_grid=dict(kernel=k, C=c, gamma=g)
    # grid = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    # grid.fit(X, y)  
    # model = GradientBoostingRegressor(random_state=0)
    # model = BayesianRidge()
    # model = SGDRegressor(max_iter=1000, tol=1e-3)
    # parameters = {'kernel' : ('linear', 'poly', 'rbf', 'sigmoid'),'C' : [1,5,10],'degree' : [2,8],'coef0' : [0.01,10,0.5],'gamma' : ('auto','scale')}
    # model_test = GridSearchCV(model, parameters)
    return model

# Main function. You don't have to change this
if __name__ == '__main__':
    # Load data
    x_pretrain, y_pretrain, x_train, y_train, x_test = load_data()
    print("Data loaded!")
    # Utilize pretraining data by creating feature extractor which extracts lumo energy 
    # features from available initial features
    feature_extractor =  make_feature_extractor(x_pretrain, y_pretrain)
    
    pipe = make_pipeline(PretrainedFeatures(feature_extractor),
                         StandardScaler(),
                         get_regression_model())
    # print(pipe)
    # different scoring methods https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    # scores = cross_val_score(pipe, x_train, y_train, cv=5, scoring="neg_root_mean_squared_error")
    scores = cross_val_score(pipe, x_train, y_train, cv=4, scoring="neg_mean_squared_error")
    print(scores)
    print(np.mean(scores))

    # PretrainedFeatureClass = make_pretraining_class({"pretrain": feature_extractor})
    # regression model
    # regression_model = get_regression_model()
    pipe.fit(x_train, y_train)
    # print(pipe["gridsearchcv"].best_params_)

    y_pred = np.zeros(x_test.shape[0])

    y_pred = pipe.predict(x_test.to_numpy())

    # pretrained_feature_class = PretrainedFeatureClass(feature_extractor = "pretrain")
    # input('a')

    # X_train_new = pretrained_feature_class.transform(x_train)
    # x_tr_new, x_val_new, y_tr, y_val = train_test_split(X_train_new, y_train, test_size=0.2, random_state=0, shuffle=True)
    # X_test_new = pretrained_feature_class.transform(x_test.to_numpy())
    # input('b')

    #### SCALING??? #### 
    # scaler = StandardScaler()
    # X_train_new = scaler.fit_transform(X_train_new)
    ####################

    # regression_model.fit(X_train_new, y_train)
    # y_pred = regression_model.predict(X_test_new)

    ###### SPLIT ######
    # regression_model.fit(x_tr_new, y_tr)
    # y_pred_val = regression_model.predict(x_val_new)
    # error = mean_squared_error(y_val, y_pred_val)
    # print(error)
    # input('d')
    ###################

    # TODO: Implement the pipeline. It should contain feature extraction and regression. You can optionally
    # use other sklearn tools, such as StandardScaler, FunctionTransformer, etc.

    assert y_pred.shape == (x_test.shape[0],)
    y_pred = pd.DataFrame({"y": y_pred}, index=x_test.index)
    y_pred.to_csv("results.csv", index_label="Id")
    print("Predictions saved, all done!")