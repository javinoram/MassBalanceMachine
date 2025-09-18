import sys, os
sys.path.append(os.path.join(os.getcwd(), '../')) # Add root of repo to import MBM

import pandas as pd
import massbalancemachine as mbm
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
from skorch.helper import SliceDataset
import torchsummary
from model import SimpleNetwork

warnings.filterwarnings('ignore')


data = pd.read_csv('./data/SouthernAndesPatagonia_monthly_dataset.csv')
cfg = mbm.Config()

# Create a new DataLoader object with the monthly stake data measurements.
dataloader = mbm.dataloader.DataLoader(cfg, data=data)
# Create a training and testing iterators. The parameters are optional. The default value of test_size is 0.3.
train_itr, test_itr = dataloader.set_train_test_split(test_size=0.3)

# Get all indices of the training and testing dataset at once from the iterators. Once called, the iterators are empty.
train_indices, test_indices = list(train_itr), list(test_itr)

# Get the features and targets of the training data for the indices as defined above, that will be used during the cross validation.
df_X_train = data.iloc[train_indices]
y_train = df_X_train['POINT_BALANCE'].values

# Get test set
df_X_test = data.iloc[test_indices]
y_test = df_X_test['POINT_BALANCE'].values

# Create the cross validation splits based on the training dataset. The default value for the number of splits is 5.
type_fold = 'group-meas-id'  # 'group-rgi' # or 'group-meas-id'
splits = dataloader.get_cv_split(n_splits=5, type_fold=type_fold)


feature_columns = df_X_train.columns.difference(cfg.metaData)
feature_columns = feature_columns.drop(cfg.notMetaDataNotFeatures)
feature_columns = list(feature_columns)
nInp = len(feature_columns)
cfg.setFeatures(feature_columns)
nNeurons = [nInp, 8]

# Create a CustomNeuralNetRegressor instance
params_init = {"device": "cpu"}
custom_nn = mbm.models.CustomNeuralNetRegressor(
    cfg,
    SimpleNetwork,
    nbFeatures=nInp,
    module__nNeurons=nNeurons,
    train_split=False, # train_split is disabled since cross validation is handled by the splits variable hereafter
    batch_size=16,
    verbose=0,
    iterator_train__shuffle=True,
    **params_init)


features, metadata = mbm.data_processing.utils.create_features_metadata(cfg, df_X_train)

# Define the dataset for the NN
dataset = mbm.data_processing.AggregatedDataset(
    cfg,
    features=features,
    metadata=metadata,
    targets=y_train
)
splits = dataset.mapSplitsToDataset(splits)

# Use SliceDataset to make the dataset accessible as a numpy array for scikit learn
dataset = [SliceDataset(dataset, idx=0), SliceDataset(dataset, idx=1)]


# For each of the XGBoost parameter, define the grid range
parameters = {
    'lr': [0.005, 0.001, 0.0005],
    'max_epochs': [500, 1000, 2000],
    'optimizer': [torch.optim.Adam],
}

# GridSearch
custom_nn.gridsearch(parameters=parameters, splits=splits, features=dataset[0], targets=dataset[1])

best_params = custom_nn.param_search.best_params_
best_estimator = custom_nn.param_search.best_estimator_
print("Best parameters:\n", best_params)
print("Best score:\n", custom_nn.param_search.best_score_)

custom_nn.save_model("nn_Pa.pkl") 
