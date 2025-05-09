# MIT LICENSE
# Written by Dr. Inal Mashukov
# Affiliation: 
# University of Massachusetts Boston,
# Department of Computer Science,
# Artificial Intelligence Laboratory 

import numpy as np
import pandas as pd
import math
import torch
import warnings 
import json
from sklearn.preprocessing import StandardScaler
# from models.model import Transformer

warnings.filterwarnings('ignore')


def get_parameters():
    path = './config/config.json'

    with open(path, 'r') as file:
        config = json.load(file)

    return config

# USE EITHER PREPROCESSING PIPELINE.
# THE ONE AFTER THIS IS MEANT FOR EXPERIMENTAL INTEGRITY AND REPRODUCABILITY USED WITH A DIFFERENT MODEL AND CAN BE DISPENSED OF.
# IN THAT CASE, UNCOMMMENT THE FOLLOWING METHOD.
# def data_loader(data_path: str = './data/'):

#     X_train, y_train = np.load(data_path + 'X_train.npy'), np.load(data_path + 'y_train.npy')
#     X_test, y_test = np.load(data_path + 'X_test.npy'), np.load(data_path + 'y_test.npy')
#     # TODO: standard normalization
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
#     X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

#     return X_train, y_train, X_test, y_test

def data_loader(data_path: str = './data/'):

    '''
    Inputs:
        - X_train: .npy file, shape = (num_seq, seq_len, n_features)
        - X_test: .npy file, shape = (num_seq, seq_len, n_features)
        - y_train: .npy file, shape = (num_seq, )
        - y_test: .npy file, shape = (num_seq, )
    '''
    X_train =  np.load(data_path + 'X_train.npy').astype(np.float32)
    X_test = np.load(data_path + 'X_test.npy').astype(np.float32)

    y_train = np.load(data_path + 'y_train.npy').astype(np.int32)
    y_test = np.load(data_path + 'y_test.npy').astype(np.int32)

    return X_train, y_train, X_test, y_test


def mean_standardize_fit(X):

    ''' 
    Args:

        - X: np.array, shape = (num_seq, seq_len, n_features)
    '''
    # get the mean across individual sequences:
    m1 = np.mean(X, axis=1)
    # get the mean across all sequences:
    mean = np.mean(m1, axis = 0)

    # get the std across individual sequences:
    s1 = np.std(X, axis = 1)
    # get the std across all sequences
    std = np.mean(s1, axis = 0)

    return mean, std

def mean_standardize_transform(X, mean, std):
    return (X-mean)/ std

def preprocess(X_train, y_train, 
               X_test, y_test,
               batch_size: int = 32):
    
    mean, std = mean_standardize_fit(X_train)
    X_train, X_test = mean_standardize_transform(X_train, mean, std), \
          mean_standardize_transform(X_test, mean, std)
    
    # dropping last incomplete batch, if needed:
    num_train_inst, num_test_inst = X_train.shape[0], X_test.shape[0]

    num_training_samples = math.ceil(num_train_inst / batch_size) * batch_size
    num_test_samples = math.ceil(num_test_inst / batch_size) * batch_size

    X_train = torch.as_tensor(X_train).float()
    X_test = torch.as_tensor(X_test).float()

    y_train = torch.as_tensor(y_train)
    y_test = torch.as_tensor(y_test)

    return X_train, y_train, X_test, y_test


def evaluate(model, dataloader):

    model.eval() # set the model into eval mode
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            outputs, _ = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

    return correct/total


