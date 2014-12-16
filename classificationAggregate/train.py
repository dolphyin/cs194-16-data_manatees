import numpy as np
import pandas as pd
from pandas import Timestamp
import time
import math, json, pdb

from sklearn import cross_validation
from sklearn import datasets
from sklearn import grid_search

from sklearn import svm
from sklearn import linear_model
from sklearn import cluster

# TODO: create training data
# TODO: cross validate models
# TODO: fine tune model
# TODO: implement function for 1,0 if crime will happen or not

def get_training(feature_path):
    """
    Get the features and a dictionary of different output vectors.
    Splits the data to feature and output vector
    Normalizes the output to be 1 or 0
    @params feature_path string path to .txt file containing features
    @return dict
    """
    pdb.set_trace() 
    features_ = np.loadtxt(feature_path)
    features = features_
    feature_size = features.shape[1] - 1
    features_in = features[:,0:feature_size]
    features_out_unnorm = features[:,-1]
    features_out = np.array(map(lambda x: x if x else 0, features_out_unnorm))
    return features_in, features_out 

def fine_tune(features, outputs, model, verbose=3, params_grid=None):
    """
    Fine tune a classifier by doing grid search over possible parameters.
    Return an fine-tuned model,

    @params features np.array Nxd array feature vectorx
    @params outputs np.array Nx1 array output vector
    @params model sklearn classifier/regression model 
    @return sklearn fine-tuned model
    """
    ### REGRESSION METHODS ###
    if params_grid == None:
        # linear regression 
        if isinstance(model, linear_model.LinearRegression):
            params_grid = [
                    {'fit_intercept': [True, False], 'normalize': [True, False]}
                    ]
        # ridge linear regression 
        elif isinstance(model, linear_model.Ridge):
            params_grid = [
                    {'alpha': [10**-8, 10**-5, 10**-3, 10**-1, 0.5, 1] }
                    ]
    
        ### CLASSIFICATION METHODS ###
        # logistic regression
        elif isinstance(model, linear_model.LogisticRegression):
            params_grid = [
                    {'C': [1, 10, 100, 1000]}
                    ]

        ### GENERAL METHODS ###
        # SVR or SVM
        elif isinstance(model, svm.SVR) or isinstance(model, svm.SVC):
        	# n=10000, C=10, gamma=0.1, kernel=rbf: [0.454920, 0.384266, 0.453706]
            params_grid = [
                   #{'C': [1, 10, 100, 1000], 'gamma':[0.00001, 0.0001, 0.001, 0.01],  'kernel': ['rbf', 'poly']},
                   {'C': [1, 10, 100], 'gamma':[0.0001, 0.001, 0.01],  'kernel': ['rbf']},
                    ]
        # kmeans
        elif isinstance(model, cluster.KMeans):
            params_grid = [
                    {'n_init': [10, 50]}
                    ] 
        
    optimized_model = grid_search.GridSearchCV(model, params_grid, scoring='r2', verbose=verbose) 
    optimized_model.fit(features, outputs)
    return optimized_model


def cross_validate(features, outputs, model, k=10):
    """
    scores = cross_validation.cross_val_score(model,
    Implements K-fold cross validation on the input training using the input model
    and returns the average accuracy rate.

    @params features np.array Nxd feature matrix
    @params outputs np.array Nx1 output vector 
    @params model classifer
    @params k int number of folds
    @return float average accuracy
    """ 
    # do k-folds cross validation
    scores = cross_validation.cross_val_score(model,
            features,
            outputs,
            k)

    # get average accuracy
    return np.average(scores)



