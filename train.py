import numpy as np
import pandas as pd
from pandas import Timestamp
import time
import math

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

def get_training(txt_path):
    """
    Get features and output values
    """ 
    training = np.loadtxt(txt_path)
    features, output = training[:,:-1], training[:,-1]
    return features, output

def get_binary_features(df_row, all_categories):
    """
    Takes in a row of a DataFrame and returns a binary feature vector
    where 1,0 represents the existence of a certain crime.
    @params df_row pd.Series row from pd.DataFrame
    @params all_categories set of all 311 categories
    @return 1xd np.array
    """ 
    reports = eval(df_row['311-reports'])

    # initialize feature values
    category_exists = dict()
    for c in all_categories:
        category_exists[c] = 0

    # set flat if report category exists
    for report in reports:
        category_exists[report['Category']] = 1  
    
    # return features sorted by category name to ensure consistent feature representation
    return np.array([category_exists[k] for k in sorted(category_exists.keys())])

def get_count_features(df_row, all_categories):
    """
    Takes in a row of a DataFrame and returns a feature vector consiting of
    counts of different categories of crime.
    @params df_row pd.Series row from pd.DataFrame
    """
    reports = eval(df_row['311-reports'])

    # initialize feature values
    category_exists = dict()
    for c in all_categories:
        category_exists[c] = 0

    # set flat if report category exists
    for report in reports:
        category_exists[report['Category']] += 1  
    
    # return features sorted by category name to ensure consistent feature representation
    return np.array([category_exists[k] for k in sorted(category_exists.keys())])

def get_binary_output(df_row, category_911):
    """
    Gets an output value which represents 1,0 binary value for whether
    the specified cateogry exists in 911 reports.
    @params df_row, pd.Series row from pd.DataFrame of 911 reports
    @params category_911 string 911 category to find
    """ 
    reports = eval(df_row['911-reports']) 

    # fast check for 'all' case
    if category_911 == "all":
        return 1 if len(reports) > 0 else 0
   
    # iterate through iterate all reports to find specified category
    for report in reports:
        if report['Category'].lower() == category_911.lower():
            return 1
    return 0

def get_count_output(df_row, category_911):
    """
    Gets an output value which represents 1,0 binary value for whether
    the specified cateogry exists in 911 reports.
    @params df_row, pd.Series row from pd.DataFrame of 911 reports
    @params category_911 string 911 category to find
    """ 
    reports = eval(df_row['911-reports']) 

    # fast check for 'all' case
    if category_911 == "all":
        return len(reports)

    # get number of reports of specified category
    return sum([1 for r in reports if r['Category'].lower() == category_911.lower()])

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
                   {'C': [1, 10, 100, 1000], 'gamma':[0.00001, 0.0001, 0.001, 0.01],  'kernel': ['rbf', 'poly']},
                    ]
        # kmeans
        elif isinstance(model, cluster.KMeans):
            params_grid = [
                    {'n_init': [10, 50]}
                    ] 
        
    optimized_model = grid_search.GridSearchCV(model, params_grid, verbose=verbose) 
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
