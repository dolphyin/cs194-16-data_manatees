import numpy as np
import pandas as pd

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

def get_binary_training(pd, category_911="all"):
    """
    Construct training set from input pandas dataframe to predict the existence
    of specified category of 911 crime using binary features and outputs. 
    Specifically, features include 1,0 binary flag for the location and 1,0  binary
    flag for for whether a 311 category existed. The output value represents whether
    or not a 911 report of the specified category existed.

    If category_911="all", then the output class represents whether or not any crime 
    happened. 

    @params pd pandas.Dataframe input data
    @params 911_catogory string category of 911 crime to predict existence of 
    @return Nxd np.array feature vectors, Nx1 np.array class output
    """
    if category_911=="all":
        # get ouput results for all 911 types
        pass
    else:
        pass
    pass

def get_count_training(pd, category_911="all"):
    """
    Construct training set from pandas dataframe. Same as get_binary_training,
    except generates features which are continuous valued. Specifically, rather
    than a 1 or 0 if a 311 report category exists, the feature represents the
    number of reports of a certain category. Additionally, the predicted output
    value represents the number of 911 crimes of the specified category.

    If category_911="all", then the output class represents the total number of
    all types of crimes.

    @params pd pandas.Dataframe
    @params 911_category string category of 911 crime to predict
    @return Nxd np.array feature vectors, Nx1 np.array regression output
    """
    if category_911 == "all":
        pass
    else:
        pass 
    pass


def fine_tune(features, outputs, model):
    """
    Fine tune a classifier by doing grid search over possible parameters.
    Return an fine-tuned model,

    @params features np.array Nxd array feature vectorx
    @params outputs np.array Nx1 array output vector
    @params model sklearn classifier/regression model 
    @return sklearn fine-tuned model
    """
    ### REGRESSION METHODS ###
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
        params_grid = [
                {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                {'C': [1, 10, 100, 1000], 'gamma': [10**-2, 10**-3], 'kernel': ['rbf']},
                {'C': [1, 10, 100, 1000], 'degree': [1, 2, 3, 4, 5], 'gamma': [10**-2, 10**-3] ,'kernel': ['poly']}
                ]
    # kmeans
    elif isinstance(model, cluster.KMeans):
        params_grid = [
                {'n_init': [10, 50, 100, 500]}
                ] 
        
    optimized_model = grid_search.GridSearchCV(model, params_grid) 
    optimized_model.fit(features, outputs)
    return optimized_model

def cross_validate(features, outputs, model, k=10):
    """
    Implements K-fold cross validation on the input training using the input model
    and returns the average accuracy rate.

    @params features np.array Nxd feature matrix
    @params outputs np.array Nx1 output vector 
    @params model classifer
    @params k int number of folds
    @return float average accuracy
    """ 
    # get features & classes
    features = training[:,:-1]
    outputs = training[:,-1]

    # do k-folds cross validation
    scores = cross_validation.cross_val_score(model,
            features,
            outputs,
            k)

    # get average accuracy
    return np.average(scores)

