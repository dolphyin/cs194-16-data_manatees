import numpy as np
import pandas as pd
from pandas import Timestamp
import pdb
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

def get_training(csv_path, binary_feature=True, binary_output=True, category_911="all"):
    """
    Construct training set from input csv file and outputs a Nxd feature vector and
    Nx1 output value. The features and outputs can either represent binary 1,0 existence
    of categories of crimes, or a count of categories of crimes. These can be set
    using the binary_feature, binary_out parameters.

    The category of 911 crimes predicted can be set using category_911. It can be set
    to a specific category (like 'ASSAULT') or to "all", in which case it will return
    1,0 if any 911 report existed (binary output) or a count of all crimes (count output).

    @params csv_path string path to csv file 
    @params binary_feature boolean to have binary or count features
    @params binary_output boolean to have binary of count output
    @params 911_category string category of 911 crime to predict existence of 
    @return Nxd np.array feature vectors, Nx1 np.array class output
    """
    load_start = time.time()
    df = pd.io.parsers.read_csv(csv_path)
    load_end = time.time()
    print("Time to load %s: %f"%(csv_path, load_end-load_start))
    num_rows = df.shape[0]
   
    stat_start = time.time()
    # get statistics to initialize numpy array
    location_bins = set()
    category_bins = set()
    for i in xrange(num_rows): 
        try:
            rows = eval(df.iloc[i]['311-reports'])
        except NameError:
            print("Error parsing row %d. Moving on..."%i)
        for r in rows:
            category_bins.add(r['Category']) 
    stat_end = time.time()
    print("Time to extract initial statistics: %f"%(stat_end-stat_start))

    # initialize numpy vector
    feat_start = time.time()
    feature_vectors = np.zeros([num_rows, len(category_bins)])
    output_vector = np.zeros([num_rows])
    invalid_rows = []
    for i in xrange(num_rows):
        print(i)
        # get features
        if binary_feature:
            features = get_binary_features(df.iloc[i], category_bins)
        else:
            features = get_count_features(df.iloc[i], category_bins)


        # check if 911-report field is nan
        row_911 = df.iloc[i]['911-reports']
        if math.isnan(row_911):
            print("Found nan 911 report value at index %d. Removing from training set"%i)
            invalid_rows.append(i)
            continue

        # get output class/value
        if binary_output:
            output = get_binary_output(df.iloc[i], category_911)
        else:
            output = get_count_output(df.iloc[i], category_911)
    
        # insert values into feature and output vectors
        feature_vectors[i] = features
        output_vector[i] = output

    feat_end = time.time()
    print("Time to extract features: %f"%(feat_end - feat_start))
    return np.delete(feature_vectors,invalid_rows) , np.delete(output_vector, invalid_rows)

def get_binary_features(df_row, all_categories):
    """
    Takes in a row of a DataFrame and returns a binary feature vector
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

train_features, train_class = get_training('./joined_data.csv',
        binary_feature=True,
        binary_output=True,
        category_911="all")
