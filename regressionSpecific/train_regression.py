import numpy as np
import pandas as pd
from pandas import Timestamp
import time, copy
import math, json, pdb

from sklearn import cross_validation
from sklearn import datasets
from sklearn import grid_search

from sklearn import svm
from sklearn import linear_model
from sklearn import cluster
from sklearn import neighbors
from sklearn import preprocessing
from sklearn import tree
from sklearn import ensemble

# TODO: create training data
# TODO: cross validate models
# TODO: fine tune model
# TODO: implement function for 1,0 if crime will happen or not

def get_training(feature_path):
    """
    Get the features and a dictionary of different output vectors.
    @params feature_path string path to .txt file containing features
    @params feature_value_path string to .txt. file containing dictionary to output vectors
    @return dict
    """ 
    data = np.loadtxt(feature_path)
    features = data[:, 0:40]
    aggregate_counts = data[:, 77] 
    return features, aggregate_counts 


def cross_validate(features, outputs, model):
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
            outputs)

    # get average accuracy
    return np.average(scores)

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
    
        # SVR or SVM
        elif isinstance(model, svm.SVR) or isinstance(model, svm.SVC):
        	# n=10000, C=10, gamma=0.1, kernel=rbf: [0.454920, 0.384266, 0.453706]
            params_grid = [
                   {'C': [1, 10, 100, 1000], 'gamma':[0.00001, 0.0001, 0.001, 0.01],  'kernel': ['rbf', 'poly']},
                    ]
	# DECISION TREE
	elif isinstance(model, tree.DecisionTreeRegressor):
	    params_grid = [
		{'splitter': ['best', 'random'],
		 'min_samples_leaf': np.arange(50, 550, 50)}
			]
	# RANDOM FORESTS
	elif isinstance(model, ensemble.RandomForestRegressor):
   	    params_grid = [
		{'n_estimators':[50], # [10, 50, 100, 300],
		'min_samples_leaf':[10]}
			]
	# ADABOOST FOREST
	elif isinstance(model, ensemble.AdaBoostRegressor):
	    params_grid = [
		{'n_estimators': [10, 50, 100, 300],
		 'learning_rate': [ 0.1, 0.5, 0.7, 1],
		 'loss': ['linear', 'square', 'exponential']}
			]
	# KNN
	elif isinstance(model, neighbors.KNeighborsRegressor):
	    params_grid = [
                {'n_neighbors': [1, 5, 10, 100, 500, 1000], 'weights': ['uniform', better_inv_dist]}
			] 
        # kmeans
        elif isinstance(model, cluster.KMeans):
            params_grid = [
                    {'n_init': [10, 50]}
                    ] 
        
    optimized_model = grid_search.GridSearchCV(model, params_grid, verbose=verbose, n_jobs=2) 
    optimized_model.fit(features, outputs)
    return optimized_model


def train(features, outputs, test_features, test_outputs, model, params_grid=None, verbose=2):
        """
        Fine tune a model on a set of features and outputs over input params_grid, and output
        the optimal model. 

        If params_grid is None, then the params_grid will be default to preset ones in fine_tune

        @params features Nxd np.array features 
        @params outputs Nx1 np.array class value 
        @params test_features Mxd np.array test_features
        @params test_outputs Mx1 np.array test class values
        @params model sklearn classifier model to train 
        @params params_grid; see  http://scikit-learn.org/stable/modules/grid_search.html
        @params verbose int how verbose messages are; higher = more
        @return sklearn classifier model
        """
        start = time.time()
        opt_model = fine_tune(features, outputs, model, params_grid=params_grid)
        score = opt_model.score(test_features, test_outputs)
        end = time.time()
        print("Time to fine tune on %s model: %f"%(type(model), score)) 
        print("Optimal model parameters: %s"%(opt_model.best_estimator_))
        print("Best fine tune score: %f"%(opt_model.best_score_))
        print("Accuracy of model on test set: %f"%(score))
        return opt_model

def train_all_categories(features, outputs, model, params_grid=None):
    """
    Train a classifier for every possible category of output using each model in model.
    Output a dictionary whose keys are the categories and the output is a list of 
    two tuples. The first tuple is (default_classifier, score). The second tuple is
    (fine_tuned_classifier, score).

    @params feature NxD np.array feature matrix
    @params outputs dict: {category: Nx1 np.array output matrix}
    @params model model module to try 
    @return 
    """
    results = dict() 
    for output_class in outputs.keys():
        output_class="ASSAULT"
        output = outputs[output_class]

    	default_model = model()
        default_score = cross_validate(features, output, default_model)

	print("Fine Tuning for class %s"%(output_class))
	
	fine_tune_model = fine_tune(features, output, model(), verbose=1, params_grid=params_grid,)
        best_model, fine_tune_score = fine_tune_model.best_estimator_, fine_tune_model.best_score_
        results[output_class] = [(default_model, default_score), (best_model, fine_tune_score)]
	break
    return results

def better_inv_dist(dist):
    c=1
    return 1. /(c+dist)

features, class_outputs = get_training('../joined_matrix_split.txt') 
# 0:27 category  28:38= supervisor district, 39 = count 40= output
features_10k, class_10k = features, class_outputs
test_features_10k, test_class_10k = features[10000:20000], class_outputs[10000:20000]

#all_models = [tree.DecisionTreeRegressor(), ensemble.RandomForestRegressor(), ensemble.AdaBoostRegressor(), neighbors.KNeighborsRegressor()] 
#all_models = [tree.DecisionTreeClassifier(), ensemble.RandomForestClassifier(), ensemble.AdaBoostClassifier(), neighbors.KNeighborsClassifier()] 
all_models = [ensemble.RandomForestRegressor()]
opt_models = dict()
for i,model in enumerate(all_models):
    param_grid = None
    opt_model = train(features_10k, class_10k, test_features_10k, test_class_10k, model, params_grid = param_grid)
    r2 = opt_model.best_score_
    opt_models[opt_model.best_estimator_] = r2

print("\n")
print("\n")

for k,v in opt_models.iteritems(): 
    print("model: %s"%k)
    print("r2: %f"%(v))
