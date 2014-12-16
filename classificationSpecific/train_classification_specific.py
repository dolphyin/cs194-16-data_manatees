import numpy as np
import pandas as pd
from pandas import Timestamp
import time, copy
import math, json, pdb

from sklearn import cross_validation
from sklearn import datasets
from sklearn import grid_search
from sklearn import metrics

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

def get_training(feature_path, category_911):
    """
    Get the features and a dictionary of different output vectors.
    @params feature_path string path to .txt file containing features
    """ 
    categories = ['KIDNAPPING', 'WEAPON LAWS', 'WARRANTS', 'PROSTITUTION', 'EMBEZZLEMENT', 'SEX OFFENSES, NON FORCIBLE', 'LOITERING', 'SUICIDE', 'DRIVING UNDER THE INFLUENCE', 'ROBBERY', 'BURGLARY', 'SUSPICIOUS OCC', 'FAMILY OFFENSES', 'BRIBERY', 'FORGERY/COUNTERFEITING', 'BAD CHECKS', 'DRUNKENNESS', 'ARSON', 'GAMBLING', 'OTHER OFFENSES', 'FRAUD', 'RECOVERED VEHICLE', 'SEX OFFENSES, FORCIBLE', 'DRUG/NARCOTIC', 'TRESPASS', 'LARCENY/THEFT', 'VANDALISM', 'NON-CRIMINAL', 'EXTORTION', 'PORNOGRAPHY/OBSCENE MAT', 'LIQUOR LAWS', 'VEHICLE THEFT', 'STOLEN PROPERTY', 'ASSAULT', 'MISSING PERSON', 'DISORDERLY CONDUCT', 'RUNAWAY']

    features = np.loadtxt(feature_path)
    features_in = features[:,0:40]
    category_index = 40 + categories.index(category_911) 
    features_out = features[:, category_index]
    return features_in, features_out

def get_equal_training_test(features, outputs):
    """
    constructs an training set which has equal number true and false outputs
    """ 

    num_positive_class = len(np.where(outputs > 0)[0])
    num_train_samples = int(num_positive_class/2)

    true_indices = np.where(outputs > 0)[0]
    false_indices = np.where(outputs == 0)[0]

    x_train_true = features[true_indices[:num_train_samples]]
    x_train_false = features[false_indices[:num_train_samples]]
    x_train = np.vstack([x_train_true, x_train_false])

    y_train_true = outputs[true_indices[:num_train_samples]]
    y_train_false = outputs[false_indices[:num_train_samples]]
    y_train = np.hstack([y_train_true, y_train_false])

    x_test_true = features[true_indices[num_train_samples:2*num_train_samples]]
    x_test_false = features[false_indices[num_train_samples:2*num_train_samples]]
    x_test = np.vstack([x_test_true, x_test_false])

    y_test_true = outputs[true_indices[num_train_samples:2*num_train_samples]]
    y_test_false = outputs[false_indices[num_train_samples:2*num_train_samples]]
    y_test = np.hstack([y_test_true, y_test_false])
 
    return x_train, y_train, x_test, y_test
     

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
        # SVR or SVM
	if isinstance(model, svm.LinearSVC):
            params_grid = [
		   {'C': [1, 10, 100, 1000], 'loss': ['l1'], 'penalty': ['l2'], 'dual': [True]},
		   {'C': [1, 10, 100, 1000], 'loss': ['l2'], 'penalty': ['l1'], 'dual': [False]}]
        elif isinstance(model, svm.SVC) or isinstance(model, svm.SVR):
        	# n=10000, C=10, gamma=0.1, kernel=rbf: [0.454920, 0.384266, 0.453706]
            params_grid = [
                   {'C': [1, 10, 100, 1000], 'gamma':[0.001, 0.01],  'kernel': ['rbf', 'poly']},
                    ]
	# DECISION TREE
	elif isinstance(model, tree.DecisionTreeClassifier):
	    params_grid = [
		{'splitter': ['best', 'random'],
		 'min_samples_leaf': np.arange(50, 550, 50)}
			]
	# RANDOM FORESTS
	elif isinstance(model, ensemble.RandomForestClassifier):
   	    params_grid = [
		{'n_estimators':[10, 50, 100, 300],
		'min_samples_leaf':[5, 50, 100, 500, 1000]}
			]
	# ADABOOST FOREST
	elif isinstance(model, ensemble.AdaBoostClassifier):
	    params_grid = [
		{'n_estimators': [10, 50, 100, 300],
		 'learning_rate': [ 0.1, 0.5, 0.7, 1]}
			]
	# KNN
	elif isinstance(model, neighbors.KNeighborsClassifier):
	    params_grid = [
                {'n_neighbors': [1, 5, 10, 100, 500, 1000], 'weights': ['uniform', better_inv_dist]}
			] 
        # kmeans
        elif isinstance(model, cluster.KMeans):
            params_grid = [
                    {'n_init': [10, 50]}
                    ] 
        
    optimized_model = grid_search.GridSearchCV(model, params_grid, verbose=verbose, n_jobs=4) 
    optimized_model.fit(features, outputs)
    return optimized_model

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
	return opt_model.best_estimator_


def get_classification_metrics(features, true_output, model):
    """
    Given a model, get the accuracy, precision, and recall
    """
    accuracy = model.score(features, true_output)
    guess = model.predict(features)
    precision = metrics.precision_score(true_output, guess)
    recall = metrics.recall_score(true_output, guess)
    return accuracy, precision, recall

def better_inv_dist(dist):
    c=1
    return 1. / (c+dist)



features, class_outputs = get_training('../joined_matrix_split.txt', 'LARCENY/THEFT') 
# 0:27 category  28:38= supervisor district, 39 = count 311, 
# 40:76 counts of 911 categories, 77=count of 911 aggregate
#class_outputs = preprocessing.binarize(class_outputs)
#features_10k, class_10k = features[:10000, 0:40], class_outputs[:10000]
#test_features_10k, test_class_10k = features[10000:20000, 0:40], class_outputs[10000:20000]

features_10k, class_10k, test_features_10k, test_class_10k = get_equal_training_test(features, class_outputs)

#call_models = [svm.LinearSVC(), svm.SVC(), tree.DecisionTreeClassifier(), ensemble.RandomForestClassifier(), ensemble.AdaBoostClassifier(), neighbors.KNeighborsClassifier()] 
#all_models = [tree.DecisionTreeClassifier(), ensemble.RandomForestClassifier(), ensemble.AdaBoostClassifier(), neighbors.KNeighborsClassifier()] 
all_models = [svm.SVR()]
opt_models = dict()
for i,model in enumerate(all_models):
    param_grid = None
    opt_model = train(features_10k, class_10k, test_features_10k, test_class_10k, model, params_grid = param_grid)
    (accuracy, precision, recall) = get_classification_metrics(test_features_10k, test_class_10k, opt_model)
    opt_models[opt_model] = (accuracy, precision, recall) 

print("\n")
print("\n")

for k,v in opt_models.iteritems(): 
    print("model: %s"%k)
    print("accuracy: %f, precision: %f, recall: %f"%(v[0], v[1], v[2]))
