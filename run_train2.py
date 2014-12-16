import numpy as np
from sklearn import svm
from sklearn import linear_model
from sklearn import cluster
from sklearn import neighbors
from sklearn import preprocessing
from sklearn import tree
from sklearn import ensemble

from sklearn import cross_validation
import time, pdb
import train

data_path = "./joined_matrix.txt"
features, output = train.get_training(data_path)
unit_features = preprocessing.scale(features, axis=1)

print("Testing SVR on first 10,000 samples")
model1 = svm.SVR()
start = time.time()
result = cross_validation.cross_val_score(model1, features[0:100],  output[0:100], verbose=True)
end = time.time()
print("Time to cross_validate on SVR model: %f"%(end - start))
print("Accuracies from cross validation %s"%result)
print("Average Accuracy %f"%np.average(result))

# model2 = cluster.KMeans()
# model2 = linear_model.Ridge()

#### SVR OPTIMIZATION ###
"""
print("\n")
print("Testing SVR on first 10,000 samples")
# best value was kernel ='rbf', C=10, gamma=0.01
start = time.time()
model2 = svm.SVR()
features = unit_features 
params_grid = [{'C': [1, 10, 100, 1000], 'gamma':[0.0001, 0.0001, 0.001, 0.01],'kernel':['rbf', 'poly']}]
opt_model = train.fine_tune(features[0:10000], output[0:10000], model2, params_grid = verbose=5)
score = opt_model.score(features[10000:20000], output[10000:20000]) 
end = time.time()
print("Time to cross_validate on SVR model: %f"%(end - start))
print("Accuracy of model on test set: %f, %s"%(score, opt_model))
"""

"""
### MORE FINE TUNE SVR OPTIMIZATION ###
start = time.time()
model2 = svm.SVR()
features = unit_features 
params_grid = [{'C': np.arange(9, 11, 0.25), 'gamma':[0.0001, 0.0001, 0.001, 0.01],'kernel':['rbf', 'poly']}]
opt_model = train.fine_tune(features[0:10000], output[0:10000], model2, params_grid=params_grid, verbose=5)
score = opt_model.score(features[10000:20000], output[10000:20000]) 
end = time.time()
print("Time to cross_validate on SVR model: %f"%(end - start))
print("Accuracy of model on test set: %f, %s"%(score, opt_model))
"""

"""
### KNN  REGRESSION  ###
def better_inv_dist(dist):
    c=1
    return 1. / (c + dist)

start = time.time()
model = neighbors.KNeighborsRegressor() 
features = unit_features 
params_grid = [{'n_neighbors': [1, 5, 10, 100, 500, 1000], 'weights': ['uniform', better_inv_dist]}]
opt_model = train.fine_tune(features[0:60000], output[0:60000], model, params_grid=params_grid, verbose=5)
score = opt_model.score(features[100000:110000], output[100000:110000]) 
end = time.time()
print("Time to cross_validate on SVR model: %f"%(end - start))
print("Accuracy of model on test set: %f, %s"%(score, opt_model))
"""
"""
### DECISION TREE REGRESSION ###
start = time.time()
model = tree.DecisionTreeRegressor()
features = unit_features 
params_grid = [{'splitter': ['best', 'random'], 'min_samples_leaf': np.arange(200, 500, 5)}]
opt_model = train.fine_tune(features[0:60000], output[0:60000], model, params_grid=params_grid, verbose=5)
score = opt_model.score(features[100000:110000], output[100000:110000]) 
end = time.time()
print("Time to fine-tune on SVR model: %f"%(end - start))
print("Optimal model parameters: %s"%opt_model.best_estimator_)
print("Best fine tune score: %f"%(opt_model.best_score_))
print("Accuracy of model on test set: %f"%(score))
"""

"""
### RANDOM FORESTS ###
start = time.time()
model = ensemble.RandomForestRegressor()
features = unit_features 
params_grid = [{'n_estimators': [10, 50, 100, 300], 'min_samples_leaf': [2, 5, 10, 50, 100, 500, 1000]}]
opt_model = train.fine_tune(features[0:60000], output[0:60000], model, params_grid=params_grid, verbose=5)
score = opt_model.score(features[100000:110000], output[100000:110000]) 
end = time.time()
print("Time to fine-tune on SVR model: %f"%(end - start))
print("Optimal model parameters: %s"%opt_model.best_estimator_)
print("Best fine tune score: %f"%(opt_model.best_score_))
print("Accuracy of model on test set: %f"%(score))

### ADABOOST FOREST REGRESSOR ###
start = time.time()
model = ensemble.AdaBoostRegressor()
features = unit_features 
params_grid = [{'n_estimators': [10, 50, 100, 300], 
		'learning_rate': [0.1, 0.3, 0.5, 0.7, 1],
		'loss': ['linear', 'square', 'exponential']}]
opt_model = train.fine_tune(features[0:60000], output[0:60000], model, params_grid=params_grid, verbose=5)
score = opt_model.score(features[100000:110000], output[100000:110000]) 
end = time.time()
print("Time to fine-tune on SVR model: %f"%(end - start))
print("Optimal model parameters: %s"%opt_model.best_estimator_)
print("Best fine tune score: %f"%(opt_model.best_score_))
print("Accuracy of model on test set: %f"%(score))



"""
