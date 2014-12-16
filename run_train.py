import numpy as np
from sklearn import svm
from sklearn import linear_model
from sklearn import cluster

from sklearn import cross_validation
import time
import train
import pdb

data_path = "joined_matrix_split.txt"
#data_path = "classificationAggregate/all_joined_matrix.txt"
mat = np.loadtxt(data_path)
pdb.set_trace()
features = mat[50000:60000, 0:40]
output = mat[50000:60000, -1]
print("Testing SVR on first 10,000 samples")
model1 = svm.SVR()
start = time.time()
result = cross_validation.cross_val_score(model1, features,  output, verbose=True)
end = time.time()
print("Time to cross_validate on SVR model: %f"%(end - start))
print("Accuracies from cross validation %s"%result)
print("Average Accuracy %f"%np.average(result))


"""
print("\n")
print("Testing SVR on first 60,000 samples")
model2 = svm.SVR()
start = time.time()
result = cross_validation.cross_val_score(model1, features[0:60000],  output[0:60000], verbose=True)
end = time.time()
print("Time to cross_validate on SVR model: %f"%(end - start))
print("Accuracies from cross validation %s"%result)
print("Average Accuracy %f"%np.average(result))

print("Testing SVR on all samples")
model2 = svm.SVR()
start = time.time()
result = cross_validation.cross_val_score(model1, features,  output, verbose=True)
end = time.time()
print("Time to cross_validate on SVR model: %f"%(end - start))
print("Accuracies from cross validation %s"%result)
print("Average Accuracy %f"%np.average(result))


"""
