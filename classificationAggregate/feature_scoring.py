from sklearn import feature_selection
import numpy as np

def get_chi_squared(features, outputs):
    return feature_selection.univariate_selection.chi2(features, outputs)

data = np.loadtxt('../joined_matrix_2.txt') #feature vector: normalized vector
features, output = data[:,:-1], data[:,-1]
chi = get_chi_squared(features, output)
