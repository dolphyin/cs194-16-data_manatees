import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import svm, datasets
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import pdb
from sklearn import ensemble
from sklearn import neighbors
from sklearn import tree
#import data
data_path = "joined_matrix_split.txt"
mat = np.loadtxt(data_path)
features = mat[50000:60000, 0:40]
features = sklearn.preprocessing.scale(features, axis=1)
output_raw = mat[50000:60000, -1]
output = sklearn.preprocessing.binarize(output_raw)

# Split into training and test
random_state = np.random.RandomState(0)
X_train, X_test, y_train, y_test = train_test_split(features, output, test_size=.5,
                                                    random_state=random_state)

n_classes = 1
#run classifier
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# Compute Precision-Recall and plot curve
precision = dict()
recall = dict()
average_precision = dict()
precision[0], recall[0], _ = precision_recall_curve(y_test, y_score)
average_precision[0] = average_precision_score(y_test, y_score)




# now do rbf kernel
classifier = OneVsRestClassifier(svm.SVC(kernel='rbf', probability=True, random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)
# Compute Precision-Recall and plot curve
precision[1], recall[1], _ = precision_recall_curve(y_test, y_score)
average_precision[1] = average_precision_score(y_test, y_score)



# now do adaboost
model = ensemble.AdaBoostClassifier()
classifier = OneVsRestClassifier(model)
y_score = classifier.fit(X_train, y_train).decision_function(X_test)
# Compute Precision-Recall and plot curve
precision[2], recall[2], _ = precision_recall_curve(y_test, y_score)
average_precision[2] = average_precision_score(y_test, y_score)

"""
pdb.set_trace()
# now do kNN classifier
model = neighbors.KNeighborsClassifier()
classifier = OneVsRestClassifier(model)
y_score = classifier.fit(X_train, y_train).decision_function(X_test)
# Compute Precision-Recall and plot curve
precision[3], recall[3], _ = precision_recall_curve(y_test, y_score)
average_precision[3] = average_precision_score(y_test, y_score)



# now do random forrest
model = ensemble.RandomForestClassifier()
classifier = OneVsRestClassifier(model)
y_score = classifier.fit(X_train, y_train).decision_function(X_test)
# Compute Precision-Recall and plot curve
precision[4], recall[4], _ = precision_recall_curve(y_test, y_score)
average_precision[4] = average_precision_score(y_test, y_score)



# now do decision trees
model = tree.DecisionTreeClassifier()
classifier = OneVsRestClassifier(model)
y_score = classifier.fit(X_train, y_train).decision_function(X_test)
# Compute Precision-Recall and plot curve
precision[5], recall[5], _ = precision_recall_curve(y_test, y_score)
average_precision[5] = average_precision_score(y_test, y_score)

# Plot Precision-Recall curve
#plt.clf()
#plt.plot(recall[0], precision[0], label='Precision-Recall curve')
#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.ylim([0.0, 1.05])
#plt.xlim([0.0, 1.0])
#plt.title('Linear SVC Precision vs. Recall: AUC={0:0.2f}'.format(average_precision[0]))
#plt.legend(loc="lower left")
#plt.show()
"""
kernel = {}
kernel[0] = "linear SVC"
kernel[1] = "rbf SVC"
kernel[2] = "AdaBoost classifier"
#kernel[3] = "k-nearest-neighbors classifier"
#kernel[4] = "random forest classifier"
#kernel[5] = "decision tree classifier"
# Plot Precision-Recall curve for each class
plt.clf()
for i in range(3):
    plt.plot(recall[i], precision[i],
             label='Precision-recall curve of {0} (area = {1:0.2f})'
                   ''.format(kernel[i], average_precision[i]))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Classification on aggregate crime; precision vs. recall')
plt.legend(loc="lower right")
plt.show()




