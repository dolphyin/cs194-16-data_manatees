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

crime_inverse = {'KIDNAPPING': 0, 'WEAPON LAWS': 1, 'WARRANTS': 2, 'FRAUD': 20, 'DRIVING UNDER THE INFLUENCE': 8, 'ROBBERY': 9, 'BURGLARY': 10, 'SUSPICIOUS OCC': 11, 'ASSAULT': 33, 'FORGERY/COUNTERFEITING': 14, 'DRUNKENNESS': 16, 'OTHER OFFENSES': 19, 'RECOVERED VEHICLE': 21, 'SEX OFFENSES, FORCIBLE': 22, 'DRUG/NARCOTIC': 23, 'TRESPASS': 24, 'VANDALISM': 26, 'MISSING PERSON': 34, 'VEHICLE THEFT': 31, 'NON-CRIMINAL': 27, 'LARCENY/THEFT': 25}
crime = {}
for k in crime_inverse:
	v = crime_inverse[k]
	crime[v] = k


#import data
data_path = "joined_matrix_split.txt"
mat = np.loadtxt(data_path)
features = mat[50000:60000, 0:40]
features = sklearn.preprocessing.scale(features, axis=1)
output_raw = mat[50000:60000, 40:]
output = sklearn.preprocessing.binarize(output_raw)

# Split into training and test
random_state = np.random.RandomState(0)

n_classes = 37
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
	if i not in crime.keys():
		continue
	X_train, X_test, y_train, y_test = train_test_split(features, output[:,i], test_size=.5, random_state=random_state)
	#run classifier
	model = ensemble.AdaBoostClassifier()
	classifier = OneVsRestClassifier(model)
	y_score = classifier.fit(X_train, y_train).decision_function(X_test)
	precision[i], recall[i], _ = precision_recall_curve(y_test, y_score)
	average_precision[i] = average_precision_score(y_test, y_score)

# Plot Precision-Recall curve for each class
plt.clf()
for i in range(n_classes):
    if i not in crime.keys():
	continue
    plt.plot(recall[i], precision[i],
             label='{0} (area = {1:0.2f})'
                   ''.format(crime[i], average_precision[i]))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('AdaBoost Decision Tree on specific Crime Categories')

#plt.legend(loc="lower right")

from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size('small')

# Shrink current axis by 20%
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop=fontP)

plt.show()




