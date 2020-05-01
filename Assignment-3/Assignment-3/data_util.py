# This contains some util functions that we will use in many contexts.

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score

# Get a list of the categorical features for a given dataframe. 
def cat_features(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	return [x for x in list(dataframe) if not(dataframe[x].dtype in [td['a'].dtype, td['b'].dtype])]


# Get the indices of the categorical features for a given dataframe. 
def cat_feature_inds(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	enums = list(zip(list(dataframe), range(len(list(dataframe)))))
	selected = [(x[0], x[1]) for x in enums if not(dataframe[x[0]].dtype in [td['a'].dtype, td['b'].dtype])]
	return [s[1] for s in selected]

# Print out common error metrics for the binary classifications.
def print_binary_classif_error_report(y_test, preds):
	print('Accuracy: ' + str(accuracy_score(y_test, preds)))
	print('Precison: ' + str(precision_score(y_test, preds)))
	print('Recall: ' + str(recall_score(y_test, preds)))
	print('F1: ' + str(f1_score(y_test, preds)))
	print('ROC AUC: ' + str(roc_auc_score(y_test, preds)))
	print("Confusion Matrix:\n" + str(confusion_matrix(y_test, preds)))
	
# Print out common error metrics for the binary classifications.
def print_multiclass_classif_error_report(y_test, preds):
	print('Accuracy: ' + str(accuracy_score(y_test, preds)))
	print('Avg. F1 (Micro): ' + str(f1_score(y_test, preds, average='micro')))
	print('Avg. F1 (Macro): ' + str(f1_score(y_test, preds, average='macro')))
	print('Avg. F1 (Weighted): ' + str(f1_score(y_test, preds, average='weighted')))
	print(classification_report(y_test, preds))
	print("Confusion Matrix:\n" + str(confusion_matrix(y_test, preds)))

def print_regression_error_report(y_test, preds):
	print('MSE, MAE, R^2, EVS: ' + str([mean_squared_error(y_test, preds), \
							   median_absolute_error(y_test, preds), \
							   r2_score(y_test, preds), \
							   explained_variance_score(y_test, preds)])) 
	
	