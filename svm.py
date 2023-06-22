# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 05:41:27 2023

@author: PC
"""

import lightgbm as ltb
import imblearn
import pandas as pd
import os
import numpy as np
import sklearn.metrics as sm
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import re
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.svm import SVC

data = pd.read_csv("F:/utkan/tez/modeldata/datagenus.csv", sep = ',')
data = data.drop(columns = ['Unnamed: 0'])
asd = data[["Autism"]]
tax = data.drop(columns = ['Autism'])
tax = tax.transpose()
tax.columns = tax.iloc[0]
tax = tax.drop(tax[0:1].index)
tax = tax.transpose()
tax = tax.drop(columns = ['Unassigned;__;__;__;__;__'])


# Assuming you have a DataFrame called 'df'

# Specify the string you want to search for
string_to_search = ";_"
string_to_search2 = "uncultured"

# Use filter() to select columns that include the specific string
filtered_df = tax.filter(regex=f"^(?!.*{string_to_search}).*$")
filtered_df2 = filtered_df.filter(regex=f"^(?!.*{string_to_search2}).*$")
tax = filtered_df2

# filtered_df now contains the DataFrame with columns that do not include the specific string



#%%
X = tax
Y = asd
X.columns = X.columns.str.translate("".maketrans({"[":"{", "]":"}","<":"^", "_":"-", ";":" "}))
X = X.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', ' ', x))
X = X.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 25)

eyo = X_train
eyo = eyo.astype(float)
eyo = pd.DataFrame(eyo)
meyo = y_train
meyo = meyo.astype(int)
meyo = pd.DataFrame(meyo)
meyo = np.ravel(meyo)
#%%
svm = SVC()

svm.fit(eyo, meyo)
y_pred = svm.predict(X_test)
y_pred_proba = svm.predict_proba(X_test)
trainacc = svm.predict(eyo)
#%%
y_test = y_test.astype(int)
y_test = pd.DataFrame(y_test)
y_pred = pd.DataFrame(y_pred)

eyo=eyo.astype(float)
eyo = pd.DataFrame(eyo)
trainacc = pd.DataFrame(trainacc)

print("\nAccuracy score of original normal-high data predicted with XGBC:")
print(accuracy_score(y_test, y_pred))
print(accuracy_score(meyo, trainacc))
print (classification_report(y_test, y_pred) + "\n")


cm1 = confusion_matrix(y_test, y_pred)

FP = cm1.sum(axis=0) - np.diag(cm1) 
FN = cm1.sum(axis=1) - np.diag(cm1)
TP = np.diag(cm1)
TN = cm1.sum() - (FP + FN + TP)
FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

TPR = TP/(TP+FN)
sen1 = round(float(TPR[0:1]), 3)
print("Sensitivity, hit rate, recall, or true positive rate = " + str(TPR))

TNR = TN/(TN+FP)
spec1 = round(float(TNR[0:1]), 3)
print("Specificity or true negative rate +" + str(TNR))

PPV = TP/(TP+FP)
prec1 = round(float(PPV[0:1]), 3)
print("Precision or positive predictive value +" + str(PPV))

cm1_df = pd.DataFrame(cm1,
                     index = ["Healthy", "Autism"], 
                     columns = ["Healthy", "Autism"])

utkan1 = plt.figure(figsize=(5,4))
sns.heatmap(cm1_df, annot=True, cmap = 'Reds')
plt.rc('ytick', labelsize=10)
plt.rc('xtick', labelsize=10)
plt.title('Confusion Matrix - LGBM')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.text(-1, 0, "Accuracy = %s" % (round(accuracy_score(y_test, y_pred)*100, 3)), fontsize = 10)
plt.text(-1.1, 0.13, "Train Accuracy = %s" % (round(accuracy_score(meyo, trainacc)*100, 3)), fontsize = 9)
plt.text(-1, 0.3, "Sensitivity = %s" % (sen1), fontsize = 8)
plt.text(-1, 0.5, "Specificity = %s" % (spec1), fontsize = 8)
plt.text(-1, 0.7, "Precision = %s" % (prec1), fontsize = 8)
plt.show()
utkan1.savefig("final_svmClassifier_genus_autism_cm.pdf", bbox_inches = "tight")


sortedb = lgbm.feature_importances_.argsort()
plt.rc('ytick', labelsize=4)
plt.rc('xtick', labelsize=7)
utak = plt.figure(figsize=(5,4))
plt.barh(X.columns[sortedb[440:]], lgbm.feature_importances_[sortedb[440:]])
plt.show()
utak.savefig("final_LGBMClassifier_autism_genus_feature_importance_barplot.pdf", bbox_inches = "tight")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc_score = roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], 'k--')  # Plot the random guess line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('roc_curve_lgbm.pdf')

# Show the plot
plt.show()




