# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 06:02:32 2023

@author: PC
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 06:39:26 2023

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
skfold = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 10, random_state = 30)

param_test0 = {
  'num_leaves' : [10,15,25,31,35,50],
  'learning_rate' : [0.01,0.001],
  'n_estimators' : [500,1000,2000]
}
gsearch0 = GridSearchCV(estimator =
                        ltb.LGBMClassifier(boosting_type = 'gbdt', num_leaves = 10,
                                           learning_rate =0.01, scale_pos_weight = 0.81,
                                           min_child_samples = 20,
                                           objective = 'binary',
                                           n_estimators=2000, max_depth=-1, min_child_weight=1e-3,
                                           reg_alpha=0, reg_lambda=0, subsample=1,
                                           colsample_bytree=1, random_state=30), 
                        param_grid = param_test0, scoring='f1_macro', n_jobs=-1, cv=skfold)
gsearch0.fit(eyo, meyo)
gsearch0.cv_results_, gsearch0.best_params_, gsearch0.best_score_

param_test1 = {
  'learning_rate' : [0.00005, 0.0001],
  'n_estimators':[75000, 100000]
}
gsearch1 = GridSearchCV(estimator =
                        ltb.LGBMClassifier(boosting_type = 'gbdt', num_leaves = 10,
                                           learning_rate =0.1, scale_pos_weight = 0.81,
                                           min_child_samples = 20,
                                           objective = 'binary',
                                           n_estimators=1000, max_depth=-1, min_child_weight=1e-3,
                                           reg_alpha=0, reg_lambda=0, subsample=1,
                                           colsample_bytree=1, random_state=30), 
                        param_grid = param_test1, scoring='f1_macro', n_jobs=-1, cv=skfold)
gsearch1.fit(eyo, meyo)
gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_

##
param_test2 = {
  'max_depth':range(2,50,2),
  'min_child_samples':range(10,30,2)
}
gsearch2 = GridSearchCV(estimator = 
                        ltb.LGBMClassifier(boosting_type = 'gbdt', num_leaves = 10,
                                           learning_rate =0.01, scale_pos_weight = 0.81,
                                           min_child_samples = 22,
                                           objective = 'binary',
                                           n_estimators=2000, max_depth=6, min_child_weight=1e-3,
                                           reg_alpha=0, reg_lambda=0, subsample=1,
                                           colsample_bytree=1, random_state=30), 
                        param_grid = param_test2, scoring='f1_macro',n_jobs=-1, cv=skfold,)
gsearch2.fit(eyo, meyo)
gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_


##
param_test3 = {
  'max_depth':[12,14,16]
}
gsearch3 = GridSearchCV(estimator = 
                        ltb.LGBMClassifier(boosting_type = 'gbdt', num_leaves = 25,
                                           learning_rate =0.01, scale_pos_weight = 0.8,
                                           min_child_samples = 10,
                                           objective = 'binary',
                                           n_estimators=1000, max_depth=8, min_child_weight=1e-3,
                                           reg_alpha=0, reg_lambda=0, subsample=1,
                                           colsample_bytree=1, random_state=30), 
                        param_grid = param_test3, scoring='f1_macro',n_jobs=8, cv=skfold)
gsearch3.fit(eyo, meyo)
gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_

##
param_test4 = {
  'reg_alpha':[i/10.0 for i in range(0,10,1)]
}
gsearch4 = GridSearchCV(estimator =
                        ltb.LGBMClassifier(boosting_type = 'gbdt', num_leaves = 10,
                                           learning_rate =0.01, scale_pos_weight = 0.81,
                                           min_child_samples = 22,
                                           objective = 'binary',
                                           n_estimators=2000, max_depth=6, min_child_weight=1e-3,
                                           reg_alpha=0, reg_lambda=0, subsample=1,
                                           colsample_bytree=1, random_state=30), 
                        param_grid = param_test4, scoring='f1_macro',n_jobs=8, cv=skfold)
gsearch4.fit(eyo, meyo)
gsearch4.best_params_, gsearch4.best_score_

##
param_test5 = {
  'subsample':[i/10.0 for i in range(7,10,1)],
  'colsample_bytree':[i/10.0 for i in range(7,10,1)]
}
gsearch5 = GridSearchCV(estimator =
                        ltb.LGBMClassifier(boosting_type = 'gbdt', num_leaves = 10,
                                           learning_rate =0.01, scale_pos_weight = 0.81,
                                           min_child_samples = 22,
                                           objective = 'binary',
                                           n_estimators=2000, max_depth=6, min_child_weight=1e-3,
                                           reg_alpha=0, reg_lambda=0, subsample=0.7,
                                           colsample_bytree=0.8, random_state=30), 
                        param_grid = param_test5, scoring='f1_macro',n_jobs=-1, cv=skfold)
gsearch5.fit(eyo, meyo)
gsearch5.best_params_, gsearch5.best_score_

##
param_test6 = {
  'subsample':[i/100.0 for i in range(5,15,2)],
  'colsample_bytree':[i/100.0 for i in range(45,55,2)]
}
gsearch6 = GridSearchCV(estimator = 
                        ltb.LGBMClassifier(boosting_type = 'gbdt', num_leaves = 20,
                                           learning_rate =0.01, scale_pos_weight = 1,
                                           min_child_samples = 20,
                                           objective = 'binary',
                                           n_estimators=500, max_depth=6, min_child_weight=3,
                                           reg_alpha=0, reg_lambda=0, subsample=1,
                                           colsample_bytree=1, random_state=30), 
                        param_grid = param_test6, scoring='f1_macro',n_jobs=-1, cv=skfold)
gsearch6.fit(eyo, meyo)
gsearch6.best_params_, gsearch6.best_score_

##
param_test7 = {
  'reg_alpha':[1e-5, 2e-5, 3e-5, 4e-5]
}
gsearch7 = GridSearchCV(estimator =
                        ltb.LGBMClassifier(boosting_type = 'gbdt', num_leaves = 15,
                                           learning_rate =0.01, scale_pos_weight = 0.8,
                                           min_child_samples = 3,
                                           objective = 'binary',
                                           n_estimators=2000, max_depth=10, min_child_weight=1e-3,
                                           reg_alpha=0, reg_lambda=0, subsample=0.4,
                                           colsample_bytree=0.4, random_state=30), 
                        param_grid = param_test7, scoring='f1_macro',n_jobs=-1, cv=skfold)
gsearch7.fit(eyo, meyo)
gsearch7.best_params_, gsearch7.best_score_

##
param_test8 = {
  'random_state': range(20,55,1)
}
gsearch8 = GridSearchCV(estimator = ltb.LGBMClassifier(boosting_type = 'gbdt', num_leaves = 15,
                   learning_rate =0.01, scale_pos_weight = 0.8,
                   min_child_samples = 3,
                   objective = 'binary',
                   n_estimators=2000, max_depth=10, min_child_weight=1e-3,
                   reg_alpha=0, reg_lambda=0.1, subsample=0.4,
                   colsample_bytree=0.4, random_state=36), 
                        param_grid = param_test8, scoring='f1_macro',n_jobs=-1, cv=skfold)
gsearch8.fit(eyo, meyo)
gsearch8.best_params_, gsearch8.best_score_

##
param_test9 = {
  'num_leaves':range(2,50,2)
}
gsearch9 = GridSearchCV(estimator = ltb.LGBMClassifier(boosting_type = 'gbdt', num_leaves = 25,
                                                       learning_rate =0.01, scale_pos_weight = 0.8,
                                                       min_child_samples = 10,
                                                       objective = 'binary',
                                                       n_estimators=1000, max_depth=8, min_child_weight=1e-3,
                                                       reg_alpha=0, reg_lambda=0, subsample=0.7,
                                                       colsample_bytree=0.9, random_state=30), 
                        param_grid = param_test9, scoring='f1_macro',n_jobs=-1, cv=skfold)
gsearch9.fit(eyo, meyo)
gsearch9.best_params_, gsearch9.best_score_

##
param_test10 = {
  'reg_lambda':[0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]
}
gsearch10 = GridSearchCV(estimator = 
                         ltb.LGBMClassifier(boosting_type = 'gbdt', num_leaves = 10,
                                            learning_rate =0.01, scale_pos_weight = 0.81,
                                            min_child_samples = 22,
                                            objective = 'binary',
                                            n_estimators=2000, max_depth=6, min_child_weight=1e-3,
                                            reg_alpha=0, reg_lambda=0, subsample=0.7,
                                            colsample_bytree=0.8, random_state=30), 
                        param_grid = param_test10, scoring='f1_macro',n_jobs=-1, cv=skfold)
gsearch10.fit(eyo, meyo)
gsearch10.best_params_, gsearch10.best_score_

##
param_test11 = {
  'min_split_gain':[0, 0.05, 0.1, 1, 10, 100]
}
gsearch11 = GridSearchCV(estimator =
                         ltb.LGBMClassifier(boosting_type = 'gbdt', num_leaves = 15,
                                            learning_rate =0.01, scale_pos_weight = 0.8,
                                            min_child_samples = 3,
                                            objective = 'binary',
                                            n_estimators=2000, max_depth=10, min_child_weight=1e-3,
                                            reg_alpha=0, reg_lambda=0.1, subsample=0.4,
                                            colsample_bytree=0.4, random_state=30), 
                        param_grid = param_test11, scoring='f1_macro',n_jobs=-1, cv=skfold)
gsearch11.fit(eyo, meyo)
gsearch11.best_params_, gsearch11.best_score_

#%%
lgbm = ltb.LGBMClassifier(boosting_type = 'gbdt', num_leaves = 25,
                   learning_rate =0.01, scale_pos_weight = 0.81,
                   min_child_samples = 3,
                   objective = 'binary',
                   n_estimators=2000, max_depth=10, min_child_weight=1e-3,
                   reg_alpha=0, reg_lambda=0.1, subsample=0.4,
                   colsample_bytree=0.4, random_state=30)
lgbm.fit(eyo, meyo)
y_pred = lgbm.predict(X_test)
y_pred_proba = lgbm.predict_proba(X_test)
y_pred_proba = y_pred_proba[:, 1]
trainacc = lgbm.predict(eyo)
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
utkan1.savefig("final_LGBMClassifier_genus_autism_cm.pdf", bbox_inches = "tight")


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
plt.savefig('roc_curve_lgbm.pdf', bbox_inches = "tight")
plt.show()
# Show the plot





