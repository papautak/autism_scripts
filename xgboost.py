

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
  'learning_rate': [0.0001, 0.001, 0.01, 0.1],
  'n_estimators': [100,200,500,1000,2000,5000]
}
gsearch0 = GridSearchCV(estimator =
                        XGBClassifier( learning_rate =0.05,
                                      n_estimators=1000, max_depth=6,
                                      min_child_weight=1, gamma=0, subsample=0.8,
                                      colsample_bytree=0.8,
                                      objective= 'binary:logistic', reg_alpha = 0,
                                      scale_pos_weight=0.81, random_state=10), 
                        param_grid = param_test0, scoring='f1_macro',n_jobs=-1, cv=skfold)
gsearch0.fit(eyo, meyo)
gsearch0.best_params_, gsearch0.best_score_

##
param_test1 = { 
  'random_state': range(1,21,1)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.01, n_estimators=1000, max_depth=5,
  min_child_weight=1, gamma=0, subsample=0.7, colsample_bytree=0.7,
  objective= 'binary:logistic', reg_alpha = 0, scale_pos_weight=1, random_state=10), 
  param_grid = param_test1, scoring='roc_auc',n_jobs=-1, cv=skfold)
gsearch1.fit(eyo, meyo)
gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_

##
param_test2 = {
  'max_depth':range(1,8,1),
  'min_child_weight':range(1,8,1)
}
gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.05,
              n_estimators=1000, max_depth=3,
              min_child_weight=1, gamma=0, subsample=0.8,
              colsample_bytree=0.8,
              objective= 'binary:logistic', reg_alpha = 0,
              scale_pos_weight=0.81, random_state=10), 
  param_grid = param_test2, scoring='f1_macro',n_jobs=-1, cv=skfold)
gsearch2.fit(eyo, meyo)
gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_

##
param_test3 = {
  'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.05,
              n_estimators=1000, max_depth=3,
              min_child_weight=1, gamma=0, subsample=0.8,
              colsample_bytree=0.8,
              objective= 'binary:logistic', reg_alpha = 0,
              scale_pos_weight=0.81, random_state=10), 
  param_grid = param_test3, scoring='f1_macro',n_jobs=-1, cv=skfold)
gsearch3.fit(eyo, meyo)
gsearch3.best_params_, gsearch3.best_score_

##
param_test4 = {
  'subsample':[i/100.0 for i in range(60,100,10)],
  'colsample_bytree':[i/100.0 for i in range(60,100,10)]
}
gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.05,
              n_estimators=1000, max_depth=3,
              min_child_weight=1, gamma=0, subsample=0.8,
              colsample_bytree=0.9,
              objective= 'binary:logistic', reg_alpha = 0,
              scale_pos_weight=0.81, random_state=100), 
  param_grid = param_test4, scoring='f1_macro',n_jobs=-1, cv=skfold)
gsearch4.fit(eyo, meyo)
gsearch4.best_params_, gsearch4.best_score_

##
# param_test5 = {
#   'subsample':[i/100.0 for i in range(75,90,3)],
#   'colsample_bytree':[i/100.0 for i in range(65,75,3)]
# }
# gsearch5 = GridSearchCV(estimator = XGBClassifier(estimator = XGBClassifier(learning_rate =0.01, n_estimators=2000, max_depth=6,
#   min_child_weight=1, gamma=0, subsample=0.6, colsample_bytree=0.4,
#   objective= 'binary:logistic', reg_alpha = 0, scale_pos_weight=0.8, random_state=10), 
#   param_grid = param_test4, scoring='accuracy',n_jobs=-1, cv=skfold)
# gsearch5.fit(eyo, meyo)
# gsearch4.best_params_, gsearch4.best_score_

##
param_test6 = {
  'reg_alpha':[0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 100]
}
gsearch6 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.05,
              n_estimators=1000, max_depth=3,
              min_child_weight=1, gamma=0, subsample=0.8,
              colsample_bytree=0.9,
              objective= 'binary:logistic', reg_alpha = 0.001,
              scale_pos_weight=0.81, random_state=100), 
  param_grid = param_test6, scoring='f1_macro',n_jobs=-1, cv=skfold)
gsearch6.fit(eyo, meyo)
gsearch6.best_params_, gsearch6.best_score_

##
# param_test7 = {
#   'scale_pos_weight':[i/10 for i in range(1,11,1)]
# }
# gsearch7 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.01, n_estimators=1000, max_depth=7,
#   min_child_weight=3, gamma=0.2, subsample=0.7, colsample_bytree=0.5,
#   objective= 'binary:logistic', reg_alpha = 1e-05, scale_pos_weight=1, random_state=9), 
#   param_grid = param_test7, scoring='roc_auc',n_jobs=-1, cv=skfold)
# gsearch7.fit(eyo, meyo)
# gsearch7.best_params_, gsearch7.best_score_
#%%
xgb = xgb.XGBClassifier(learning_rate =0.05,
              n_estimators=1000, max_depth=3,
              min_child_weight=1, gamma=0, subsample=0.8,
              colsample_bytree=0.9,
              objective= 'binary:logistic', reg_alpha = 0.001,
              scale_pos_weight=0.81, random_state=100)
xgb.fit(eyo, meyo)
y_pred = xgb.predict(X_test)
y_pred_proba = xgb.predict_proba(X_test)
y_pred_proba = y_pred_proba[:, 1]
trainacc = xgb.predict(eyo)
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
utkan1.savefig("final_XGBClassifier_genus_autism_cm.pdf", bbox_inches = "tight")


sortedb = xgb.feature_importances_.argsort()
plt.rc('ytick', labelsize=4)
plt.rc('xtick', labelsize=7)
utak = plt.figure(figsize=(5,4))
plt.barh(X.columns[sortedb[440:]], xgb.feature_importances_[sortedb[440:]])
plt.show()
utak.savefig("final_XGBClassifier_autism_genus_feature_importance_barplot.pdf", bbox_inches = "tight")

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
plt.savefig('roc_curve_xgb.pdf', bbox_inches = "tight")

# Show the plot
plt.show()
