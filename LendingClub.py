#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 19:07:22 2018

@author: jonathan
"""
import matplotlib.pyplot as plt
import os
os.getcwd()
os.chdir('/home/jonathan/Downloads')
import pandas as p
import numpy as np
lendingclub = p.read_stata('LendingClub.dta')

default = p.get_dummies(lendingclub['loan_status'])
lendingclub = lendingclub.join(default)

#drop all loans in grace period or late to focus on charged off vs current/fully-paid
lendingclub = lendingclub[lendingclub['Late (16-30 days)'] == 0]
lendingclub = lendingclub[lendingclub['Late (31-120 days)'] == 0]
lendingclub = lendingclub[lendingclub['In Grace Period'] == 0]
lendingclub= lendingclub[lendingclub['Default']==0] #default has no data except for one line
lendingclub['Good Loan'] = lendingclub['Fully Paid'] + lendingclub['Current']
lendingclub = lendingclub.drop(['Late (16-30 days)','Late (31-120 days)','In Grace Period','Default'], axis = 1)
lendingclub['Bad Loan'] = lendingclub['Charged Off']
lendingclub = lendingclub.drop(['Charged Off','Current','Fully Paid'], axis = 1)
lendingclub = lendingclub[lendingclub.index != 3765] #bad data
lendingclub = lendingclub[lendingclub.index != 3324] #bad data

#dummies and some feature engineering
grade = p.get_dummies(lendingclub['grade'])
grade_sub = p.get_dummies(lendingclub['sub_grade'])
verification_status = p.get_dummies(lendingclub['verification_status'])
verification_status['Verified_combined'] = verification_status['Source Verified'] + verification_status['Verified']
verification_status = verification_status[['Not Verified','Verified_combined']]
home_ownership = p.get_dummies(lendingclub['home_ownership'])
employment_length = p.get_dummies(lendingclub['emp_length'])
lendingclub = lendingclub.join(grade)
lendingclub = lendingclub.join(grade_sub)
lendingclub = lendingclub.join(verification_status)
lendingclub = lendingclub.join(home_ownership)
lendingclub = lendingclub.join(employment_length)
lendingclub['Loan/Inst Ratio']= lendingclub['loan_amnt'] / lendingclub['installment']
grab = lendingclub['revol_util'].str.strip('%')
grab = grab.str.strip(' ')
grab = p.to_numeric(grab)
lendingclub['revol_util_clean'] = grab.astype(float)

#Check how much data we have at each subgrade
#['A1','A2','A3','A4','A5','B1','B2','B3','B4','B5','C1','C2','C3','C4','C5','D1','D2','D3','D4','D5','E1','E2','E3','E4','E5','F1','F2','F3','F4','F5','G1','G2','G3','G4','G5']
lendingclub[['A1','A2','A3','A4','A5','B1','B2','B3','B4','B5','C1','C2','C3','C4','C5','D1','D2','D3','D4','D5','E1','E2','E3','E4','E5','F1','F2','F3','F4','F5','G1','G2','G3','G4','G5']].sum()

#fit simple model first to visualize
from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression()
logistic_model.fit(lendingclub[['A1','A2','A3','A4','A5','B1','B2','B3','B4','B5','C1','C2','C3','C4','C5','D1','D2','D3','D4','D5','E1','E2','E3','E4','E5','F1','F2','F3','F4','F5','G1','G2','G3','G4','G5']], lendingclub['Bad Loan'])
#predicted probability in sample
pred_probs = logistic_model.predict_proba(lendingclub[['A1','A2','A3','A4','A5','B1','B2','B3','B4','B5','C1','C2','C3','C4','C5','D1','D2','D3','D4','D5','E1','E2','E3','E4','E5','F1','F2','F3','F4','F5','G1','G2','G3','G4','G5']])
#visualize
plt.scatter(lendingclub["sub_grade"], pred_probs[:,1])
logistic_model.coef_

#fit another - use this one to swap variables and play around
logistic_model1 = LogisticRegression()
#lendingclub.loc[lendingclub['revol_util_clean']==0,'revol_util_clean'] = 0.1
lendingclub = lendingclub[lendingclub['revol_util_clean'] >= 0] #take out 50 zeros, creates error, still good enough size to infer, for revol_util calc only
logistic_model1.fit(lendingclub[['1 year','10+ years','2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','< 1 year','n/a']], lendingclub['Bad Loan'])
pred_probs1 = logistic_model1.predict_proba(lendingclub[['1 year','10+ years','2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','< 1 year','n/a']])
plt.scatter(lendingclub["emp_length"],pred_probs1[:,1])
plt.xticks(rotation=90)

#re fit model with different variables
#original model: ['int_rate', 'A','B','C','D','E','F','G','annual_inc']
#since lending club sets its own rates based on grade/sub-grade, lets built model without to see if we can get good results
#there is no FICO score in this example data, can be an important missing category
#my model on raw loan data --> ['annual_inc','Loan/Inst Ratio','dti','inq_last_6mths','n/a','OTHER','pub_rec']
logistic_model = LogisticRegression()
logistic_model.fit(lendingclub[['annual_inc','Loan/Inst Ratio','dti','inq_last_6mths','n/a','OTHER','pub_rec']], lendingclub['Bad Loan'])

fitted_labels = logistic_model.predict(lendingclub[['annual_inc','Loan/Inst Ratio','dti','inq_last_6mths','n/a','OTHER','pub_rec']])
pred_probs = logistic_model.predict_proba(lendingclub[['annual_inc','Loan/Inst Ratio','dti','inq_last_6mths','n/a','OTHER','pub_rec']])

#prediction, this may be able to be deleted
#no loan has a more than 50 percent chance of defaulting so all predictions are no default
plt.scatter(lendingclub["int_rate"], fitted_labels)

#accurancy
labels = logistic_model.predict(lendingclub[['annual_inc','Loan/Inst Ratio','dti','inq_last_6mths','n/a','OTHER','pub_rec']])
lendingclub["predicted_label"] = labels
lendingclub["actual_label"] = lendingclub["Bad Loan"]
matches = lendingclub["predicted_label"] == lendingclub["actual_label"] #when it was right
correct_predictions = lendingclub[matches]
print(correct_predictions.head())
accuracy = len(correct_predictions) / len(lendingclub)
print(accuracy)
#can we trade accuracy for differentiation?

#TP,TN,FP,FN
#this analysis wont hold much merit with all predictions the same
true_positive_filter = (lendingclub["predicted_label"] == 1) & (lendingclub["actual_label"] == 1)
true_positives = len(lendingclub[true_positive_filter])
true_negative_filter = (lendingclub["predicted_label"] == 0) & (lendingclub["actual_label"] == 0)
true_negatives = len(lendingclub[true_negative_filter])
print(true_positives)
print(true_negatives)
false_negative_filter = (lendingclub["predicted_label"] == 0) & (lendingclub["actual_label"] == 1)
false_negatives = len(lendingclub[false_negative_filter])
#true positive rate
sensitivity = true_positives / (true_positives + false_negatives)
print(sensitivity)
false_positive_filter = (lendingclub["predicted_label"] == 1) & (lendingclub["actual_label"] == 0)
false_positives = len(lendingclub[false_positive_filter])
#true negative rate
specificity = (true_negatives) / (false_positives + true_negatives)
print(specificity)
#one specificity , zero sensitivity

##empirical probabilities are less than 50 percent, we care about relative rankings so lets look at naive bayes
from sklearn.naive_bayes import MultinomialNB
#fit the prior to manipulate the output
#I am basically setting sensitivity with prior
clf = MultinomialNB(alpha=1, class_prior=[.01,.999], fit_prior=True)
clf.fit(lendingclub[['annual_inc','Loan/Inst Ratio','dti','inq_last_6mths','n/a','OTHER','pub_rec']], lendingclub["Bad Loan"])
out = clf.predict(lendingclub[['annual_inc','Loan/Inst Ratio','dti','inq_last_6mths','n/a','OTHER','pub_rec']])
lendingclub["predicted_label_NB"] = out
matches = lendingclub["predicted_label_NB"] == lendingclub["actual_label"] #when it was right
correct_predictions = lendingclub[matches]
print(correct_predictions.head())
accuracy = len(correct_predictions) / len(lendingclub)
print(accuracy)

#TP,TN,FP,FN
#this time will be better
true_positive_filter = (lendingclub["predicted_label_NB"] == 1) & (lendingclub["actual_label"] == 1)
true_positives = len(lendingclub[true_positive_filter])
true_negative_filter = (lendingclub["predicted_label_NB"] == 0) & (lendingclub["actual_label"] == 0)
true_negatives = len(lendingclub[true_negative_filter])
print(true_positives)
print(true_negatives)
false_negative_filter = (lendingclub["predicted_label_NB"] == 0) & (lendingclub["actual_label"] == 1)
false_negatives = len(lendingclub[false_negative_filter])
#true positive rate
sensitivity = true_positives / (true_positives + false_negatives)
print(sensitivity)
false_positive_filter = (lendingclub["predicted_label_NB"] == 1) & (lendingclub["actual_label"] == 0)
false_positives = len(lendingclub[false_positive_filter])
#true negative rate
specificity = (true_negatives) / (false_positives + true_negatives)
print(specificity)
#since not many loans have default, true positive rate seems important (bad loan = 1)
#we want to be sensititve to a bad loan characteristic even if we give up accuracy
total_defaults = np.sum(lendingclub['Bad Loan'])
#with this uneducated prior capture 3625/5657 defaults

#If I invest in all loans
#good loan column --> 1 for good loan, multiply bad loans by zero
lendingclub['gains'] = lendingclub['loan_amnt']*lendingclub['int_rate'] * lendingclub['Good Loan']
gains = np.sum(lendingclub['gains'])
lendingclub['losses'] = lendingclub['loan_amnt'] * lendingclub['Bad Loan']*-1
losses = np.sum(lendingclub['losses'])
profit_arb = gains + losses

#If I invest in just loans selected by Naive Bayes
#multiply by predicted_label_NB
lendingclub['profit'] = lendingclub['gains'] + lendingclub['losses']
lendingclub['Good_Loan_NB'] = np.abs(1-lendingclub['predicted_label_NB']) 
lendingclub['profit_NB'] = lendingclub['Good_Loan_NB'] * lendingclub['profit']
profit_NB = np.sum(lendingclub['profit_NB'])
#not necessarily apples to apples as benchmark buys all loans
#and mine buys small subset, good way to screen and reduce overall investment

#superficial analysis but we see improvement
start_NB = np.sum(lendingclub['Good_Loan_NB']*lendingclub['loan_amnt'])
start_arb = np.sum(lendingclub['loan_amnt'])
return_NB = (start_NB + profit_NB) / (start_NB) - 1 
return_arb = (start_arb + profit_arb) / (start_arb) - 1 
#assuming no recoveries
#assuming we invest full value in all loans --> likely want more of an equal weight
#signal isnt clean --> this is more of an example of workflow

logistic_model.coef_


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X = lendingclub[['A1','A2','A3','A4','A5','B1','B2','B3','B4','B5','C1','C2','C3','C4','C5','D1','D2','D3','D4','D5','E1','E2','E3','E4','E5','F1','F2','F3','F4','F5','G1','G2','G3','G4','G5']]
y = lendingclub[['Bad Loan']]

clf = RandomForestClassifier(max_depth=5,random_state=0)
clf.fit(X,y)
out = clf.predict_proba(X)
