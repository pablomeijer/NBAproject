# -*- coding: utf-8 -*-
"""
Created on Mon May  9 17:49:27 2022

@author: pablo
"""

# Base --------------------------------------------------------------------------
import pandas as pd
import numpy as np
from plotnine import *

# Viz ---------------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
plt.style.use('seaborn')
rcParams['figure.figsize'] = 15, 6

# Models ------------------------------------------------------------------------
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score 
from sklearn.metrics import precision_recall_curve, plot_precision_recall_curve

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.model_selection import learning_curve, ShuffleSplit, validation_curve

# Association Rules -----------------------------------------
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


#All interpretation questions / written questions are not in this doc 
nba = pd.read_csv("nba_longevity.csv")
nba.rename(columns = {"TARGET_5Yrs" : "Target_5yrs"}, inplace = True)
#Checking if all columns are numerical - true
nba.dtypes

features = list(nba.select_dtypes(include = [np.number]).drop('Target_5yrs', axis = 1).columns)

#Creating parameters so that no value can be 0, which would wrongly classify the outcomes
params = {'var_smoothing': np.logspace(0, -3, 20)}

model = GaussianNB()
skf = StratifiedKFold(n_splits = 10)

# Grid Search in the parameter space
search = GridSearchCV(model,
                      param_grid = params,
                      cv = skf,
                      scoring = 'accuracy',
                      n_jobs = -1,
                      return_train_score = True)

# Train-validation split
x_train, x_test, y_train, y_test = train_test_split(nba[features],
                                                    nba['Target_5yrs'],
                                                    random_state = 101,
                                                    stratify = nba['Target_5yrs'])


# Fit the grid
search.fit(x_train, y_train)

#Finding the best constant
search.best_estimator_
#0.6951927961775606

#Classifier - changed the 
classifier = GaussianNB(var_smoothing = 0.6951927961775606)
classifier.fit(x_train, y_train)

#Validation Curve
param_range = np.logspace(-9, 0, 20)
train_scores, test_scores = validation_curve(classifier,
                                             nba[features],
                                             nba['Target_5yrs'],
                                             cv = 10,
                                             n_jobs = -1,
                                             scoring = 'accuracy',
                                             param_range = param_range,
                                             param_name = 'var_smoothing')

train_scores_mean = np.mean(train_scores, axis = 1)
train_scores_std = np.std(train_scores, axis = 1)

test_scores_mean = np.mean(test_scores, axis = 1)
test_scores_std = np.std(test_scores, axis = 1)


#Plotting the Validation Curve
plt.figure(figsize = (10,5))
plt.semilogx(param_range, train_scores_mean, label = 'Training Scores', color = 'red')
plt.fill_between(param_range,
                 train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std,
                 color = 'red',
                 alpha = 0.2,
                 lw = 2)

plt.semilogx(param_range, test_scores_mean, label = 'Test Scores', color = 'blue')
plt.fill_between(param_range,
                 test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std,
                 color = 'blue',
                 alpha = 0.2,
                 lw = 2)

plt.title('Validation Curve', fontsize = 18)
plt.xlabel('Var Smoothing', fontsize = 14)
plt.ylabel('Accuracy', fontsize = 14)
plt.legend(loc = 'best')

plt.show()

#LEARNING CURVE
train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(classifier,
                                                                   nba[features],
                                                                   nba['Target_5yrs'],
                                                                   n_jobs = -1,
                                                                   cv = ShuffleSplit(n_splits = 100, test_size = 0.20, random_state = 101),
                                                                   train_sizes = np.linspace(.1, 1.0, 10),
                                                                   return_times = True)

# Scores
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
fit_times_mean = np.mean(fit_times, axis=1)
fit_times_std = np.std(fit_times, axis=1)

# Plotting the Learning Curve and Scalability
plt.figure(figsize = (18,8))
plt.suptitle("Naive Bayes", fontsize = 25)

plt.subplot2grid((1,2), (0,0))
plt.fill_between(train_sizes, 
                 train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, 
                 alpha=0.1,
                 color="red")
plt.fill_between(train_sizes, 
                 test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, 
                 alpha=0.1,
                 color="green")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.xlabel("Training examples", fontsize = 15)
plt.ylabel("Score", fontsize = 15)
plt.title("Learning Curve", fontsize = 20)
plt.legend(loc="best")

plt.subplot2grid((1,2), (0,1))
plt.plot(train_sizes, fit_times_mean, 'o-')
plt.fill_between(train_sizes, 
                 fit_times_mean - fit_times_std,
                 fit_times_mean + fit_times_std, 
                 alpha=0.1)
plt.xlabel("Training examples", fontsize = 15)
plt.ylabel("Fit Times", fontsize = 15)
plt.title("Scalability of the model", fontsize = 20)

plt.tight_layout(rect = (0,0,0.95,0.95))
plt.show()



# Train the Model

x_train, x_test, y_train, y_test = train_test_split(nba[features],
                                                    nba['Target_5yrs'],
                                                    random_state = 101,
                                                    test_size = 0.19701493,
                                                    stratify = nba['Target_5yrs'])

classifier.fit(x_train, y_train)

# Predictions
preds = classifier.predict(x_test)
output = pd.DataFrame({'true': y_test,
                       'predictions': preds})

#Confusion Matrix
confusion_matrix(y_test, preds)


#Metrics
metrics.precision_score(y_test, preds, pos_label = 1)
metrics.recall_score(y_test, preds, pos_label = 1)
metrics.accuracy_score(y_test, preds)

#ROC AND PR Curve

model_probs = classifier.predict_proba(x_test.values)
fp, tp, thesholds = roc_curve(y_test, model_probs[:, 0], pos_label = 1)
prec, rec, _ = precision_recall_curve(y_test, model_probs[:, 0], pos_label = 1)

plt.figure(figsize = (10,5))
plt.subplot(1,2,1)
plt.plot(fp, tp)
plt.plot([0,1], [0,1], 'k--')
plt.title('ROC Curve', fontsize = 18)
plt.xlabel('False Positive Rate', fontsize = 14)
plt.ylabel('True Positive Rate', fontsize = 14)

plt.subplot(1,2,2)
plt.plot(rec, prec)
plt.title('PR Curve', fontsize = 18)
plt.xlabel('Precision', fontsize = 14)
plt.ylabel('Recall', fontsize = 14)

plt.show()

#UNSUPERVISED LEARNING CASE STUDY----------

groceries = pd.read_csv("groceries.csv")

columns = groceries.columns

groceries.dtypes
a = list(groceries.to_numpy())

for num in range(9835):
    a[num] = list(a[num])

for lst in range(9835):
    a[lst].pop(0)
    for num in range(a[lst].count(np.nan)):
        a[lst].remove(np.nan)

trans_enc = TransactionEncoder()
basket_trans = trans_enc.fit(a).transform(a)
basket_df = pd.DataFrame(basket_trans, columns = trans_enc.columns_)
basket_df

basket_df.shape
supports = apriori(basket_df, min_support=0.001, use_colnames=True)

ruled = association_rules(supports, metric = 'lift', min_threshold=0)

ruled_lifted = ruled.sort_values(by = ["lift"], ascending = False )
b = ruled_lifted.head(6)

q = ruled_lifted.loc[ruled_lifted.confidence > 0.5, :].loc[ruled_lifted.lift > 5, :].loc[ruled_lifted.conviction > 2, :].head(6)

#higher support for more obvious conclusions

supports_other = apriori(basket_df, min_support=0.0015, use_colnames=True)

ruled_other = association_rules(supports_other, metric = 'lift', min_threshold=0)

ruled_lifted_other = ruled_other.sort_values(by = ["lift"], ascending = False )
b = ruled_lifted.head(6)

other = ruled_lifted.loc[ruled_lifted_other.confidence > 0.5, :].loc[ruled_lifted_other.lift > 5, :].loc[ruled_lifted_other.conviction > 2, :]
