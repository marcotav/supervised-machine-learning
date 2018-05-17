## Churn Analysis [[view code]](http://nbviewer.jupyter.org/github/marcotav/machine-learning-regression-models/blob/master/retail/notebooks/retail-recommendations.ipynb)
![image title](https://img.shields.io/badge/work-in%20progress-blue.svg) ![image title](https://img.shields.io/badge/statsmodels-v0.8.0-blue.svg) ![Image title](https://img.shields.io/badge/sklearn-0.19.1-orange.svg) ![Image title](https://img.shields.io/badge/seaborn-v0.8.1-yellow.svg) ![Image title](https://img.shields.io/badge/pandas-0.22.0-red.svg) ![Image title](https://img.shields.io/badge/numpy-1.14.2-green.svg) ![Image title](https://img.shields.io/badge/matplotlib-v2.1.2-orange.svg)

**The code is available [here](http://nbviewer.jupyter.org/github/marcotav/machine-learning-regression-models/blob/master/retail/notebooks/retail-recommendations.ipynb) or by clicking on the.**

This project was done in collaboration with [Corey Girard](https://github.com/coreygirard/)

<br>

<p align="center">
  <img src="images/cellphone.jpg", width = "400">
</p>                                                                  
<p align="center">
  <a href="#goals"> Goals </a> •
  <a href="#importance"> Why this is important? </a> •
  <a href="#mods"> Importing modules and reading the data </a> •
  <a href="#dh"> Data Handling and Feature Engineering </a> •
  <a href="#Xy"> Features and target </a> •
  <a href="#pp"> Using `pandas-profiling` and rejecting variables with correlations above 0.9 </a> •
  <a href="#scale">  Scaling </a> •
  <a href="#rf"> Building a random forest classifier using GridSearch to optimize hyperparameters </a>
</p>


<a id = 'goals'></a>
### Goals
From Wikipedia, 

> Churn rate is a measure of the number of individuals or items moving out of a collective group over a specific period. It is one of two primary factors that determine the steady-state level of customers a business will support [...] It is an important factor for any business with a subscriber-based service model, [such as] mobile telephone networks.

Our goal in this analysis was to predict the churn rate from a mobile phone company based on customer attributes including:
- Area code
- Call duration at different hours
- Charges
- Account length

See [this website](http://blog.yhat.com/posts/predicting-customer-churn-with-sklearn.html) for a similar analysis.

<a id = 'importance'></a>
### Why this is important? 

It is a well-known fact that in several businesses (particularly the ones involving subscriptions), the acquisition of new customers costs much more than the retention of existing ones. A thorough analysis of what causes churn-rates and how to predict them can be used to build efficient customer retention strategies.

<a id = 'mods'></a>
## Importing modules and reading the data
```
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```
Reading the data:
```
df = pd.read_csv("data.csv")
```
<p align="center">
  <img src="images/df_churn_new.png", width = "950">
</p> 

<a id = 'dh'></a>
## Data Handling and Feature Engineering
In this section the following steps are taken:
- Conversion of strings into booleans 
- Conversion of booleans to integers
- Converting the states column into dummy columns
- Creation of several new features (feature engineering)

The commented code follows (most of the lines were ommited for brevity):
```
# convert binary strings to boolean ints
df['international_plan'] = df.international_plan.replace({'Yes': 1, 'No': 0})
#convert booleans to boolean ints
df['churn'] = df.churn.replace({True: 1, False: 0})
# handle state and area code dummies
state_dummies = pd.get_dummies(df.state)
state_dummies.columns = ['state_'+c.lower() for c in state_dummies.columns.values]
df.drop('state', axis='columns', inplace=True)
df = pd.concat([df, state_dummies], axis='columns')
area_dummies = pd.get_dummies(df.area_code)
area_dummies.columns = ['area_code_'+str(c) for c in area_dummies.columns.values]
df.drop('area_code', axis='columns', inplace=True)
df = pd.concat([df, area_dummies], axis='columns')
# feature engineering
df['total_minutes'] = df.total_day_minutes + df.total_eve_minutes + df.total_intl_minutes
df['total_calls'] = df.total_day_calls + df.total_eve_calls + df.total_intl_calls
```

<a id = 'Xy'></a>
### Features and target
Defining the features matrix and the target (the churn):
```
X = df[[c for c in df.columns if c != 'churn']]
y = df.churn
```

<a id = 'pp'></a>
### Using `pandas-profiling` and rejecting variables with correlations above 0.9

The package `pandas-profiling` contains a method `get_rejected_variables(threshold)` which identifies variables with correlation higher than a threshold.
```
import pandas_profiling
profile = pandas_profiling.ProfileReport(X)
rejected_variables = profile.get_rejected_variables(threshold=0.9)
X = X.drop(rejected_variables,axis=1)
```
<a id = 'scale'></a>
### Scaling
```
from sklearn.preprocessing import StandardScaler
cols = X.columns.tolist()
scaler = StandardScaler()
X[cols] = scaler.fit_transform(X[cols])
X = X[cols]
```
We can now build our models.

<a id = 'rf'></a>
## Building models and using `GridSearch` to optimize hyperparameters


### Finding best hyperparameters

```
n_estimators = list(range(20,160,10))
max_depth = list(range(2, 16, 2)) + [None]
def rfscore(X,y,test_size,n_estimators,max_depth):

    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, test_size = test_size, random_state=42) 
    rf_params = {
             'n_estimators':n_estimators,
             'max_depth':max_depth}   # parameters for grid search
    rf_gs = GridSearchCV(RandomForestClassifier(), rf_params, cv=5, verbose=1, n_jobs=-1)
    rf_gs.fit(X_train,y_train) # training the random forest with all possible parameters
    max_depth_best = rf_gs.best_params_['max_depth']      # getting the best max_depth
    n_estimators_best = rf_gs.best_params_['n_estimators']  # getting the best n_estimators
    print("best max_depth:",max_depth_best)
    print("best n_estimators:",n_estimators_best)
    best_rf_gs = RandomForestClassifier(max_depth=max_depth_best,n_estimators=n_estimators_best) # instantiate the best model
    best_rf_gs.fit(X_train,y_train)  # fitting the best model
    best_rf_score = best_rf_gs.score(X_test,y_test) 
    print ("best score is:",round(best_rf_score,3))
    preds = best_rf_gs.predict(X_test)
    df_pred = pd.DataFrame(np.array(preds).reshape(len(preds),1))
    df_pred.columns = ['predictions']
    print('Features and their importance:\n')
    feature_importances = pd.Series(best_rf_gs.feature_importances_, index=X.columns).sort_values().tail(10)
    print(feature_importances)
    print(feature_importances.plot(kind="barh", figsize=(6,6)))
    return (df_pred,max_depth_best,n_estimators_best)


triple = rfscore(X,y,0.3,n_estimators,max_depth)
```
```
df_pred = triple[0]
```
The predictions are:
```
df_pred['predictions'].value_counts()/df_pred.shape[0]
```

<p align="center">
  <img src="images/predictions.png", width = "150">
</p> 



### Cross Validation
```
def cv_score(X,y,cv,n_estimators,max_depth):
    rf = RandomForestClassifier(n_estimators=n_estimators_best,
                                max_depth=max_depth_best)
    s = cross_val_score(rf, X, y, cv=cv, n_jobs=-1)
    return("{} Score is :{:0.3} ± {:0.3}".format("Random Forest", s.mean().round(3), s.std().round(3)))
```
```
dict_best = {'max_depth': triple[1], 'n_estimators': triple[2]}
n_estimators_best = dict_best['n_estimators']
max_depth_best = dict_best['max_depth']
cv_score(X,y,5,n_estimators_best,max_depth_best)
```
The output is:
```
'Random Forest Score is :0.774 ± 0.054'
```

### Confusion Matrix

```
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, test_size = 0.3, random_state=42) 

best_rf_gs = RandomForestClassifier(n_estimators=n_estimators_best,
                                max_depth=max_depth_best)

best_rf_gs.fit(X_train,y_train)  # fitting the best model
best_rf_score = best_rf_gs.score(X_test,y_test) 
preds = best_rf_gs.predict(X_test)
pd.crosstab(pd.concat([X_test,y_test],axis=1)['churn'], preds, 
            rownames=['Actual Values'], colnames=['Predicted Values'])
```
The confusion matrix is:

<p align="center">
  <img src="images/cm.png", width = "200">
</p> 
