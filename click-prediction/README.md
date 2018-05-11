# Predicting clicks on ads

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![image title](https://img.shields.io/badge/work-in%20progress-blue.svg) ![image title](https://img.shields.io/badge/statsmodels-v0.8.0-blue.svg) ![Image title](https://img.shields.io/badge/sklearn-0.19.1-orange.svg) ![Image title](https://img.shields.io/badge/seaborn-v0.8.1-yellow.svg) ![Image title](https://img.shields.io/badge/pandas-0.22.0-red.svg) ![Image title](https://img.shields.io/badge/numpy-1.14.2-green.svg) ![Image title](https://img.shields.io/badge/matplotlib-v2.1.2-orange.svg)
<br>

<p align="center">
  <img src="images/click1.png" width="120",height="120">
</p>                                                                  
<p align="center">
  <a href="#Problem Statement"> Problem Statement </a> •
  <a href="#Dataset"> Dataset </a> •
</p>

<a id = 'Problem Statement'></a>
## Problem Statement

Borrowing from [here](https://turi.com/learn/gallery/notebooks/click_through_rate_prediction_intro.html):


> Many ads are actually sold on a "pay-per-click" (PPC) basis, meaning the company only pays for ad clicks, not ad views. Thus your optimal approach (as a search engine) is actually to choose an ad based on "expected value", meaning the price of a click times the likelihood that the ad will be clicked [...] In order for you to maximize expected value, you therefore need to accurately predict the likelihood that a given ad will be clicked, also known as "click-through rate" (CTR).

In this project I will predict the likelihood that a given online ad will be clicked.

## Dataset 

- The two files `train_click.csv` and `test_click.csv` contain ad impression attributes from a campaign.
- Each row in `train.csv` includes a `click` column.

## Import the relevant libraries and the files

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute   # used for feature imputation algorithms
pd.set_option('display.max_columns', None) # display all columns
pd.set_option('display.max_rows', None)  # displays all rows
%matplotlib inline
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all" # so we can see the value of multiple statements at once.
```

## Import the data

```
train = pd.read_csv('train_click.csv',index_col=0)
test = pd.read_csv('test_click.csv',index_col=0)
```

## Data Dictionary

The meaning of the columns follows:
- `location` – ad placement in the website
- `carrier` – mobile carrier 
- `device` – type of device e.g. phone, tablet or computer 
- `day` – weekday user saw the ad
- `hour` – hour user saw the ad
- `dimension` – size of ad

## Imbalance
The `click` column is **heavily** unbalanced. I will correct for this later.

```
import aux_func_v2 as af
af.s_to_df(train['click'].value_counts())
```

### Checking the variance of each feature

Let's quickly study the variance of the features to have an estimate of their impact on clicks. But let us first consider the cardinalities.

#### Train set cardinalities

```
cardin_train = [train[col].nunique() for col in train.columns.tolist()]
cols = [col for col in train.columns.tolist()]
d = {k:v for (k, v) in zip(cols,cardin_train)}
cardinal_train = pd.DataFrame(list(d.items()), columns=['column', 'cardinality'])
cardinal_train.sort_values('cardinality',ascending=False)
```

#### Test set cardinalities
```
cardin_test = [test[col].nunique() for col in test.columns.tolist()]
cols = [col for col in test.columns.tolist()]
d = {k:v for (k, v) in zip(cols,cardin_test)}
cardinal_test = pd.DataFrame(list(d.items()), columns=['column', 'cardinality'])
cardinal_test.sort_values('cardinality',ascending=False)
```

#### High and low cardinality in the training data

We can set *arbitrary* thresholds to determine the level of cardinality in the feature categories:

```
target = 'click'
cardinal_train_threshold = 33  # our choice
low_cardinal_train = cardinal_train[cardinal_train['cardinality'] 
                                    <= cardinal_train_threshold]['column'].tolist()
low_cardinal_train.remove(target)
high_cardinal_train = cardinal_train[cardinal_train['cardinality'] 
                                     > cardinal_train_threshold]['column'].tolist()
print('Features with low cardinal_train:\n',low_cardinal_train)
print('')
print('Features with high cardinal_train:\n',high_cardinal_train)
```

#### High and low cardinality in the test data

```
cardinal_test_threshold = 25  # chosen for low_cardinal_set to agree with low_cardinal_train
low_cardinal_test = cardinal_test[cardinal_test['cardinality'] 
                                  <= cardinal_test_threshold]['column'].tolist()
high_cardinal_test = cardinal_test[cardinal_test['cardinality']
                                   > cardinal_test_threshold]['column'].tolist()
print('Features with low cardinal_test:\n',low_cardinal_test)
print('')
print('Features with high cardinal_test:\n',high_cardinal_test)
```

#### Now let's look at the features' variances. 

From the bar plot below we see that `device_type` has non-negligible variance

```
from matplotlib import pyplot
import matplotlib.pyplot as plt

for col in low_cardinal_train:
    ax = train[target].groupby(train[col]).sum().plot(kind='bar', 
                                                                title ="Clicks per " + col, 
                                                                figsize=(10, 5), fontsize=12);
    ax.set_xlabel(col, fontsize=12);
    ax.set_ylabel("Clicks", fontsize=12);
    plt.show();
```

### Dropping some features

Notice that some of the features are massively dominated by **just one level**. We will drop those. We have to
do that for both train and test sets:

```
cols_to_drop = ['location']
train_new = train.drop(cols_to_drop,axis=1)
test_new = test.drop(cols_to_drop,axis=1)
```

<a id = 'dtypes'></a> 
### Data types

```
train_new.dtypes
test_new.dtypes
```

#### Converting some of the integer columns into strings:

```
cols_to_convert = test_new.columns.tolist()
for col in cols_to_convert:
    train_new[col] = train_new[col].astype(str)
    test_new[col] = test_new[col].astype(str)
```


## Handling missing values

The only column with missing values is the `domain` column. There are several ways to fill missing values including:
- Dropping the corresponding rows
- Filling `NaNs` using most the frequent value.
- Using Multiple Imputation by Chained Equations of MICE is a more sophisticated option

In our case, the are only a relatively small percentage of `NaNs` in just one column, namely, $\approx$ 13$\%$ of domain values are missing. I opted for values imputation to avoid dropping rows. Future analysis using MICE should improve final results.

```
train_new['website'] = train_new[['website']].apply(lambda x:x.fillna(x.value_counts().index[0]))  
train_new.isnull().any()
test_new['website'] = test_new[['website']].apply(lambda x:x.fillna(x.value_counts().index[0]))  
test_new.isnull().any()
```

<a id = 'dummies'></a> 
### Dummies

We can transform the categories with low cardinality into dummies using hot encoding:

```
cols_to_keep = ['carrier', 'device', 'day', 'hour', 'dimension']
low_cardin_train = train_new[cols_to_keep]
low_cardin_test = test_new[cols_to_keep]
dummies_train = pd.concat([pd.get_dummies(low_cardin_train[col], drop_first = True, prefix= col) 
                     for col in cols_to_keep], axis=1)
dummies_test = pd.concat([pd.get_dummies(low_cardin_test[col], drop_first = True, prefix= col) 
                     for col in cols_to_keep], axis=1)
dummies_train.head()
dummies_test.head()

train_new.to_csv('train_new.csv')
test_new.to_csv('test_new.csv')
```

#### Concatenating with the rest of the `DataFrame`:

```
train_new = pd.concat([train_new[high_cardinal_train + ['click']], dummies_train], axis = 1)
test_new = pd.concat([test_new[high_cardinal_test], dummies_test], axis = 1)
```

Now, to treat the columns with high cardinality, we will break them up into percentiles based on the number of impressions (number of rows). 

#### Building up dictionaries for creation of dummy variables

```
train_new['count'] = 1   # auxiliar column
test_new['count'] = 1
```

#### In the next cell, I use `pd.cut` to rename column entries using percentiles

```
def series_to_dataframe(s,name,index_list):
    lst = [s.iloc[i] for i in range(s.shape[0])]
    new_df = pd.DataFrame({name: lst})  # transforms list into dataframe
    new_df.index = index_list
    return new_df
def ranges(df1,col):
        df = series_to_dataframe(df1['count'].groupby(df1[col]).sum(),
                             'sum of ads',
                             df1['count'].groupby(df1[col]).sum().index.tolist()).sort_values('sum of ads',ascending=False)
        #print('How the pd.cut looks like:\n')
        #print(pd.get_dummies(pd.cut(df['sum of ads'], 3)).head(3))
        df = pd.concat([df,pd.get_dummies(pd.cut(df['sum of ads'], 3), drop_first = True)],axis=1)
        df.columns = ['sum of ads',col + '_1',col + '_2']
        return df
website_train = ranges(train_new,'website')
publisher_train = ranges(train_new,'publisher')
website_test = ranges(test_new,'website')
publisher_test = ranges(test_new,'publisher')
website_train.reset_index(level=0, inplace=True)
publisher_train.reset_index(level=0, inplace=True)
website_test.reset_index(level=0, inplace=True)
publisher_test.reset_index(level=0, inplace=True)
website_train.columns = ['website', 'sum of impressions', 'website_1', 'website_2']
publisher_train.columns = ['publisher', 'sum of impressions', 'publisher_1', 'publisher_2']
website_test.columns = ['website', 'sum of impressions', 'website_1', 'website_2']
publisher_test.columns = ['publisher', 'sum of impressions', 'publisher_1', 'publisher_2']
train_new = train_new.merge(website_train, how='left')
train_new = train_new.drop('website',axis=1).drop('sum of impressions',axis=1)
train_new = train_new.merge(publisher_train, how='left')
train_new = train_new.drop('publisher',axis=1).drop('sum of impressions',axis=1)
test_new = test_new.merge(website_test, how='left')
test_new = test_new.drop('website',axis=1).drop('sum of impressions',axis=1)
test_new = test_new.merge(publisher_test, how='left')
test_new = test_new.drop('publisher',axis=1).drop('sum of impressions',axis=1)
```

## Imbalanced classes

<a id = 'umb'></a>
#### Imbalanced classes in general

- We can account for unbalanced classes using:
  - Undersampling: randomly sample the majority class, artificially balancing the classes when fitting the model
  - Oversampling: boostrap (sample with replacement) the minority class to balance the classes when fitting the model. We can oversample using the SMOTE algorithm (Synthetic Minority Oversampling Technique) 
- Note that it is crucial that we **evaluate our model on the real data!!**

```
zeros = train_new[train_new['click'] == 0]
ones = train_new[train_new['click'] == 1]
counts = train_new['click'].value_counts()
proportion = counts[1]/counts[0]
train_new = ones.append(zeros.sample(frac=proportion))
#train_new['response'].value_counts()
#train_new.isnull().any()
```

# Models

```
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer, TfidfTransformer
import seaborn as sns
from sklearn.metrics import confusion_matrix
%matplotlib inline

X_test = test_new
```

# Defining ranges for the hyperparameters to be scanned by the grid search
```
n_estimators = list(range(20,120,10))
max_depth = list(range(2, 22, 2)) + [None]
def random_forest_score(df,target_col,test_size,n_estimators,max_depth):
    
    X_train = df.drop(target_col, axis=1)   # predictors
    y_train = df[target_col]                # target
    X_test = test_new
    
    rf_params = {
             'n_estimators':n_estimators,
             'max_depth':max_depth}   # parameters for grid search
    rf_gs = GridSearchCV(RandomForestClassifier(), rf_params, cv=5, verbose=1, n_jobs=-1)
    rf_gs.fit(X_train,y_train) # training the random forest with all possible parameters
    print('The best parameters on the training data are:\n',rf_gs.best_params_) # printing the best parameters
    max_depth_best = rf_gs.best_params_['max_depth']      # getting the best max_depth
    n_estimators_best = rf_gs.best_params_['n_estimators']  # getting the best n_estimators
    print("best max_depth:",max_depth_best)
    print("best n_estimators:",n_estimators_best)
    best_rf_gs = RandomForestClassifier(max_depth=max_depth_best,n_estimators=n_estimators_best) # instantiate the best model
    best_rf_gs.fit(X_train,y_train)  # fitting the best model
    preds = best_rf_gs.predict(X_test)
    feature_importances = pd.Series(best_rf_gs.feature_importances_, index=X_train.columns).sort_values().tail(5)
    print(feature_importances.plot(kind="barh", figsize=(6,6)))
    return 

random_forest_score(train_new,'click',0.3,n_estimators,max_depth)
```
```
X = train_new.drop('click', axis=1)   # predictors
y = train_new['click']  

def cv_score(X,y,cv,n_estimators,max_depth):
    rf = RandomForestClassifier(n_estimators=n_estimators_best,
                                max_depth=max_depth_best)
    s = cross_val_score(rf, X, y, cv=cv, n_jobs=-1)
    return("{} Score is :{:0.3} ± {:0.3}".format("Random Forest", s.mean().round(3), s.std().round(3)))

dict_best = {'max_depth': 14, 'n_estimators': 80}
n_estimators_best = dict_best['n_estimators']
max_depth_best = dict_best['max_depth']
cv_score(X,y,5,n_estimators_best,max_depth_best)

n_estimators = list(range(20,120,10))
max_depth = list(range(2, 16, 2)) + [None]

def random_forest_score_probas(df,target_col,test_size,n_estimators,max_depth):
    
    X_train = df.drop(target_col, axis=1)   # predictors
    y_train = df[target_col]                # target
    X_test = test_new
    
    rf_params = {
             'n_estimators':n_estimators,
             'max_depth':max_depth}   # parameters for grid search
    rf_gs = GridSearchCV(RandomForestClassifier(), rf_params, cv=5, n_jobs=-1)
    rf_gs.fit(X_train,y_train) # training the random forest with all possible parameters
    max_depth_best = rf_gs.best_params_['max_depth']      # getting the best max_depth
    n_estimators_best = rf_gs.best_params_['n_estimators']  # getting the best n_estimators
    best_rf_gs = RandomForestClassifier(max_depth=max_depth_best,n_estimators=n_estimators_best) # instantiate the best model
    best_rf_gs.fit(X_train,y_train)  # fitting the best model
    preds = best_rf_gs.predict(X_test)
    prob_list = [prob[0] for prob in best_rf_gs.predict_proba(X_test).tolist()]
    df_prob = pd.DataFrame(np.array(prob_list).reshape(53333,1))
    df_prob.columns = ['probabilities']
    df_prob.to_csv('probs.csv')
    return df_prob

random_forest_score_probas(train_new,'click',0.3,n_estimators,max_depth).head()

def random_forest_score_preds(df,target_col,test_size,n_estimators,max_depth):
    
    X_train = df.drop(target_col, axis=1)   # predictors
    y_train = df[target_col]                # target
    X_test = test_new
    
    rf_params = {
             'n_estimators':n_estimators,
             'max_depth':max_depth}   # parameters for grid search
    rf_gs = GridSearchCV(RandomForestClassifier(), rf_params, cv=5, verbose=1, n_jobs=-1)
    rf_gs.fit(X_train,y_train) # training the random forest with all possible parameters
    max_depth_best = rf_gs.best_params_['max_depth']      # getting the best max_depth
    n_estimators_best = rf_gs.best_params_['n_estimators']  # getting the best n_estimators
    best_rf_gs = RandomForestClassifier(max_depth=max_depth_best,n_estimators=n_estimators_best) # instantiate the best model
    best_rf_gs.fit(X_train,y_train)  # fitting the best model
    preds = best_rf_gs.predict(X_test)
    df_pred = pd.DataFrame(np.array(preds).reshape(53333,1))
    df_pred.columns = ['predictions']
    df_pred.to_csv('preds.csv')
    return df_pred

random_forest_score_preds(train_new,'click',0.3,n_estimators,max_depth)
```
