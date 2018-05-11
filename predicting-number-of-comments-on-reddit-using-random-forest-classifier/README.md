# Predicting Comments on Reddit 


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![image title](https://img.shields.io/badge/python-v3.6-green.svg) ![image title](https://img.shields.io/badge/ntlk-v3.2.5-yellow.svg) ![Image title](https://img.shields.io/badge/sklearn-0.19.1-orange.svg) ![Image title](https://img.shields.io/badge/BeautifulSoup-4.6.0-blue.svg) ![Image title](https://img.shields.io/badge/pandas-0.22.0-red.svg) ![Image title](https://img.shields.io/badge/numpy-1.14.2-green.svg) ![Image title](https://img.shields.io/badge/matplotlib-v2.1.2-orange.svg)


<br>
<br>
<p align="center">
  <img src="https://github.com/marcotav/predicting-the-number-of-comments-on-reddit/blob/master/Reddit-logo.png" 
       width="150" height="150">
</p>
<br>

<p align="center">
  <a href="#ps"> Problem Statement </a> •
  <a href="#steps"> Steps </a> •
  <a href="#webscraping"> Bird's-eye view of webscraping  </a> •
  <a href="#writingfunctions"> Writing functions to extract data from Reddit </a> •
  <a href="#nlp"> Quick review of NLP techniques </a> •
  <a href="#preprocess"> Preprocessing the text </a> •
  <a href="#models">Models </a> 
</p>

<a id = 'ps'></a>
## Problem Statement

Determine which characteristics of a post on Reddit contribute most to the overall interaction as measured by number of comments.

<a id = 'steps'></a>
## Steps

This project had three steps:
- Collecting data by scraping a website using the Python package `requests` and using the Python library `BeautifulSoup` which efficiently extracts HTML code. We scraped the 'hot' threads as listed on the <br> [Reddit homepage](https://www.reddit.com/) (see figure below) and acquired the following pieces of information about each thread:

   - The title of the thread
   - The subreddit that the thread corresponds to
   - The length of time it has been up on Reddit
   - The number of comments on the thread
  
  <br>
<br>
<p align="center">
  <img src="https://github.com/marcotav/predicting-the-number-of-comments-on-reddit/blob/master/redditpage.png" 
       width="750">
</p>
<br>

- Using Natural Language Processing (NLP) techniques to preprocess the data. NLP, in a nutshell, is "how to transform text data and convert it to features that enable us to build models." NLP techniques include:

   - Tokenization: essentially splitting text into pieces based on given patterns
   - Removing stopwords  
   - Lemmatization: returns the word's *lemma* (its base/dictionary form)
   - Stemming: returns the base form of the word (it is usually cruder than lemmatization).

- After the step above we obtain *numerical* features which allow for algebraic computations. We then build a `RandomForestClassifier` and use it to classify each post according to the corresponding number of comments associated with it. More concretely the model predicts whether or not a given Reddit post will have above or below the _median_ number of comments.

<a id = 'webscraping'></a>  
### Bird's-eye view of webscraping 

The general strategy is:
- Use the `requests` Python packages to make a `.get` request (the object `res` is a `Response` object):   
```
res = requests.get(URL,headers={"user-agent":'mt'})
```     
- Create a BeautifulSoup object from the HTML
```
soup = BeautifulSoup(res.content,"lxml")
```
- Use `.extract` to see the page structure:
```
soup.extract
```
<a id = 'writingfunctions'></a>  
### Writing functions to extract data from Reddit
Here I write down the the functions that will extract the information needed. The structure of the functions depends on the HTML code of the page. The page has the following structure:
- The thread title is within an `<a>` tag with the attribute `data-event-action="title"`.
- The time since the thread was created is within a `<time>` tag with attribute `class="live-timestamp"`.
- The subreddit is within an `<a>` tag with the attribute `class="subreddit hover may-blank"`.
- The number of comments is within an `<a>` tag with the attribute `data-event-action="comments"`.

The functions are:
```
def extract_title_from_result(result,num=25):
    titles = []
    title = result.find_all('a', {'data-event-action':'title'})
    for i in title:
        titles.append(i.text)
    return titles

def extract_time_from_result(result,num=25):
    times = []
    time = result.find_all('time', {'class':'live-timestamp'})
    for i in time:
        times.append(i.text)
    return times

def extract_subreddit_from_result(result,num=25):
    subreddits = []
    subreddit = result.find_all('a', {'class':'subreddit hover may-blank'})
    for i in subreddit:
        subreddits.append(i.string)
    return subreddits

def extract_num_from_result(result,num=25):
    nums_lst = []
    nums = result.find_all('a', {'data-event-action': 'comments'})
    for i in nums:
        nums_lst.append(i.string)
    return nums_lst
```
 I then write a function that finds the last `id` on the page, and stores it:
 ```
def get_urls(n=25):
    j=0   # counting loops
    titles = []
    times = []
    subreddits = []
    nums = []
    URLS = []
    URL = "http://www.reddit.com"
    
    for _ in range(n):
        
        res = requests.get(URL, headers={"user-agent":'mt'})
        soup = BeautifulSoup(res.content,"lxml")
        
        titles.extend(extract_title_from_result(soup))
        times.extend(extract_time_from_result(soup))
        subreddits.extend(extract_subreddit_from_result(soup))
        nums.extend(extract_num_from_result(soup))         

        URL = soup.find('span',{'class':'next-button'}).find('a')['href']
        URLS.append(URL)
        j+=1
        print(j)
        time.sleep(3)
        
    return titles, times, subreddits, nums, URLS
 ```

I then build a pandas `DataFrame`, perform some exploratory data analysis and create:
- A binary column that classifies the number of comments comparing the values with their median
- A set of dummy columns for the subreddits
- Concatenate both

```
df['binary'] = df['nums'].apply(lambda x: 1 if x >= np.median(df['nums']) else 0)
# dummies created and dataframes concatenated
df_subred = pd.concat([df['binary'],pd.get_dummies(df['subreddits'], drop_first = True)], axis = 1)
```
<a id = 'nlp'></a>  
### Quick review of NLP techniques
Before applying NLP to our problem, I will provide a quick review of the basic procedures using `Python`. We use the package `nltk` (Natural Language Toolkit) to perform the actions above. The general procedure is the following. We first import `nltk` and the necessary classes for lemmatization and stemming
```
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
```
We then create objects of the classes `PorterStemmer` and `WordNetLemmatizer`: 
```
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
```
To use lemmatization and/or stemming in a given string `text` we must first tokenize it. To do that, we use `RegexpTokenizer` where the argument below is a regular expression. 
```
tokenizer = RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(text)
tokens_lemma = [lemmatizer.lemmatize(i) for i in tokens]
stem_text = [PorterStemmer().stem(i) for i in tokens]
```
<a id = 'preprocess'></a>  
### Preprocessing the text
To preprocess the text, before creating numerical features from it, I used the following `cleaner` function:
```
def cleaner(text):
    stemmer = PorterStemmer()                                          
    stop = stopwords.words('english')    
    text = text.translate(str.maketrans('', '', string.punctuation))   
    text = text.translate(str.maketrans('', '', string.digits))        
    text = text.lower().strip() 
    final_text = []
    for w in text.split():
        if w not in stop:
            final_text.append(stemmer.stem(w.strip()))
    return ' '.join(final_text)
```
I then use `CountVectorizer` to create features based on the words in the thread titles. `CountVectorizer` is scikit-learn's bag of words tool. I then combine this new table `df_all` and the subreddits features table and build a model.

```
cvt = CountVectorizer(min_df=min_df, preprocessor=cleaner)
cvt.fit(df["titles"])
cvt.transform(df['titles']).todense()
X_title = cvt.fit_transform(df["titles"])
X_thread = pd.DataFrame(X_title.todense(), 
                        columns=cvt.get_feature_names())
df_all = pd.concat([df_subred,X_thread],axis=1)                     
```

<img src="https://github.com/marcotav/predicting-the-number-of-comments-on-reddit/blob/master/redditwordshist.png" width="400">


<a id = 'models'></a>  
### Models
Finally, now with the data properly treated, we use the following function to fit the training data using a `RandomForestClassifier` with optimized hyperparameters obtained using `GridSearchCV`. The range of hyperparameters is:
```
n_estimators = list(range(20,220,10))
max_depth = list(range(2, 22, 2)) + [None]
```

The following function does the following:
- Defines target and predictors
- Performs a train-test split of the data
- Uses `GridSearchCV` which performs an "exhaustive search over specified parameter values for an estimator" (see the docs). It searches the hyperparameter space to find the highest cross validation score. It has several important arguments namely:

| Argument | Description |
| --- | ---|
| **`estimator`** | Sklearn instance of the model to fit on |
| **`param_grid`** | A dictionary where keys are hyperparameters and values are lists of values to test |
| **`cv`** | Number of internal cross-validation folds to run for each set of hyperparameters |

- After fitting, `GridSearchCV` provides information such as:

| Property | Use |
| --- | ---|
| **`results.param_grid`** | Parameters searched over. |
| **`results.best_score_`** | Best mean cross-validated score.|
| **`results.best_estimator_`** | Reference to model with best score. |
| **`results.best_params_`** | Parameters found to perform with the best score. |
| **`results.grid_scores_`** | Display score attributes with corresponding parameters. | 

- The estimator chosen here was a `RandomForestClassifier`. The latter fits a set of decision tree classifiers on sub-samples of the data, averaging to improve the accuracy and avoid over-fitting. 
- Fits several models using the training data, for all parameters within the parameter grid `rf_params` and find the best model i.e. the model with best mean cross-validated score.
- Instantiates the best model and fits it
- Scores the model and makes predictions
- Determines the most relevant features and prints out a bar plot showing them.

```
def rfscore(df,target_col,test_size,n_estimators,max_depth):
    
    X = df.drop(target_col, axis=1)   # predictors
    y = df[target_col]                # target
    
    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, test_size = test_size, random_state=42)
    # definition of a grid of parameter values
    rf_params = {
             'n_estimators':n_estimators,
             'max_depth':max_depth}   # parameters for grid search
             
    # Instantiation       
    rf_gs = GridSearchCV(RandomForestClassifier(), rf_params, cv=5, verbose=1, n_jobs=-1)
    
    # fitting using training data with all possible parameters
    rf_gs.fit(X_train,y_train) 
    
    # Parameters that have been found to perform with the best score
    max_depth_best = rf_gs.best_params_['max_depth']      
    n_estimators_best = rf_gs.best_params_['n_estimators'] 
    
    # Best model
    best_rf_gs = RandomForestClassifier(max_depth=max_depth_best,n_estimators=n_estimators_best) 
    
    # fitting best model using training data with all possible parameters
    best_rf_gs.fit(X_train,y_train)  
    
    # scoring
    best_rf_score = best_rf_gs.score(X_test,y_test) 
    
    # predictions
    preds = best_rf_gs.predict(X_test)
    
    # finds the most important features and plots a bar chart
    feature_importances = pd.Series(best_rf_gs.feature_importances_, index=X.columns).sort_values().tail(5)
    print(feature_importances.plot(kind="barh", figsize=(6,6)))
    return 
```
The function below that performs cross-validation, to obtain the accuracy score for the model with best parameters obtained from the `GridSearch`:

```
def cv_score(X,y,cv,n_estimators,max_depth):
    rf = RandomForestClassifier(n_estimators=n_estimators_best,
                                max_depth=max_depth_best)
    s = cross_val_score(rf, X, y, cv=cv, n_jobs=-1)
    return("{} Score is :{:0.3} ± {:0.3}".format("Random Forest", s.mean().round(3), s.std().round(3)))
```
The most important features according to the `RandomForestClassifier` are shown in the graph below:
<br>

   <img src="https://github.com/marcotav/predicting-the-number-of-comments-on-reddit/blob/master/redditRF.png" width="400">

