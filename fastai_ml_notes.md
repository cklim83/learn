### Lecture 1:
#### Administrative
    - crestle and paperspace both have cloud based jupyter environment with
    fastai library. Alternatively, one can create local env from github

#### Jupyter Notebook Functions
~~~ python
# Allows code modifications to imported libraries to take effect without reloading
%load_ext autoreload
%autoreload 2
~~~
- Fn + shift + enter to show source of a function
- ?function_name to display document
- ??function_name to show source code
- ! allows bash commands to run in jupyter notebook.
~~~ python
!ls {PATH} # notebook will expand PATH variable before passing to shell for ls
~~~

#### Download Kaggle Data
- Navigate to site in firefox, press ctrl+shift+i to open browser network tab
- select download tab to show network connections initiated
- right click and copy as curl (unit command that download stuffs)
- open terminal, paste curl command, remove --2.0 embedded in curl command,
append -o filename.zip to save output as zip file, else curl will print to
stdout

#### Steps in Machine Learning
- Step 1: Data Exploration. Look at data sufficiently to know their type,
missing values etc.
~~~ python
!head data/bulldozers/Train.csv # look at start of data in jupyter using bash

# View last 5 entries, tranpose columns to rows as column nums are big
df.tail().transpose()
~~~
- Step 2: Determine evaluation metric. For price prediction, use RMSLE if
evaluation metric not given. RMSLE is relative measure as gives us percent
error rather than absolute error (scale dependent)

#### Why Random Forest
- Suitable for both regression and classification
- Robust and hard to overfit
- Provide useful OOB score
- No statistical assumptions.
    - Data need not be normally distributed
    - Doesnt assume linear relationship
    - Doesnt need to specify interaction terms

#### Curve of Dimensionality
- Theory postulates that when feature count is too high, points in sparse
matrix is more probable to sit on edge of high dimensional space, making
distance computation between points meaningless and things do not work
- In practice, relative distance computation still works

#### No Free Lunch Theory
- No algorithm work well on all datasets
- This is true in theory, since for any random data, there should be some
algorithm that can fit it better.
- In practice, data are not random but have cause effect relationships.
- Ensemble methods based on decision trees work well most times.

#### Data Preparation for Modelling
- Need to convert all non-number fields to numbers, Random Forest cannot work
on strings
- Feature engineer dates using add_date_part method in fasti. Will generate
features such as day_of_week, day_of_month, year, month, weekend automatically.
 Can also include is holiday/event, days to/from event.
- Engineer as much feature as one can think of
- Categorical data
    - Use train_cats to convert strings to category codes. Set it for train
    set, and use apply_cats to apply to valid/test sets to ensure mapping
    consistency
    - train_cats will give missing values a code of 0 and others in running
    order.
    - Actually orders in ordinal categorical columns is not important. Even in
     wrong order, Trees can still extract specific categories, but maybe in
     more splits. e.g. 'Low' can be extracted from ['High', 'Low', 'Medium']
     via 2 decision splits, i.e. > 'High', then less then 'Medium' vs one step
      < 'Medium' if properly ordered. Splitting efficiency is just better when
      ordered.
~~~ python
# To Reorder
df[col].cat.set_categories([correct order],
      ordered=True, inplace=True)
~~~
- Missing Values
    - df.isnull().sum().sort_index()/len(df) will give % missing values for
    each column

- Saving to Feather
    - use df.to_feather(path) for fast saving and subsequent data loading.
- proc_df function in fastai
    - make a copy of df
    - extract y
    - fix missing in X
        - if column is numeric and had null values, create colname_na column
        and set rows with missing values to 1. This allows one to track which
        rows were filled. Can then replace missing values with median.
        - if column not numeric, we call numericalize to replace it with code
        +1 so that missing values have value of 0.
- Random Forest is easily parallizable by seting RF(n_jobs=-1) to use all
available CPU cores
- RF.score gives the R2.

Cross Validatioon
- Sort data by data, create validation set as similar as test size.


### Lecture 2
#### Administrative
- Symbolic Link is like a shortcut to get to a path directly
~~~ bash
ln -s src_dir dest_dir
# src_dir could be ../../fastai
dest_dir could be ./
~~~

#### R-Squared
- Random forest score function returns the R-squared
- R-squared indicates the proportion of variance explained by a model
- Ranges from -infinity to 1
- R^2 = 1 - (SS_resi/SS_total)
    - SS_total = Sum_over_i(y_i - y_mean)^2 i.e. error predicting using mean
    - SS_resi = Sum_over_i(y_i - y_pred_i)^2 i.e. error of model prediction
    - if y_pred is only as good as naive mean, R^2 = 1-1 = 0
    - if y_pred is poorer than mean, R^2 is < 0
    - if y_pred is perfect, R^2 = 1


#### Validation
- Validation set should be created such that performance on validation set will
 tell us how well model will perform on test set
- Validation set must be representative of test data.
- If data has time element, validation and test set should be time ordered
rather than random shuffle

#### Sampling for Iteractive Prototyping
- Use %time when fitting model to determine train time
    - CPU time: total CPU time
    - Wall time: actual elapsed time after parallelizing on multiple cores
- Should endeavor to keep Wall Time < 10s to enable iteractive prototyping
- If dataset is too big, we prototype using sample subset.
~~~ python
df_train, y_train = proc_df(df_raw, 'SalePrice', subset=30k) # 30k sample
X_train, _ = split_vals(df_train, 20k) # first 20k in time order
y_train, _ = split_vals(y_train, 20k)

# reset_rf_samples() to turn off sampling when we want to use full data for
# final model
~~~
- When training on sample, we cannot set OOB_Score on as it will test on ALL
untrained samples, which will be too long for large dataset.
- With sampling, no dataset is too big. Having 20-50k samples is generally good
enough for most problems.
- Insights/Feature Importance using data subsets and smaller number of trees
should still be similar but faster.

#### Building 1 Tree
~~~ python
# 1 small tree sample w/o replacement
RF(n_estimators=1, max_depth=3, bootstrap=False)
~~~
- Simple but poor model. But easy to visualize
- Simple Decision Tree Algorithm
    - At each split, iterate over all features, iterate over all possible split
     values (i.e. unique values) such that reduction in prediction errors of
     children is greatest
    - Each split only need to be binary as we can always split again
    - Repeat split until either
        - max depth reached OR
        - leaf node only have min_sample_leaf observations or 1 sample OR
        - errors of children is higher than parent (not true for XGBoost, which
         may still go a few levels lower to see if there is overall resultant
         improvement)
        - Note at its extreme, each leaf has only 1 sample. Train R^2 will be 1
         but valid R^2 will be bad due to overfitting.
- Single tree will not produce good result -> Multiple trees used in bagging
to build a forest to improve performance

#### Bagging
- Create multiple trees, each somewhat predictive
- Each tree is built using different subset of samples and features
- As different data are used to build each tree, they find different insights
and generate uncorrelated errors.
- Combine these trees in an ensemble and average their predictions. Since
errors are random and uncorrelated, averaging them should reduce overall error

#### Objective of Random Forest
- Each tree to be as predictive as possible
- Trees to be as uncorrelated to each other as possible (bagging)
- Recent research found having uncorrelated trees to be more important in
driving overall result compared to having very accurate individual trees.
    - Advent of extra tree regressor/classifier. Rather than trying every
    possible splits for every variable, it randomly tries a few splits of a few
    variables
    - Much faster due to less greediness in search and more randomness
    - However, each tree is more crappy and larger number of trees is needed
- Adding n_estimators improve predictions up to a certain point but slows
things down.

#### OOB Score
- Set oob_score=True
- useful as another metric besides validation score to determine overfitting
- Calculation: for each row, select trees not trained on that row for
prediction. Average those predictions and average errors across rows to get
OOB_score.
- Should R^2 for OOB be smaller than R^2 for validation?
    - Not necessarily but generally is so because OOB predictions are generated
     with less trees than the full forest. As number trees increase, accuracy
     tends to increase. Hence validation scores tends to be better as it uses
     the entire forest for prediction. However, if validation data distribution
      is different from train set, validation R^2 could still be poorer.

#### Hyperparameters
- min_sample_leaf
    - Try [1, 3, 5, 10, 25]
    - Try bigger values if dataset is bigger
- max_features
    - try [0.5, 'sqrt', 'log2']
    - if there is a very strong predictors, all trees will choose that
    predictor -> trees will be highly correlated, bagging effect drops
    - By allowing only a random subset of features to be used at each split,
    there is increased chance to select combinations of features that
    individually are not as strong, but collectively through interaction are
    more important than most important individual predictor
- min_sample_leaf vs max_depth
    - Personal thought: both are similar in they control the amount of split a
    tree can do, i.e. control model flexibility. However, min_sample_leaf may
    offer finer control as we can have trees with different depth but all
    having leaves of similar sizes.


### Lecture 3
#### Choice of Algorithms
- When should we try algorithms other than Random Forest?
    - Unstructured data (e.g. pictures in image recognition). We want to use
    neural nets.
    - Collaborative Filtering problems in recommender systems

#### Proc_df Function in Fastai Library
- For columns with missing values, proc_df automatically add a colname_na
column to mark rows with missing value before filling those rows with median
- It turns categorical columns to codes, with 0 reserved for missing values
- Possible other problems:
    - Test set have missing values in columns that have no missing values in
    the training set. This will lead to error
    - Median in test set may be different to train set
- Solution:
     - proc_df accepts nas, a dict where keys are names of columns with
     missing values and the value is the median fill value used. proc_df in
     turn also returns nas, an updated dict for other columns whose missing
     values are also treated.
     - In this way, proc_df can generate a nas containing keys of
     columns with missing values and their filled median values, and pass them
     to the test set so allow missing values in test set to be processed
     constantly.
     - If test set have new missing columns, their treatment will be updated
     in the returned nas as well.

#### Grocery Prediction
- Predict unit sale for each category of products for each store for each day
over 2 weeks
- Data available:
    - Unit sales for each category of product for each store for each day over
    past 4 years
    - Some meta data such as oil prices, product categories in star schema
    arrangement where central transactional dataset can be joined with these
    metadata
- Working with large dataset (125M records)
    - To accompany large dataset in Pandas, we create a dictionary and specify
    the smallest datatype for each column
    - Use **shuf** at command prompt to get random samples of data to know data
    range and smallest type to use
    - Use samples to prototype first. There is no relationship between size of
    dataset and time to build forest. Relationship is between number of
    estimators and sample size.
    ~~~ python
    set_rf_sample(20k)
    ~~~

- Train set: 4 years or less of historical data
- Test set: 2 weeks
- When predicting quantity, clip y to minimum of 0, then take np.log1p i.e. log
 (1+y) as minimum of y is 0
~~~ python
np.clip(y,0,None)
# minimum is 0
# no maximum, hence None
~~~
- If data not all numeric, run the following to convert category to code:
~~~ python
train_cats(raw_train)
apply_cats(raw_valid, raw_train)
~~~
- use %prun to profile runs to see which code component takes longer to run
    - For example, notice that rf.fit(X,y) always convert X to np.array of
    floats before building trees and this takes 1 min 37s.
    - As we will call fit multiple times, we can convert X to np.array of
    floats once so that all subsequent fit calls can skip this step so that
    overhead is only incurred once.
- When using set_rf_sample(subset)
    - We cannot use OOB_score as it will by default call all non-training rows
    as validation set. We should use an explict validation set instead.

#### Calibrating Validation Set
- Insert picture show good vs poor validation set

#### Prediction Uncertainty
- We should know the confidence of any forest prediction
- For regression, a proxy is the standard deviation of predictions by trees in
the forest. For classification, it is the probability of a class (values close
to 0.5 implies uncertainty for binary classification)
- If some observations are raw, most trees will not have the opportunity to
construct a well defined path for these types of observations.
- During prediction, these observations will likely end up in very different
part of each trees, leading to widely differing predictions and a large
standard deviation.
- To see the standard deviation of predictions we could use the following:
~~~ python
preds = np.stack([t.predict(X_valid) for t in m.estimators_]) # in series
np.mean([preds[:,0]), np.std(preds[:,0]) # prediction and std_dev

# Fastai provides parallel_trees class to predict in parallel
def get_preds(t)
    return t.predict(X_valid)
%time preds = np.stack(parallel_trees(m, get_preds))
~~~
- We could also compute std_dev/prediction for regression problems as a
normalized ratio of prediction uncertainty
- Application: e.g. Loans. More uncertainty of repayment, smaller loan quantum.

#### Feature Importance
- This is the model important part of modelling. The first thing in modelling
is to get a quick sense of the most important features,visualize then discuss
with domain experts to understand why.
- We could often have co-linear predictors in the model. It doesn't necessarily
degrade model performance, but affects interpretability as weights are split
amongst them
- To solve this, we try to identify colinear feature groups (e.g. linear/rank
correlation), and remove them in sequence to see if the model degrades
significantly. If not, these features can be omitted to make the model simpler.
If there are colinear groups, we permutate within that group and see to retain
only one feature from each group as far as possible.

#### Random Forest Feature Importance vs Permutation Importance
- Feature importance implementation in Random Forest has a bias towards
favoring features of high cardinality as the higher split options they provide
increases their chance of a split point that could generate purer descendents.
As a result, feature importance by Random Forest might churn out features that
does not make intuitive sense.
- Empirical studies have shown that permutation importance offers an efficient
alternative approach to identify important features
    - Model is train on all features. Performance on validation set is set as
    baseline.
    - Iterate through each feature, randomly shuffle values in that feature to
    see the degradation in validation performance. Ideally, these shuffling and
     test of validation degradation should be repeated multiple times for each
     feature to ensure relative rank is stable and repeatability. eli5 library
     has this built-in.
    - If the shuffling of a feature results in great deterioration in
    validation accuracy, it is an important feature and there is no good
    substitutes amongst other feature
    - If the shuffling of a feature does not worsen/even improves the
    validation performance, the feature is either unimportant or has suitable
    replacement (i.e. colinear features) such that its loss is not felt and may
     be removed.
    - Features are then rank by the degradation in validation performance, the
    bigger the degradation, the higher its importance.
    - Features with positive values imply shuffling the feature results in
    better model performance.
    - Permutation importance is more efficient because model is only train once
     with all features and we just shuffle the feature under test when
     predicting on validation set. If we want to remove feature under test, we
     have to retrain the model for each feature omitted, which is very time
     consuming.

#### Visualization of Important Features vs Dependent Variable
- Look for relationship between top features and dependent variable. Check if
there are noise or weird encodings we need to fix for these features.


### Lecture 4:
#### Administrative
- Jupyter notebooks are not suitable for version control
- Every time we pull a notebook, we could save a copy and rename by prefixing a
 tmp_ to filename. This will hide the file from git version control. Subsequent
 pull will then not conflict with this file.

#### Hyperparameter Tuning
- set_rf_sample [20k - 100k]
    - This will determine the number of rows used to create each tree
    - Assuming balance tree, 20k rows will have depth = log2 (20k)
    - Hence smaller sample size, less variety in leaf nodes resulting in
    smaller variety of possible predictions
    - Less overfit but less flexible model
- min_sample_leaf [1, 3, 5, 10, 25, 100]
    - if min_sample_leaf=2, max depth = log2(20k) -1 => 10k leaf_nodes,
    assuming 20k samples.
    - as min_sample_leaf increases, we have less leaf nodes, hence each
    estimator has less prediction variety, less prone to overfit but also less
    flexible.
    - training speed with shorten as less split decisions are required.
- max_features [0.5, sqrt, log2, None] (None means use all features)
    - number of features to consider for each split point
    - smaller max_features -> each trees are less correlated but individual
    trees are less accurate, potentially needed more trees.
    - Taken to extreme, we could build each tree with just 1 feature. However,
    this will miss out on feature interaction (e.g. spliting on both year_made
    + sales_date = age of equivalent)
    - Hypothetically, if we know ahead of time the most important features +
    interaction terms, we could use logistical regression and get as good
    results as random forest. Problem is this is never possible in real time.

#### Colinear Features
- Features linearly related to each other
- Random Forest will underestimate their importance due to weight sharing and
compensating behaviors

#### Keeping Important Feature
- Visualizing bar plot of feature importance, cut-off when weight of less
importance features are almost equal
- Even better if when these less important features are omitted, model doesnt
degrade much. If they do, incrementally include the omitted to see when needs
to be retained.
- After removal of least important features, the feature importance weight may
be re-distributed and the relative importance might change.

#### Problem: Poorer Validation Score After Introducing Engineered Feature
- Two possibilities:
    - Overfit to training set. If so, OOB will be also be worse off than train
    score.
    - validation set is not a random sample (e.g. it is from a distinct time
    period). Data that was true in the train set might not be so in the
    validation set.

#### Assumption of a Linear Regression Model
- General form: Y = b0 + b1X1+ b2X2 + ... + bnXn
- the coefficients of features are their importance score
- While theoretically simple to formulate, this form failed to capture
interaction terms, and non-linear effects. Since the form is wrong, the
coefficient and relative importance are also wrong. Truth is nobody knows the
true relationship.
- Even if we assume the relationship is somewhat know, there might be a need to
 transform the features such that interpretation becomes difficult. 1 unit
 increase in X1 no longer meant b unit increase in Y.
- Random Forest makes no such linear assumption and are able to infer
interactions through sequence splitting. Also feature importance is based on
impact on validation accuracy.

#### One-Hot Encoding for Categorical Variables
- We can elevate levels in categorical variable to be a feature using one-hot
encoding, provided the cardinality of that categorical variable is low. fastai
have a max_category parameter to enforce this.
- Doing so may/may not improve overall model performance, but it could help
identify if any levels are important features, allowing us to understand
drivers better.
- One-hot encoding also help make it easy to sieve out important levels, one
step vs possible multiple splits if based on order in categorical variable.
- In the problem example, one-hot encoding generated a slightly worse off model

#### Finding Similar Features (Agglomerative Clustering)
- Insert Picture of Rank Ordering for Spearman Rank Correlation
- Spearman rank correlation measures the strength and direction of monotonic
relationship between 2 variables for ordinal/ratio/interval data
- Pearson correlation measures the strength of linear relationship, which is
more restrictive that rank correlation.
- Insert picture of clustering chart
- The earlier two features join together, the more correlated they are rank
wise. We can identity feature groups this way and see if we could just choose 1
 feature from each group in the final model as long as model doesnt degrade
 significantly to have a simpler model.

#### Partial Dependence Plot
- Univariate Plot
    - Get samples before plotting with ggplot as there is no point in plotting
    millions of points that mostly overlap. Use get_sampler.
    - Can use ggplot (good visualization package from R) + stat_smooth to add a
    linear regression line. Use method='loess' for localized linear regression
    by data range + std dev to see standard error range across variable.
    - Problem of univariate plot is the effect on y is often not soley
    attributable to the change in the X included in the same plot but due to
    other factors not visualized.
- To overcome limitation of univariate plot, we can use partial dependence plot
to visualize how the most important features varies with the dependent
variable.
- Partial dependence plot isolates the feature under study by keeping all other
variables constant, vary the single feature and use the model to generate the
predicted y to see the corresponding change in y.
- One problem with partial dependence plot is it assumes the data combinations
generated by varying feature under study when keeping other features constant
are feasible in real life (e.g. a aircon excavator in 1900). However, the time
invariant assumption of this method is a limitation we have to be mindful.
- Besides plotting the average line, we could also use plot_pdp(cluster=X) to
group similar lines into various groups to see the various distinct behaviors.
~~~ python
plot_pdp(['Enclosure_EROPS_w_AC' ...'], num_cluster)
~~~
- We could also do pdp_interaction plot to show how two features interact to
affect y.
~~~ python
feats = [salesElapsed, YearMade]
p = pdp.pdp_interact(model, num, feats)
~~~

#### Treating Data Post Exploration
- From data exploration, we can derived feature age = salesYear - year_made
- Set year_made < 1950, which are missing fields to 1950
- After retraining model, age turns out to be most important feature although
model accuracy deteriorate somewhat.

#### Tree Interpreter
- This library allows use to decompose a prediction into bias (prediction at
tree root), and contributions by each feature for a single row.
- For single tree, prediction = bias + contribution from split by feature_X
- For forest, it is the weighted contribution by feature_X across all trees,
ignoring the split values.


### Lecture 5: Performing Good Validation
- Key concern of modelling is generalization error
- OOB score is computed from samples not used to fit the trees
    - On average OOB score should be less good than validation score since it
    uses less tree for prediction. Validation score could be worse if it is a
    disjoint set separated on time and there is behavior change over time or
    data distribution is different from train set.
- Why do we need a validation set when there is OOB score?
    - Validation set should often be time disjoint
- Why random validation using train test split doesnt work?
    - In most actual prediction systems in production, there is a systematic
    error in-built to use old data to predict future unseen data
    - Using randomness to construct validation set means some newer data
    closer in characteristic to validation set could be present and used in
    training. This is not replicable in production, hence modelling accuracy
    could be overstated.
    - In practice, we order observations by time, and use most recent
    data for validation/test
- If model performs similarly on train and OOB, there is no overfitting on
statistical sense. If it perform poorly on validation, odds are something has
occurred in validation data that was different from the past.
- How to build a good validation set
    - Determine the prediction horizon of test set. This is a function of how
    frequently a model can be refreshed in production.
    - A validation set should be as similar to test as possible. Can try one of
     the following:
        - random
        - last month
        - last 2 week
        - same day range 1 month ago
        - same number of weekends
    - Calibrate and select best validation set
        - We construct 5 models of varying accuracy
            - simple average
            - train on full train set
            - train on most recent 2 years
            - Linear model
            - RF model
         - We want the validation set such the validation and test scores of
         the 5 models exhibit a montonical relationship. i.e. if valid score is
          better, test score is also better. If we can achieve this, we can be
          confident that once a model performs better on validation, we should
          also see a better test performance. Otherwise, we cannot be sure.
          Insert picture of good validation set.

- Cross Validation (CV)
    - Benefits
        - uses the entire dataset
        - Can get different models, each with different train data, and build
        an assemble.
    - Disadvantages
        - Takes too long as multiple forests are built
        - Validation set are randomly constructed

#### Random Forest Extrapolation Problem (0:50:00)
- Mitigation measures include:
    - Avoid time based predictors
        - Add one more isValid column, set to False for train observations and
        true for validation observations.
        - Train classifier on features to predict isValid column.
        - If classifier can predict isValid well, we do not have random sample
        - Top features (e.g. salesID, salesElapsed, MachineID) of isValid
        classifier are time dependent
        - Iteratively remove these time dependent features from original model
        so that accuracy do not deteriorate much. Those that can be removed
        without accuracy compromise are removed from final model. This force
        the tree to use time invariant features to do prediction.

#### Random Forest From Scratch (01:02:00)
- To reimplement something already existing, we test the accuracy against
existing implementation
- Write code top down
    - assume required but unwritten functions already implemented. Define these
     function inputs and outputs but defer implementation until the end.
- OOP concept introduction


### Lecture 6: Data Product & Live Coding
- Read Jeremy Howard Book on Data Products
- Drive train approach
    - Define objective
    - Identify levers (things company have control to drive objectives)
    - Identify data company can collect
    - Models (generate predictive models that fit simulation models to show how
     levers influence objective, e.g. show how price change profitability
     using elasticity model * margin model). We want to predict change in
     behavior and the absolute probability to see effect of a lever (e.g.
     change in purchase probability from a call rather than absolute purchase
     probability. No point to target most probable customers if they are likely
      to buy even without intervention)
    - A simulation model is one based on similar relationships. Part of it can
    be values from various scenarios while another part are stochastic fields
    filled by predictive models
    - We are most interested in the intersection between model key features
    that drive outcomes and company levers.
- Majority of models help provide information but doesnt automate anything

#### Revision (0:37:00)
- Base prediction confidence on variance amongst trees
    - wider the variance, greater the uncertainty
    - We can for example approval a lower loan quantum for applications with
    greater uncertainty
    - We can also group observations by uncertainty bands to understand why
    model is less sure (e.g. after grouping, we realise these observations are
    rarer, i.e. lower count)
- Feature importance
    - Permutation importance established observing the decrease in
    accuracy/metric after shuffling a particular feature.
    - Presence of perfectly colinear features may mislead by saying one feature
     is not importance. Need to shuffle colinear features ideally.
- Partial Dependence Plot
    - univariate plots (feature_x and y) are often misleading as the change in
    y is often not due to feature_x but rather due to other features/factors.
    - Partial dependence plot mitigates this somewhat by keep other
    independence variables constant and varying feature under study through all
     possible values and observe the change in y. A downside to this
     technique is relationship could be due to value combinations that are
     impossible in real life.
- Tree Interpreter
    - Helps to breakdown the prediction for a row into overall tree bias and
    contributions by each feature across a tree.
    - Pip install waterfall library by Chris

#### Portfolio Building By Contributing to Open Source Libraries
- google https://hub.github.com
- git fork -> git push your_user_feature -> git pull-request

#### Missing Values
- Never remove rows with missing values. They often contain valuable
information (e.g. data collection constraints, data leakages)

#### Tree Based Feature Interaction Importance (1:17:22)
- We could do tree interpreter for each row to obtain another form of feature
importance
- Insert tree image
    - Enclosure interact Year_made = 9.7 - 10 = -0.3
    - Year_made interact Meter = 9.4-9.5 = -0.1
    - Meter interact Enclosure = ?

#### Live Coding for Extrapolation Using Simulated Data (1:25:00)
- Insert picture of trending data for prediction using trees
- Lecture 5, we tried to mitigate extrapolation limitation by reducing model
dependency on time dependent features. However, improvement using this
technique is limited. To address extrapolation problem well, we can:
    - Use a function that can extrapolate form of data well e.g. neural nets OR
    - Fit a time series, detrend it, use Random Forest to predict detrend data.


### Lecture 7:
- Random Forest, Gradient Boosting Machines and Neural Nets are the 3 classes
of algorithms needed in machine learning problems
- Benefits of Random Forest over GBM:
    - Harder to screw up
    - Easier to scale

#### Sizing Validation Set (until 0:18:52)
- Depends on how precise one needs to quantify the accuracy

#### Random Forest From Scratch (0:19:00)
- @property decorator allows a class method to be called without (). These
methods do not accept parameters and are usually used to perform calculations
- Only leaves have score = infinity as they cant be split further
    - scores here refer to improvement post split

#### 01:20:00 How to Ask for Help
- Checkout Wiki page on how to ask for help
- Create a gist (github equivalent) to share code for jupyter notebook
- This is available under jupyter extension that can be install under pip
- Select gist it and collapsible heading under jupyter extension tab

#### 01:23:30 Neural Networks
- Why neural networks?
    - Trees are limited in situations when there is need to
    extrapolate/calculate
    - Linear models can do that but cannot model non-linearity
- Machine Learning course will only cover fully connected neural nets and
include details such as stochastic gradient descent
- Other architectures and best practices are covered in deep learning course


### Lecture 8
- Random Forest is a form of nearest neighbours algorithm achieved through tree
 based feature space partitioning
- However neural nets are better at unstructured data. For example they are
adept in image recognition problems when spatial relations picked up at initial
layers are combined at latter layers.
- Pickle works for any python object but is not optimised for any.
- Terminology:
    - Vector == 1D array == rank 1 tensor
    - Matrix == 2D array == rank 2 tensor
    - 3D array == rank 3 tensor
    - Dim axis = 1 : Across columns
    - Dim axis = 0 : Down a column/Across rows

#### Scaling
- Random Forest do not have statistical assumptions and are based on relative
orders when splitting. Hence no scaling need. Order of X vs (X-mean)/Std_dev is
 the same relative order.
- Scale matters for neural networks, hence we need to scale data before
training/predicting.
- We get the mean and std-deviation from trainset and apply them to valid/test
set. For multiple input channels (e.g. RGB for image recognition), we can scale
 by mean and std_deviation for each channel.

#### Tensor Dimensions and Manipulations
- Almost all neural net libraries expect 1st parameter to represent the number
of observations
- Be familiar with np.reshape to get the right dimensions
~~~ python
np.reshape(tensor, (-1, 28, 28)) # reshape will figure out value for -1 field
~~~
- Practice common tensor manipulations:
    - OpenCV requires channels in Blue, Green, Red but data is in RGB, how to
    reorder?
    - PIL reads images in row x column x channels but pytorch needs it in
    channels x rows x columns, how to reorder?

#### Neural Networks (0:38:35)

#### Why You Blog (0:45:43)

#### Pytorch (0:47:35)
- Code developed in pytorch can run on GPU. Operation interfaces look similar
to numpy
- AWS setup from 0:53:30
- torch.nn is a pytorch module for neural nets.
~~~ python
import torch.nn as nn
# Sequential is a container for network archi from input to output order
nn.Sequential(
    nn.Linear(28x28, 10), # Linear layer with 28x28 input and 10 output
    nn.LogSoftmax() # softmax activation
).cuda() # call cuda to copy neural network to run on GPU

# model data structure wrapper for train, valid and optional test data
md = ImageClassifierData.from_arrays(path, (x,y), (x_valid, y_valid)

# aka -ve loglikelihood loss or cross entropy (binary/categorical(multi-class))
loss = nn.NLLLoss()
~~~

#### 1:18:00 Neural Net from Scratch
- we can create custom neural network layer/architecture from scratch by
creating class that inherits from nn.Module
- Every module in pytorch is either a neural network or layer of neural net.
Pytorch allow plug and play of pre-existing or custom module
- Subclass need to initialize superclass using super().__init__()
- parameter is initialized by controlling variance based on Kaiming He
~~~ python
torch.randn(dims)/dims[0] # dims[0] is number of inputs
~~~
- get_weights() need to return a Parameter type so that pytorch knows that is
the object to update during gradient descent
- Each custom layer just need to implement the forward method for forward pass
and pytorch will provide the derivative computation automatically.
- forward method is called when a layer is called when receiving input.
- pytorch uses view for reshaping.
~~~ python
x.view(x.size(0), -1)
# view is equivalent to reshape in numpy
# x.size(0) gives m, the number of observations)
# reshaping to (m, -1) is equivalent to flattening
~~~


### Lecture 9: Regularization, Learning Rate and NLP

#### Interesting Content
- Structuring the unstructured is a pictorial way to understand how random
forest predicts and effects of bagging
- Parfit is a library by fastai alumni to do parallel fitting for trees

#### Lesson 9 Objective
- Implement neural net library, fit function and optimiser
- We just need to write python code, rely on PyTorch to run code on GPU, and
auto-differentiate to find gradient for us.

#### Building Own Custom Layer/Architecture
- Pytorch allows us to build our own custom layer/neural net architecture for
insertion into larger net architecture
- To do this, our layer needs to inherit from nn.Module, then in our
constructor, use super to call Module constructor
- We need to **overload a forward function** which pytorch auto-calls to
calculate the output from this layer given an input
~~~ python
output = torch.matmul(x, self.l1_w) + self.l1_b
x @ self_l1_w # python simplified form of matrix mulitplication
~~~

- We choose appropriate activation function based on required input/output
mapping.
- softmax is used as activation of output layer as individual values range from
 0-1 and collectively sums to 1, best to select 1 class out of many.
- To predict multiple items in image, we use sigmoid activation.
- get_weights return Parameter type, which are auto picked up by Pytorch for
updating using optimiser
- md = ImageClassifierData.from_arrays is similar to python generator. It grabs
 data and generates mini-batches for training NN.
- iter(md.train_dl) converts it to iterator, next command use to retrieve each
mini-batch
- To enable auto-differentiation, Pytorch utilises Variable object to remember
computations performed. Hence, we need to wrap a tensor using a Variable, which
 has a superset of APIs vs a tensor.
~~~ python
pred = net2.(vxmb).exp()
# vxmb represents Variable X mini batch
# We take exponential to reverse the log applied
~~~

- By adding a non-linear activation (RelU), logistic regression becomes neural
network
- Instead of having argmax command to find index of max value, PyTorch provides
 max function, which returns both max values and index of max values.

#### 47:00 Broadcasting
~~~ python
# Element wise addition
np.array([10,6,-4]) + np.array([2,8,7]) = np.array([12,14,3])

# PyTorch Tensor Equivalent
T([10,6,-4]) + T([2,8,7]) = T([12,14,3])
~~~
- NDarrays/Tensors in numpy/PyTorch can perform vectorized operations due to
Single Instruction Multiple Data (SIMD), which is much faster than loops
- Most critical to write loopless code in deep learning for efficient compute
- 1D array == Rank 1 Tensor
- 2D array == Matrix == Rank 2 Tensor
~~~ python
a = np.array([10,6,-4])
a > 0 # Returns np.array([True, True, False])
~~~
- Broadcasting allows efficient computation between compatible shapes
- It doesnt replicate by copying values but repeat computation at same location
 for certain times to make shape compatible.
- Ways to change dimensions
    - np.expand_dims() can be used to increment rank e.g. (3,) to (3,1)
    - a[None].shape # (1,3)
    - a[:, None].shape #(3,1)
    - a[None,:,None].shape # (1,3,1)
    - We can use np.broadcast_to(c, target.shape) to see result of broadcasting

#### 01:02:45 Broadcasting Rules
- Need to understand broadcasting rules to minimize bugs in deep learning
computation.
- Common to have rank 4 tensor (e.g. obs,pix_x, pix_y, channel) operation with
rank 1 tensor
- When numpy/PyTorch compare shapes, it starts with trailing dimensions,then
works inwards. Two dimensions are compatible if
    - They are equal
    - One of them is 1.

Example1
m.shape is (3,3), c.shape is (3,1)
m       3,  | 3
c       *1* | 3
Result  3,    3
# Missing value is given a 1, which is axis that is repeated

Example2
3D array for image: 256(Pixels) x 256(Pixels) x 3(Channels)
1D array (Scale):   *1*         x *1*         x 3 (Subtract channel mean)
Result              256         x 256         x 3

Example3
a[None] * a[:, None]
(1,3) * (3, 1) = (3,3)


#### 01:19 Fit Function

~~~ python
l = loss(y_pred, Variable(yt).cuda())
~~~
- to get gradient, we call backward from loss
- Need to set gradient to zero to remove old values before each gradient
calculation



### Lecture 10: NLP and Columnar Data

#### Interesting Content
- Check out winning entry by Michael Jahror for Kaggle Competition: Porto
Seguro Safe Driver Prediction
- Incoporated semi-supervised learning techniques to incorporate unlabelled
data in modelling. Used gradient boosting on 5 neural nets for structured data

#### 09:00 Resume walk-through of bottom-up workflow
- V(x) for variable, a data structure in pytorch that tracks the computations
performed for these data to enable gradients to be computed subsequently. This
is similar to T() for tensor (similar to numpy arrays)
- Checkout Pytorch.autograd.variables documentation for more info on Variables

#### 20:15
- Weights w and b are of type Variable. To access underlying tensor, we need to
 access their data attribute e.g. w.data
- zero_() function will replace content with 0, _ indicate inplace.
- Small learning rate allows more stable convergence but is slow and may need
many more epochs.
- optimizer.zero_grad() and optimizer.step() are just wrapper functions to
reset tensor and update the weight using backpropagation
~~~ python
optimizer.zero_grad() is equivalent to
if w.grad is not None:
    w.grad.data.zero_()
    b.grad.data.zero_()
optimizer.step() is equivalent to
w.data -= w.grad.data * lr
b.grad -= b.grad.data * lr
~~~
- Next, we will put computations for the epoch loops to a fit function
- nn.Sequential(a, b) represents doing a then use the output for b.

#### 34:15
- Get diagram
- Learning rate annealing refers to lowering the learning when loss values
start oscillating to prevent overshoots and find paramters that gives us a
lower overall loss value
- set_lrs(optimizer, new_learn_rate) is available in fastai library to change
learning rate.

#### 40:00
- Getting parameters of neural network
- net.parameters() returns the tensors for each layer, numel() returns the
shape
- total parameters approx 100k, may overfit, consider regularization
~~~ python
[o.numel() for o in net.parameters()]
[78400, 100, 10000, 100, 1000, 10]
784 inputs x 100 output, 100 bias_1, 100 input x 100 output, 100 bias_2, 100
input x 10 output, 10 bias_3
~~~

#### 47:00
- Regularization
    - L1: abs(w)
    - L2: Loss + a(w)**2 (Applied on loss function)
    - L2: delta = 2aw (Applied on gradient term i.e. weight decay)
- Having weight decay should result in higher final train loss but lower
validation loss due to better generalization.
- However, sometimes it smoothen the loss function such as training loss
decline faster at the initial stage compared to when no regularization is
applied like in this case. (Insert picture for good vs bad loss function)
- If training loss >> validation loss -> underfitting, reduce regularization
- If training loss << validation loss -> overfitting, add regularization
- **Trick to get good model accuracy is to have large number of parameters,
then control overfitting using regularization**.

#### 1:00
- Making neural networks more interpretable is avenue with rich contribution
potential. E.g. one could explore using gradient to see which input has most
impact to output.

#### 01:02 NLP
- Check source code: ??texts_from_folders
- We use countVectorizer to obtain bag of words
    - features are unique words in text, values are the number of occurrence of
    that word in particular review.
- The resultant loss of word order doesnt degrades sentiment analysis too much
but tends to be critical in other NLP applications
- Use fit_transform to build the word dictionary on training set. transform to
map words in validation/test set using dictionary built on train. Have unknown
as container for new words in valid/test not seen in train.
- countVectorizer results in a sparse matrix for efficient storage. (e.g.
observation 1, feature_4, value = 3)

#### 01:17 Naive Bayes
- Excel computation
~~~~
Text                    Label   this| movie| is| good| the| bad | P(R|L=1)  | P(R|L=0) | Ratio
This movie is good      1       1   | 1    | 1 | 1   | 0  | 0   | 0.67      | 0.222    | 3
The movie is good       1       0   | 1    | 1 | 1   | 1  | 0
This movie is bad       0       1   | 1    | 1 | 0   | 0  | 1
The movie is bad        0       0   | 1    | 1 | 0   | 1  | 1
                        ones    1   | 1    | 1 | 1   | 1  | 1
                   p(word|L=1)  0.67| 1    | 1 | 1   |0.67| 0.33
                   p(word|L=0)  0.67| 1    | 1 |0.33 |0.67| 1
Bayes Rule:
P(L=1|Review) = P(Review|L=1)* P(L=1) / P(Review)

Odds of Positive to Negative Review
P(L=1|R)    P(R|L=1) * P(L=1)   P(R|L=1)   (num_1+1)/(num_1+num_0+2)
-------- =  ----------------- = -------- * -------------------------
P(L=0|R)    P(R|L=0) * P(L=0)   P(R|L=0)   (num_0+1)/(num_1+num_0+2)

  P(R|L=1)   (num_1+1)       len(L=1)+1
= -------- * --------- = r * ----------  = log(r) + log(b)
  P(R|L=0)   (num_2+1)       len(L=0)+1

For Naive Bayes
P(R|L=1) = P(word_1|L=1)*...*P(word_n|L=1) for the words that appear in that
review. This assumes word co-occurrence is independent which is unfortunately
not true. This is the trade-off using simple naive bayes assumption.

Odds >> 1 => greater probability positive review
Odds less than 1 => greater probability negative review
~~~~
- Code implementation
    - p/p.sum() = p(word|L)
    - r equals odds ratio
    - Use log to allow multiplication to become addition
    - val_term_doc.sign() binarize the values, >0 = 1, <0 = -1
    - Reasonable results for theory based naive bayes approach
- Option 2 is to fit logistic regression on the count, label dataset
    - We use dual = true in this case because the term document matrix is wider
    then tall, solving its dual equivalent is faster
    - Binarize version turns out better
    - We can add regularization by setting C from large value (i.e.
    regularization off) to small value (turn it on) since there are many
    features for only 25k training examples
    - CountVectorizer(ngram_range(1,3)...) generate bag of words features
    containing 1 to 3 words (e.g. 'good', 'is good', 'movie is good'). This
    massively inflate the feature count so setting max_feature can limit the
    top most frequently occuring features.


### Lecture 11: Embeddings

- Most methods in machine learning eventually decomposes to trees or matrix
multiplication (e.g. naive bayes, neural networks)
- Prediction using theory (Naive Bayes with feature independence assumption)
performs less well then parameters learning using training (Logistic
Regression)
- Regularization help model generalize better

#### 0:22:00 Incoporate Naive Bayes as Prior
- use product of naive bayes ratio with term document matrix rather than the
latter alone led to better performance
~~~ python
x_nb = x.multiply(r) # r is naive bayes prior
~~~
- Since we fit x_nb*w => x*r*w => x.(rw), shouldnt it be mathematically
equivalent to logistic regression functional form, why does it perform better?
    - Regularisation in logistic regression meant it favors small values in w.
    - Incorporating prior coupled with small w meant small change from prior
    belief given by naive bayes computation, which is suitable in this use case
    - Idea arose from Research paper: Simple, Good Sentiment & Topic
    Classifications by Sida Wang and Christopher Manning which achieved 91.22
    using NBSVM-bigrams)
    - Fastai interpreted the reasoning behind the out-performance
    (incorporating prior in bayesian statistics manner) and improved the code
    using GPU to facilitate fast training up to trigrams to improve upon their
    benchmark
    - Shows the strength of using pytorch to customize concepts and experiment
    using GPU processing.

#### 0:43:40 Fastai Version of NBSVM: NBLR
~~~ python
md = TextClassifierData.from_bow(trn_term_doc, trn_y, val_term_doc, val_y, sl)
# TextClassiferData is a data model for text classification
# from_bow means from bag of words
# trn_term_dec: words
# trn_y: labels
# sl: Max number of unique words
md.dotprod_nb_learner() # Construct learner, train for a few epoch
~~~

- DotProdNB class added a bias to w weight called w_adj
- default value of 0.4 is derived empirically from few NLP datasets
- w=0 imply disregarding Prior from Naive Bayes while w=1 imply
not deviating from Naive Bayes. Intuitively, the ideal final w should be
between 0 and 1. However, when regularization added, it favors small w due to
loss penalization. Thus, this input is to bias it towards a non-zero value at
start of training. My Initial thought, perhaps, this may not be necessary if we
 tune regularization C as a hyperparameter instead.

#### 01:07:00 Embeddings
- We use nn.embedding instead of nn.linear because the term document x is
represented as a sparse matrix rather fully expanded matrix for space
efficiency reasons. Embedding provides a simple look-up function to retrieve
the right dimensions from the stored indices for the computation.
- Now that we can represent both numeric and categorical data (embeddings), we
can use neural nets for any problem sets beyond NLP, example grocery
predictions.
- Refer to Entity Embedding of Categorical Variables by Guo/Berkhahn and Deep
Learning Lesson 3 for more details on embeddings

#### Rossmen Grocery Prediction
- One flaw of this competition is the inclusion of weather and google
trend data which will not be available for used in prediction in production
- Third place team had an elegant and simple solution which current solution is
 based on
- Variable "month_open" is treated as categorical instead of continuous, with
max value clipped at to avoid high cardinality
- For variables that can be treated as either continuous or categorical, it may
 be best to treat them as categorical if their cardinality (unique values) is
 not too high. This is because post one-hot encoding, each category will have
 an associated neural network parameter for tuning, which tends to provide
 better performance. This best practice recommendation is contrary to my prior
 belief that numeric data confers more information and is best retained in that
  form for better model accuracy.


### Lecture 12: Complete Rossman, Ethical Issues
- Most time series dataset are often associated with some events (e.g.
weekend, public holiday, promotion) that have big impact on dependent variable
we are predicting
- Besides knowing if the day we are predicting falls on event date, it is often
 useful to include counters to/post event features (e.g. days to coming event,
 days post most recent event).
- When we need to iterate through series data in a dataframe, it is much faster
 to do it using numpy rather than pandas interface due to lower overheads.
~~~ python
# to iterate through dataframe rows
for s, v, d in zip(df.store.values, df[fld].values, df.Date.values) # fast
    ...
for row in df.iterrows(): # very slow
    ...
~~~
- There is a section of code with df=train[columns], then df=test[columns]. It
is meant to first run df=train[columns], then code following it, then change df
 to test[columns] and rerun those code following it.

#### 0:16:30 Pandas Time Series API
- Pandas provide window function called rolling(periods_to_avg, min_period=1).
min_period parameter is meant for edge cases (e.g. at start of window) when we
do not have enough data points, we can use 1.
- Pandas timeseries API has very rich commonly used operations for timeseries
data. Good to be familiar.

#### 0:22:00
- We separate the data into continuous(e.g. max_temp,
distance_to_nearest_competitor) and categorical features (embedding with latent
 features)
- Pytorch requires all continuous features to be set to float type.
- When training, always start by running a small sample, if everything appears
to be working correctly, then run on full.
- Before training, we need to set do_scale to true to normalize input to
standard normal distribution to ensure faster convergence of gradient descent
- mapper variable stores the mean and std_deviation values for each feature in
training set for use in normalizing the test set.
- To optimise for root mean square percentage error, we could use rmsle since
log y - log y_pred also yields ratio difference i.e. log (y/y_pred).
- Once model is properly tuned, retrain it on train+validation set on same
parameters, before using it on test set.
- ColumnarModelData object is a data structure for both training and validation
 datasets. It also allows to specify which columns should be treated as
 categories.
- cat_sz = number of unique values (dim) e.g. ('store', 1116)
- Rule of thumb: For NLP, empirial findings suggest have 600 latent features in
 embedding matrix to yield the best results. Less than that results in poorer
 performance but higher values increase computation time but yield negligible
 benefits.
- Other domains tend to be less complex and require less latent features. A
rule of thumb is to start with half the number of unique values subject to cap
of 50 and finetune from there.
- As number of latent features increase, so do the number of parameters. We
control overfitting by using regularization. The ultimate best choice of
paramters is a function of amount of data available and how good a feature is.
~~~ python
md.get_learner(emb_szs, len(df.columns)-len(cat_vars),
      0.04,1, [1000,500], [0.001, 001])
# emb_szs: how big is embedding matrix
# len(df.columns)-len(cat_vars): No of continuous variables
# [1000,500]: Activations for each layer
# [0.001, 001]: Drop outs to use
~~~
- In this example, our result do poorly on the public leaderboard but well on
private leaderboard. Thus, it is always important to rely on well constructed
validation set rather than overfit on public leaderboard dataset.

#### 0:44:00 Summary of Entire Course
- Trees: Utilises bagging or boosting to improve final prediction
    - One hot encoding, thought optional for trees, can help trees get to key
    level with less splits, make it more efficient. It also allow key levels to
    be identified as important features.
    - It usually does not worsen Random Forest performance unless the
    cardinality is too high which may split data too much.
- Neural Nets with Stochastic Gradient Descent: Not concerned with
colinearity. Colinearity due to one-hot encoding could lead to saddle
points of the loss functions surface, making it hard to train. This may be
eased by adding regularization to smooth the cost function
    - We could try permutation importance to interpret neural networks
    - Shuffling feature values and measure change in output is similar to
    calculating gradient of output with respect to input. Can check if the
    feature importance by permutation importance corresponds to importance
    determine by gradient by examining the gradient in py_torch
    - Try partial dependence plots for neural net too.
- How can we relate statistical significance with feature importance determined
 by permutation importance?
    - We could disregard statistical significance in big data realm. Stats
    significance is driven by sample size rather effect size in big data.
    - Stats significance only useful for small datasets where it is costly to
    collect more data. For small datasets, we could determine statistical
    significance using bootstrap, train a few times and get the confidence
    interval. If CI includes 0, not statistically significant.

#### 01:03:20 Ethics in Data Science
- Our model should influence behavior
    - Consider what kind of behavior and how? (Mindful of choices that broaden
    perspective or narrow it and make it extremist?)
    - Mindful of bias inherent in data used to train models -> positive
    feedback loop