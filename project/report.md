#%% md

# Predict Bike Sharing Demand with AutoGluon Template

#%% md

## Project: Predict Bike Sharing Demand with AutoGluon
This notebook is a template with each step that you need to complete for the project.

Please fill in your code where there are explicit `?` markers in the notebook. You are welcome to add more cells and code as you see fit.

Once you have completed all the code implementations, please export your notebook as a HTML file so the reviews can view your code. Make sure you have all outputs correctly outputted.

`File-> Export Notebook As... -> Export Notebook as HTML`

There is a writeup to complete as well after all code implememtation is done. Please answer all questions and attach the necessary tables and charts. You can complete the writeup in either markdown or PDF.

Completing the code template and writeup template will cover all of the rubric points for this project.

The rubric contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this notebook and also discuss the results in the writeup file.

#%% md

## Step 1: Create an account with Kaggle

#%% md

### Create Kaggle Account and download API key
Below is example of steps to get the API username and key. Each student will have their own username and key.

#%% md

1. Open account settings.
![kaggle1.png](attachment:kaggle1.png)
![kaggle2.png](attachment:kaggle2.png)
2. Scroll down to API and click Create New API Token.
![kaggle3.png](attachment:kaggle3.png)
![kaggle4.png](attachment:kaggle4.png)
3. Open up `kaggle.json` and use the username and key.
![kaggle5.png](attachment:kaggle5.png)

#%% md

## Step 2: Download the Kaggle dataset using the kaggle python library

#%% md

### Open up Sagemaker Studio and use starter template

#%% md

1. Notebook should be using a `ml.t3.medium` instance (2 vCPU + 4 GiB)
2. Notebook should be using kernal: `Python 3 (MXNet 1.8 Python 3.7 CPU Optimized)`

#%% md

### Install packages

#%%

!pip install -U pip
!pip install -U setuptools wheel
!pip install -U "mxnet<2.0.0" bokeh==2.0.1
!pip install autogluon --no-cache-dir
# Without --no-cache-dir, smaller aws instances may have trouble installing

#%% md

### Setup Kaggle API Key

#%%

# create the .kaggle directory and an empty kaggle.json file
!mkdir -p /root/.kaggle
!touch /root/.kaggle/kaggle.json
!chmod 600 /root/.kaggle/kaggle.json

#%%

# Fill in your user name and key from creating the kaggle account and API token file
import json
kaggle_username = "clintonchikwata"
kaggle_key = "3994fd68dbfcfcd655136873980ad056"

# Save API token the kaggle.json file
with open("/root/.kaggle/kaggle.json", "w") as f:
    f.write(json.dumps({"username": kaggle_username, "key": kaggle_key}))

#%%

# Add the Kaggle Package 
!pip install kaggle

#%% md

### Download and explore dataset

#%% md

### Go to the bike sharing demand competition and agree to the terms
![kaggle6.png](attachment:kaggle6.png)

#%%

# Download the dataset, it will be in a .zip file so you'll need to unzip it as well.
!kaggle competitions download -c bike-sharing-demand
# If you already downloaded it you can use the -o command to overwrite the file
!unzip -o bike-sharing-demand.zip

#%%

import pandas as pd
from autogluon.tabular import TabularPredictor

#%%

# Create the train dataset in pandas by reading the csv
# Set the parsing of the datetime column so you can use some of the `dt` features in pandas later
train = pd.read_csv('train.csv', parse_dates=['datetime'])
train.head()

#%%

# Simple output of the train dataset to view some of the min/max/varition of the dataset features.
train.describe()

#%%

# Create the test pandas dataframe in pandas by reading the csv, remember to parse the datetime!
test = pd.read_csv('test.csv', parse_dates=['datetime'])
test.head()

#%%

test.info()

#%%

test.describe()

#%%

# Same thing as train and test dataset
submission =  pd.read_csv('sampleSubmission.csv', parse_dates=['datetime'])
submission.head()

#%% md

## Step 3: Train a model using AutoGluon’s Tabular Prediction

#%% md

Requirements:
* We are predicting `count`, so it is the label we are setting.
* Ignore `casual` and `registered` columns as they are also not present in the test dataset. 
* Use the `root_mean_squared_error` as the metric to use for evaluation.
* Set a time limit of 10 minutes (600 seconds).
* Use the preset `best_quality` to focus on creating the best model.

#%%

# remove casual and registered columns
df_train_cols = train.columns.to_list()
df_train_cols.remove('casual')
df_train_cols.remove('registered')

df_train = train[df_train_cols]
df_train.head()

#%%

predictor = TabularPredictor(
    label='count', 
    problem_type='regression', 
    eval_metric='root_mean_squared_error'
    ).fit(
    train_data=df_train, 
    time_limit=600, 
    presets='best_quality', 
    )

#%% md

### Review AutoGluon's training run with ranking of models that did the best.

#%%

predictor.fit_summary()

#%% md

### Create predictions from test dataset

#%%

predictions = predictor.predict(test, as_pandas=True)
predictions.head()

#%% md

#### NOTE: Kaggle will reject the submission if we don't set everything to be > 0.

#%%

# Describe the `predictions` series to see if there are any negative values
predictions.describe()

#%%

# How many negative values do we have?
(predictions < 0).sum()

#%%

# Set them to zero
predictions[predictions < 0] = 0

#%% md

### Set predictions to submission dataframe, save, and submit

#%%

submission["count"] = predictions
submission.to_csv("submission.csv", index=False)

#%%

!kaggle competitions submit -c bike-sharing-demand -f submission.csv -m "first raw submission"

#%% md

#### View submission via the command line or in the web browser under the competition's page - `My Submissions`

#%%

!kaggle competitions submissions -c bike-sharing-demand | tail -n +1 | head -n 6

#%% md

#### Initial score of `1.80412`

#%% md

## Step 4: Exploratory Data Analysis and Creating an additional feature
* Any additional feature will do, but a great suggestion would be to separate out the datetime into hour, day, or month parts.

#%%

# Create a histogram of all features to show the distribution of each one relative to the data. This is part of the exploritory data analysis
train.hist(figsize=(15,15))

#%%

# create a new feature
train['hour'] = train.datetime.dt.hour
test['hour'] = test.datetime.dt.hour

#%% md

## Make category types for these so models know they are not just numbers
* AutoGluon originally sees these as ints, but in reality they are int representations of a category.
* Setting the dtype to category will classify these as categories in AutoGluon.

#%%

train["season"] = train.season.astype('category')
train["weather"] = train.weather.astype('category')
test["season"] = test.season.astype('category')
test["weather"] = test.weather.astype('category')

#%%

# View are new feature
train.head()

#%%

# View histogram of all features again now with the hour feature
train.hist(figsize=(15,15))

#%% md

## Step 5: Rerun the model with the same settings as before, just with more features

#%%

# remove casual and registered columns since they are not recorded in the testset
df_train_new_features_cols = train.columns.to_list()
df_train_new_features_cols.remove('casual')
df_train_new_features_cols.remove('registered')

df_train_new_features = train[df_train_new_features_cols]
df_train_new_features.head()

#%%

predictor_new_features = TabularPredictor(
    label='count', 
    problem_type='regression', 
    eval_metric='root_mean_squared_error'
    ).fit(
    train_data=df_train_new_features, 
    time_limit=600, 
    presets='best_quality', 
    )

#%%

predictor_new_features.fit_summary()

#%%

# Remember to set all negative values to zero
predictions_new_features = predictor_new_features.predict(test, as_pandas=True)

#%%

(predictions_new_features < 0).sum()

#%%

predictions_new_features[predictions_new_features < 0] = 0

#%%

# Same submitting predictions
submission_new_features = pd.read_csv('submission.csv')
submission_new_features["count"] = predictions_new_features
submission_new_features.to_csv("submission_new_features.csv", index=False)

#%%

!kaggle competitions submit -c bike-sharing-demand -f submission_new_features.csv -m "new features"

#%%

!kaggle competitions submissions -c bike-sharing-demand | tail -n +1 | head -n 6

#%% md

#### New Score of `0.67715`

#%% md

## Step 6: Hyper parameter optimization
* There are many options for hyper parameter optimization.
* Options are to change the AutoGluon higher level parameters or the individual model hyperparameters.
* The hyperparameters of the models themselves that are in AutoGluon. Those need the `hyperparameter` and `hyperparameter_tune_kwargs` arguments.

#%%

import autogluon.core as ag
# hyperparameter settings
gbm_options = {
    'num_boost_round' : ag.space.Int(lower=100, upper=500, default=100),
    'num_leaves' : ag.space.Int(lower=6, upper=10),
    'learning_rate' : ag.space.Real(lower=0.01, upper=0.3, log=True)
}


cat_options = {
    'iterations' : 100,
    'learning_rate' : ag.space.Real(lower=0.01, upper=0.3, log=True),
    'depth' : ag.space.Int(lower=6, upper=10)
}
xgb_options = {
    'n_estimators' : ag.space.Int(lower=100, upper=500, default=100),
    'max_depth' : ag.space.Int(lower=6, upper=10, default=6),
    'eta' : ag.space.Real(lower=0.01, upper=0.3, log=True)
}

#%%

# Choosing specific hyperparameter options
# Models : GBM, CAT, XGB, RF, XT, KNN, LR, NN_MXNET, NN_TORCH
hyperparameters = {
    'XGB' : xgb_options,
    'GBM' : gbm_options,
    'CAT' : cat_options
}

num_trials = 5
search_strategy = 'random'

hyperparameter_tune_kwargs = {
    'num_trials' : num_trials,
    'scheduler' : 'local',
    'searcher' : search_strategy
}

#%%

predictor_new_hpo  = TabularPredictor(
    label='count', 
    problem_type='regression', 
    eval_metric='root_mean_squared_error',
    path='AutogluonModels/xgb_gbm_cat_tuning_v1'
    ).fit(
    train_data=df_train_new_features, 
    time_limit=600, 
    presets='best_quality',
    hyperparameters=hyperparameters,
    hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
    )

#%%

predictor_new_hpo.fit_summary()

#%%

# Remember to set all negative values to zero

predictions_new_hpo = predictor_new_hpo.predict(test, as_pandas=True)

#%%

predictions_new_hpo[predictions_new_hpo < 0] = 0

#%%

# Same submitting predictions
submission_new_hpo = submission
submission_new_hpo["count"] =  predictions_new_hpo
submission_new_hpo.to_csv("submission_new_hpo.csv", index=False)

#%%

!kaggle competitions submit -c bike-sharing-demand -f submission_new_hpo.csv -m "new features with hyperparameters"

#%%

!kaggle competitions submissions -c bike-sharing-demand | tail -n +1 | head -n 6

#%% md

#### New Score of `0.47948`

#%% md

## Step 7: Write a Report
### Refer to the markdown file for the full report
### Creating plots and table for report

#%%

# Taking the top model score from each training run and creating a line plot to show improvement
# You can create these in the notebook and save them to PNG or use some other tool (e.g. google sheets, excel)
fig = pd.DataFrame(
    {
        "model": ["initial", "add_features", "hpo"],
        "score": [?, ?, ?]
    }
).plot(x="model", y="score", figsize=(8, 6)).get_figure()
fig.savefig('model_train_score.png')

#%%

# Take the 3 kaggle scores and creating a line plot to show improvement
fig = pd.DataFrame(
    {
        "test_eval": ["initial", "add_features", "hpo"],
        "score": [1.80412,  0.67715 , 0.47948]
    }
).plot(x="test_eval", y="score", figsize=(8, 6)).get_figure()
fig.savefig('model_test_score.png')

#%% md

### Hyperparameter table

#%%

# The 3 hyperparameters we tuned with the kaggle score as the result
pd.DataFrame({
    "model": ["initial", "add_features", "hpo"],
    "hpo1": [
        'default_vals', 
        'default_vals', 
        'GBM (Light gradient boosting) : num_boost_round: [lower=100, upper=500], num_leaves:[lower=6, upper=10], learning_rate:[lower=0.01, upper=0.3, log scale], applying random search with 5 trials'
        ],
    "hpo2": [
        'default_vals', 
        'default_vals', 
        'XGB (XGBoost): n_estimators : [lower=100, upper=500], max_depth : [lower=6, upper=10], eta (learning_rate) : [lower=0.01, upper=0.3, log scale] applying random search with 5 trials'
        ],
    "hpo3": [
        'default_vals', 
        'default_vals', 
        'CAT (CATBoost) : iterations : 100, depth : [lower=6, upper=10], learning_rate  : [lower=0.01, upper=0.3, log scale] applying random search with 5 trials'
    ],
    "score": [1.80412,  0.67715 , 0.47948]
})

#%%


