import os
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import shap
from mlflow.tracking import MlflowClient

## read in data ##

input_df = pd.read_csv('data/train.csv')

input_df = input_df[input_df['Year'] <= 2016]

# Train/test split on <= 2014 vs 2015-2016
input_df_train = input_df[input_df['Year'] <= 2014]
input_df_test = input_df[input_df['Year'] > 2014]

X = input_df.drop('WHOSIS_000001', axis=1)
y = input_df['WHOSIS_000001']
X_train = input_df_train.drop('WHOSIS_000001', axis=1)
y_train = input_df_train['WHOSIS_000001']
X_test = input_df_test.drop('WHOSIS_000001', axis=1)
y_test = input_df_test['WHOSIS_000001']


## configure MLFlow ##

# set URI based on databricks CLI profile (https://docs.databricks.com/dev-tools/cli/index.html)
os.environ['MLFLOW_TRACKING_URI'] = 'databricks://azure-field-eng'

EXPERIMENT_NAME = '/Users/dave.carlson@databricks.com/mlflow-demo'
RUN_NAME = 'xgboost-demo'

client = MlflowClient()

# create experiment if it does not exist
if not client.get_experiment_by_name(EXPERIMENT_NAME):
    client.create_experiment(EXPERIMENT_NAME)

experiment_id = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id


## train xgboost model and log to MLFlow ##

with mlflow.start_run(experiment_id=experiment_id, run_name=RUN_NAME) as run:

        progress = {}
  
        xgb_params = {
            'objective': 'reg:squarederror', 
            'eval_metric': 'rmse', 
            'max_depth': 27, 
            'learning_rate': 0.07630042641646584, 
            'reg_alpha': 0.009556862202281441, 
            'reg_lambda': 58.72292518686038, 
            'gamma': 0.0054796949918053785, 
            'min_child_weight': 14.382406241662876, 
            'importance_type': 'total_gain', 
            'seed': 0
            }

        # log model parameters
        mlflow.log_params(xgb_params)

        # train model
        train = xgb.DMatrix(data=X_train, label=y_train)
        test = xgb.DMatrix(data=X_test, label=y_test)
        booster = xgb.train(params=xgb_params, dtrain=train, num_boost_round=1000, early_stopping_rounds=10, evals=[(test, "test")], evals_result=progress)

        # log best iteration and test metric
        mlflow.log_param('best_iteration', booster.best_iteration)
        mlflow.log_metric('test-rmse', progress['test']['rmse'][booster.best_iteration])

        # log model
        mlflow.xgboost.log_model(booster, "xgboost")

        # log shap plot
        code_lookup_df = pd.read_csv('data/descriptions.csv')
        code_lookup = {r['Code']:r['Description'] for r in code_lookup_df.to_dict(orient='records')}
        display_cols = [code_lookup[c] if c in code_lookup else c for c in X.columns]

        shap_values = shap.TreeExplainer(booster).shap_values(X, y=y)
        shap.summary_plot(shap_values, X, feature_names=display_cols, plot_size=(14,6), max_display=10, show=False)
        plt.savefig("summary_plot.png", bbox_inches="tight")
        plt.close()
        mlflow.log_artifact("summary_plot.png")
