from typing import List, Literal, Union, Tuple
from collections import defaultdict

import cudf
import cupy
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import auc, f1_score, precision_score, recall_score

from cuml.metrics import confusion_matrix, precision_recall_curve, roc_auc_score
from cuml.metrics.accuracy import accuracy_score
from cuml.model_selection import train_test_split

import itertools
import os


from config_schema import XGBSingleConfig, XGBListConfig, XGBHyperparametersList, XGBHyperparametersSingle
from typing import  Union


def tran_sg_xgboost(
      param_combinations: List[Tuple[int, float, int, float]],
      data_dir,
      output_dir,
      model_file_name):
  
  dataset_dir = data_dir
  xgb_data_dir = os.path.join(dataset_dir, 'xgb')
  models_dir = output_dir

  train_data_path = os.path.join(xgb_data_dir, "training.csv")

  print(f'training data path = {train_data_path}')

  df = cudf.read_csv(train_data_path)

  # Target column
  target_col_name = df.columns[-1]

  # Split the dataframe into features (X) and labels (y)
  y = df[target_col_name]
  X = df.drop(target_col_name, axis=1)

  # Split data into train and test sets
  
  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

  # Convert the training and test data to DMatrix
  dtrain = xgb.DMatrix(data=X_train, label=y_train)
  deval = xgb.DMatrix(data=X_val, label=y_val)



  # Grid search for the best hyperparameters

  best_score = float("inf")  # Initialize best score
  best_params = None  # To store best hyperparameters

  for params_comb in param_combinations:
      
      # Create a dictionary of parameters
      params = {
          'max_depth': params_comb[0],
          'learning_rate': params_comb[1],
          'gamma': params_comb[3],
          'eval_metric': 'logloss',
          'objective': 'binary:logistic',  # For binary classification
          'tree_method': 'hist',
          'device': 'cuda'
      }


      # Train the model using xgb.train and the Booster
      evals = [(dtrain, 'train'), (deval, 'eval')]
      bst = xgb.train(params, dtrain, num_boost_round=params_comb[2], evals=evals, 
                      early_stopping_rounds=10, verbose_eval=False)
      
      # Get the evaluation score (logloss) on the validation set
      score = bst.best_score  # The logloss score (or use other eval_metric)

      print(f'trained with {params}, score = {score}')

      # Update the best parameters if the current model is better
      if score < best_score:
          best_score = score
          best_params = params
          best_num_boost_round = bst.best_iteration


  # Train the final model using the best parameters and best number of boosting rounds
  dtrain = xgb.DMatrix(data=X, label=y)
  final_model = xgb.train(best_params, dtrain, num_boost_round=best_num_boost_round)

  # Save the best model
  if not os.path.exists(models_dir):
      os.makedirs(models_dir)
  final_model.save_model(os.path.join(models_dir, model_file_name))


def run_sg_xgboost_training(
      data_dir: str,
      output_dir: str,
      training_config: Union[XGBSingleConfig, XGBListConfig],
      idx_config: int) -> None:
  """
  This function does something with the validated Pydantic object.
  """
  
  # Generate all combinations of hyperparameters

  if isinstance(training_config.hyperparameters, XGBHyperparametersList):   
    param_combinations = list(itertools.product(*training_config.hyperparameters.dict().values()))

  elif isinstance(training_config.hyperparameters, XGBHyperparametersSingle):
    h_dict = training_config.hyperparameters.dict()
    hyperparameters = {key: [h_dict[key]] for key in h_dict}
    param_combinations = list(itertools.product(*hyperparameters.values()))
    
  print(param_combinations)  
  print("** Total number of parameter combinations:", len(param_combinations))
  tran_sg_xgboost(param_combinations, data_dir, output_dir, model_file_name=f'{training_config.kind}_{idx_config}.json')

