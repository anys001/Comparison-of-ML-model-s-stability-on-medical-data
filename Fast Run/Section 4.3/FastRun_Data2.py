# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import  StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier, Pool
import pickle
import os
import time
import multiprocessing
from tqdm import tqdm

# Data loading

num_cpu_cores = multiprocessing.cpu_count()
n_jobsf = int(num_cpu_cores * 0.9)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
TARGET_NAME_SRC = 'HeartDiseaseorAttack'
TARGET_NAME = 'Target'
# reading datas
datam = pd.read_csv("./Data/data.csv", delimiter=',')

dis0 = datam[datam[TARGET_NAME_SRC].isin([0.0])]
dis1 = datam[datam[TARGET_NAME_SRC].isin([1.0])]
dis2 = dis0.sample(n=24000, random_state=42)
data = pd.concat([dis1, dis2])

data[TARGET_NAME] = data[TARGET_NAME_SRC].map({0.0: 0, 1.0: 1})

LABELS = { 0: "No Heart Diseaseor Attack", 1: "Heart Diseaseor Attack"}

FEATURES_CTG = {
  "Education": [1, 2, 3, 4, 5, 6]
  , "GenHlth": [1, 2, 3, 4, 5]
}

# For categorical features
for feature, categories in FEATURES_CTG.items():
    data[feature] = pd.Categorical(data[feature], categories=categories)
data = pd.get_dummies(data).astype('float32')

FEATURES = data.drop(columns=[TARGET_NAME_SRC, TARGET_NAME]).columns.tolist()

def transform_get_fitted_StandardScaler(df, columns):
    scaler_std = StandardScaler()
    df[columns] = scaler_std.fit_transform(df[columns])
    return df, scaler_std

# Function part

def prepare_datasets(data, sample, features, target_name='Target', random_state=42):
  Y = data[target_name]
  train_ids, test_ids = train_test_split(
    data.index, #X, Y,
    test_size = sample, #for test part
    random_state = random_state,
    shuffle = True,
    stratify=Y
  )
  X_train = data.loc[train_ids][features]
  y_train = data.loc[train_ids][[target_name]]
  X_test = data.loc[test_ids][features]
  y_test = data.loc[test_ids][[target_name]]

  X_train[features], scaler_st = transform_get_fitted_StandardScaler(X_train, features)
  df_X_train = X_train[features]
  X_test[features] = scaler_st.transform(X_test[features])
  df_X_test = X_test[features]

  X_train = X_train.values
  X_test = X_test.values
  y_train = y_train.values.reshape(-1)
  y_test = y_test.values.reshape(-1)

  scoring = {
    # 'f1_macro': 'f1_macro',
    'f1_w': 'f1_weighted',
    'AUC': 'roc_auc_ovr',
    'accuracy': make_scorer(accuracy_score),
  }

  return ((X_train, y_train), (X_test, y_test)), scoring

class MyClass:
    pass

def modelfunc(data, modelnm, random_state=42, n_jobs1 = n_jobsf):

  monitor = 'f1_w'
  ((X_train, y_train), (X_test, y_test)), scoring = data
  if modelnm == "Regression":

    model = LogisticRegression(max_iter=500, random_state=random_state, verbose=3, n_jobs=1)

    # Defining the pipeline
    pipeline = Pipeline([
    # ('standard_transform', StandardScaler()),
    # ('quantile_transform', QuantileTransformer(output_distribution='uniform')),
      ('model', model)
    ])
    param_grid = {
      'model__C': np.logspace(-2, 1, 15),  # Specify 'C' within the 'model' step of the pipeline
      'model__penalty': ['l1', 'l2'],
      'model__class_weight': [None, 'balanced', {0: 1, 1: 1.69}],
      'model__solver': ['saga', 'liblinear'],
    }

    # Creating a GridSearchCV object
    grid_search = GridSearchCV(
      estimator=pipeline, # a pipeline with feature scaling
      param_grid=param_grid,
      cv=5,
      scoring=scoring,
      refit=monitor,  # Setting the metric for re-training the best model
      verbose=3,
      n_jobs=n_jobs1
    )

    # Training using GridSearchCV
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    params = grid_search.best_params_
    best_model.fit(X_train, y_train)

  elif modelnm == "DecTree":

    pipeline = Pipeline([
      # ('standard_transform', StandardScaler()),
      # ('quantile_transform', QuantileTransformer(output_distribution='uniform')),
        ('model', DecisionTreeClassifier(criterion="gini", random_state=random_state))
    ])

    param_grid = {
        'model__max_depth': [6, 8, 10 ],
        'model__min_samples_split': [30, 70, 85],
        'model__min_samples_leaf': [8, 15, 21,25] ,
        'model__class_weight': ['balanced'],
        'model__max_features': ['sqrt','log2'],
        #'model__random_state': [random_state]
      #'min_impurity_decrease': [0.1, 0.5, 0.01]
    }

    # Creating object GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline,  # a pipeline with feature scaling
        param_grid=param_grid,
        cv=5,
        scoring=scoring,
        refit=monitor,  # Setting the metric for re-training the best model
        verbose=3,
        n_jobs=n_jobs1
    )

   # Training using GridSearchCV
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    params = grid_search.best_params_
    best_model.fit(X_train, y_train)

  elif modelnm == "RandForest":

    pipeline = Pipeline([
    # ('standard_transform', StandardScaler()),
    # ('quantile_transform', QuantileTransformer(output_distribution='uniform')),
        ('model', RandomForestClassifier())
    ])
    param_grid = {
        'model__n_estimators': [100, 250, 300],
        'model__max_features': ['sqrt', 'log2'],
        'model__max_depth': [4, 5, 6],
        'model__min_samples_split': [ 90, 100, 140],
        'model__min_samples_leaf': [ 17, 21, 34],
        'model__bootstrap': [True],
        'model__class_weight': ['balanced'],
        'model__random_state': [random_state]
    }


    # Creating object GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline, # a pipeline with feature scaling
        param_grid=param_grid,
        cv=5,
        scoring=scoring,
        refit=monitor,  # Setting the metric for re-training the best model
        verbose=3,
        n_jobs=n_jobs1
    )

    # Training using GridSearchCV
    grid_search.fit(X_train, y_train.reshape(-1))
    best_model = grid_search.best_estimator_
    params = grid_search.best_params_
    best_model.fit(X_train, y_train)

  elif modelnm == "Bagging":

    pipeline = Pipeline([
        # ('standard_transform', StandardScaler()),
        # ('quantile_transform', QuantileTransformer(output_distribution='uniform')),
        ('model', BaggingClassifier(random_state=random_state))
    ])
    param_grid = {
        'model__n_estimators': [50, 100, 150],
        'model__max_samples': [6, 8, 10],
        'model__max_features': [5, 10, 20],
        'model__bootstrap': [True],
        'model__bootstrap_features': [True, False],
        'model__oob_score': [True, False]
    }

    # Creating object GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline,  # a pipeline with feature scaling
        param_grid=param_grid,
        cv=5,
        scoring=scoring,
        refit=monitor,  # Setting the metric for re-training the best model
        verbose=3,
        n_jobs=n_jobs1
    )

    # Training using GridSearchCV
    grid_search.fit(X_train, y_train.reshape(-1))
    best_model = grid_search.best_estimator_
    params = grid_search.best_params_
    best_model.fit(X_train, y_train)

  elif modelnm == "CatBoost":

    # Data preparation
    pool_train = Pool(X_train, y_train)
    pool_test = Pool(X_test, y_test)

    # Model initialization
    best_model = CatBoostClassifier(eval_metric='AUC:hints=skip_train~false', random_seed=random_state, loss_function='Logloss', verbose=10)

    # Defining the parameter grid for the search, including iterations
    param_grid = {
        'iterations': [1500],
        'depth': [4, 6, 8],
        'learning_rate': [0.04, 0.1, 0.25],
        'l2_leaf_reg': [0.5, 1, 2, 3, 5, 8],
    }
    # Search start
    grid_search_result = best_model.grid_search(param_grid, pool_train, refit=True)

    grid_search = MyClass()
    grid_search.best_params_ = grid_search_result['params']
    grid_search.best_score_ = grid_search_result['cv_results']['test-AUC-mean']

  return best_model, grid_search

data_prepared = prepare_datasets(data, sample = 0.2, features = FEATURES)

def maintr_new(data, sample, best_model, optfl, modelnm, prepare_data, random_state=42):
  data = prepare_data(data, sample, features = FEATURES, target_name='Target', random_state=42)
  ((X_train, y_train), (X_test, y_test)), scoring = data
  if optfl == 'Y':
    best_model, grid_search = modelfunc(data, modelnm, random_state=random_state)

    Ytrue = y_test
    Yprob = best_model.predict_proba(X_test)
    Ypred = best_model.predict(X_test)

    #print(classification_report(Ytrue, Ypred, digits=3, target_names=list(LABELS.values())))
    report = classification_report(Ytrue, Ypred, digits=3, target_names=list(LABELS.values()), output_dict=True, zero_division=0)
    x_clrep = pd.DataFrame(report).transpose()

    best_params1 = grid_search.best_params_

    df = pd.DataFrame([best_params1])
    acc = accuracy_score(Ytrue,Ypred)
    df['Accuracy'] = acc

    return df
  else:
    Ytrue = y_test
    Ypred = best_model.predict(X_test)

    report = classification_report(Ytrue, Ypred, digits=3, target_names=list(LABELS.values()), output_dict=True, zero_division=0)
    acc = accuracy_score(Ytrue,Ypred)

    my_dict = {'DataSize':len(data),'Accuracy':acc}
    df = pd.DataFrame([my_dict])

    return df

# With optimization function

BestModel_rg, _ = modelfunc(data = data_prepared, modelnm = "RandForest") #main

# Main experiment part

def stab_alldf(data, step, sample_ratio, BestModel, model_name, optimize='N', restore_state_path=None, download_state=False, samples_count=10, seed=None):
    if restore_state_path and os.path.exists(restore_state_path):
        with open(restore_state_path, 'rb') as file:
            state = pickle.load(file)
        dfs, percent_removed, save_count = state['dfs'], state['percent_removed'], state['save_count']
        print(f"Restored state from {restore_state_path} with {percent_removed}% removed.")
    else:
        dfs = []
        percent_removed = 0
        save_count = 0

    pbar = tqdm(total=91, initial=percent_removed, desc="Progress")
    start_time = time.time()  # Record the starting time
    while percent_removed < 91:
        n_samples = int(len(data) * (1 - percent_removed / 100))

        for i in range(samples_count if percent_removed > 0 else 1):
            sampled_data = data.sample(n=n_samples, random_state=i + (0 if seed is None else int(seed))) if n_samples < len(data) else data

            accuracy = maintr_new(data=sampled_data, sample=sample_ratio, best_model=BestModel, optfl=optimize, modelnm=model_name, prepare_data=prepare_datasets, random_state=seed)

            accuracy['PercentRemoved'] = percent_removed
            dfs.append(accuracy)

        pbar.update(step)
        percent_removed += step
        if percent_removed > 91:
            percent_removed = 91

        # Checkint the time
        if time.time() - start_time >= 3600:
            concatenated_df = pd.concat(dfs, ignore_index=True)
            save_results(concatenated_df, save_count, model_name)
            save_count += 1
            start_time = time.time()
            print(f"Progress: {percent_removed:.2f}% deleted")
            dfs = []

    pbar.close()

    concatenated_df = pd.concat(dfs, ignore_index=True)
    save_results(concatenated_df, save_count, model_name)

    return concatenated_df

def save_results(data, count, model_name):
    destination_folder = f'Output/{model_name}'
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    file_path = f'./{destination_folder}/result_df_rf_{count}.pkl'
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

    print(f"File saved as: {file_path}")
# Call the function
result_df = stab_alldf(
    data,
    step=0.07,
    sample_ratio=0.2,
    BestModel=BestModel_rg,
    model_name="RandForest",
    optimize='N',
    #restore_state_path="/content/backup_state/last_state_20240407-092026.pkl",
    #download_state=True,
    seed=42,
    samples_count=10,
)