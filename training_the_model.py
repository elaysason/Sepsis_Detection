import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from xgboost import plot_importance
import warnings
import pickle
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


def read_data(path, test, largely_missing=None):
    """
    reading the data and performing preprocess stages and saving it
    :param path: path to data
    :param largely_missing: columns that have above 85% missing values
    """
    data_path = path
    folder_files = os.listdir(data_path)
    labels = []
    patients_dfs = []
    df = pd.read_csv(f"{data_path}/{folder_files[0]}", sep='|')
    total_columns = df.shape[1]
    columns = df.columns
    missing = [[] for _ in range(total_columns)]

    for file_name in folder_files:
        df = pd.read_csv(f"{data_path}/{file_name}", sep='|')
        total_values = df.shape[0]
        missing_values_count = (df.isna().sum() / total_values)
        for j in range(len(missing_values_count)):
            missing[j].append(missing_values_count[j])
        df = df.ffill()
        patient_number = file_name.split('_')[1].split('.')[0]
        df['patient_number'] = int(patient_number)
        patients_dfs.append(df)
        all_df = None
    if not test:
        largely_missing = [k for k in range(total_columns) if sum(missing[k]) / len(missing[k]) > 0.85]

    i = 0
    for patient in patients_dfs:
        print(f"[{i}/{len(patients_dfs)}], {patient['patient_number'].unique()[0]}")
        df = patient
        df = df.fillna(-1)
        if sum(df['SepsisLabel']) != 0:
            labels.append(1)
        else:
            labels.append(0)
        index = df[df['SepsisLabel'] == 1].index.min()
        df = df.loc[:index].drop('SepsisLabel', axis=1)
        df_avg = df.mean().to_frame().T

        if all_df is None:
            all_df = df_avg
        else:
            all_df = pd.concat([all_df, df_avg])
        i += 1
    all_df['SepsisLabel'] = labels
    for col_index in largely_missing:
        all_df[columns[col_index] + '_missing_percent'] = missing[col_index]
    # train_df=train_df.drop(train_df.columns[0], axis=1)
    if test:
        all_df.to_csv('data_test333.csv')
    else:
        all_df.to_csv('data_train333.csv')
        return largely_missing


largely_missing_train = read_data('./data/train', False)

train = pd.read_csv('./data_train333.csv')
train = train.drop(train.columns[0], axis=1)
y_train = train['SepsisLabel']
train_index = train['patient_number']
X_train = train.drop(['SepsisLabel', 'patient_number'], axis=1)


def f1_eval(y_pred, train_data):
    """
    alternative eval metric that will optimize f1_score
    :param y_pred: the prediction
    :param train_data: the actual data
    :return:
    """
    y_true = train_data.get_label()
    # calculating the 1 - f1_score
    err = 1 - f1_score(y_true, np.round(y_pred))
    return 'f1_err', err


# First model: XGboost

XGBClassifier = xgb.XGBClassifier(
    objective='binary:logistic', eval_metric=f1_eval, seed=7, random_state=7)

parameters = {
    'max_depth': range(2, 8),
    'n_estimators': range(60, 120, 20),
    'learning_rate': [0.1, 0.35, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
}

# The columns that got chosen by the forward selection model
chosen_columns = [39, 58, 65]

grid_search_xgb = GridSearchCV(
    estimator=XGBClassifier,
    param_grid=parameters,
    scoring='f1',
    cv=10
)
grid_search_xgb.fit(X_train.iloc[:, chosen_columns], y_train)
# saving the model
pickle.dump(grid_search_xgb.best_estimator_, open("xgb_model.pkl", "wb"))
