import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import sys
import pickle as pkl

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

# creating the data


def read_data(path, largely_missing=None):
    """
    :param path: path to data
    :param largely_missing: columns that have above 85% missing values
    :return: The dataframe that hold the data after preprocessing
    """
    data_path = path
    folder_files = os.listdir(data_path)
    labels = []
    patients_dfs = []
    df = pd.read_csv(f"{data_path}/{folder_files[0]}", sep='|')
    total_columns = df.shape[1]
    columns = df.columns
    missing = [[] for _ in range(total_columns)]
    # counting missing values and imputing missing values and reading the files

    for file_name in sorted(folder_files):
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
    # continue imputing  and averging the values per patient

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
    return all_df

# The columns that had above 85% averge missing data on the train
largely_missing =[7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
df = read_data(sys.argv[1],largely_missing)
# df = df.drop(df.columns[0], axis=1)

df_index = df['patient_number']
df = df.drop(['SepsisLabel','patient_number'],axis=1)

def loading_pickle_file(pkl_name):
    """
    loading a pickle file
    :param pkl_name:
    :return: The file that got read
    """
    with open(pkl_name, 'rb') as f:
        obj = pkl.load(f)
    return obj

model = loading_pickle_file('xgb_model.pkl')
# The columns index that got choose by the forward feature selection
chosen_columns = [39, 58, 65]
predictions = model.predict(df.iloc[:, chosen_columns])
ids=[f'patient_{index+1}' for index, _ in enumerate(list(df_index))]
data_dict = {'id':ids,'prediction':predictions}
df_preidcted = pd.DataFrame(data=data_dict)
df_preidcted.to_csv('prediction.csv',index=False)

