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


# creating the data

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
print(largely_missing_train)
read_data('./data/test', True, largely_missing_train)

# reading the data

train = pd.read_csv('./data_train333.csv')
train = train.drop(train.columns[0], axis=1)
y_train = train['SepsisLabel']
train_index = train['patient_number']
X_train = train.drop(['SepsisLabel', 'patient_number'], axis=1)

test = pd.read_csv('./data_test333.csv')
test = test.drop(test.columns[0], axis=1)
train_index = test['patient_number']
y_test = test['SepsisLabel']
X_test = test.drop(['SepsisLabel', 'patient_number'], axis=1)


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


"""First model: XGboost"""


XGBClassifier = xgb.XGBClassifier(
    objective='binary:logistic', eval_metric=f1_eval, seed=7)

"""Selecting features using forward feature selection"""

sfs_sk = SequentialFeatureSelector(XGBClassifier, n_features_to_select='auto', scoring='f1', tol=0.005)
sfs_sk.fit(X_train, y_train)

chosen_columns = sfs_sk.get_support(indices=True)
#chosen_columns = [39, 58, 65]

# Running a grid search to find the best hyper parameters for the xgboost classifier.
# Training it and the grid_search.best_estimator is the trained model

parameters = {
    'max_depth': range(2, 8),
    'n_estimators': range(60, 120, 20),
    'learning_rate': [0.1, 0.35, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
}

grid_search_xgb = GridSearchCV(
    estimator=XGBClassifier,
    param_grid=parameters,
    scoring='f1',
    cv=10
)
grid_search_xgb.fit(X_train.iloc[:, chosen_columns], y_train)

# Testing f1 score on train and test sets

print('Xgboost model f1 score train: ',
      f1_score(y_train, grid_search_xgb.best_estimator_.predict(X_train.iloc[:, chosen_columns])))

print('Xgboost model f1 score test: ',
      f1_score(y_test, grid_search_xgb.best_estimator_.predict(X_test.iloc[:, chosen_columns])))
print('----------------------------')

# Looking at the importance the algorithm(XgboostClassifier) gave to each feature

plot_importance(grid_search_xgb.best_estimator_)
plt.tight_layout()
plt.show()

#saving the model
pickle.dump(grid_search_xgb.best_estimator_, open("xgb_model.pkl", "wb"))

# Model 2: Random Forest
# Running a grid search to find the best hyper parameters for the random forest classifier.

rf_classifier = RandomForestClassifier()
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [None, 4, 5, 6, 7],
    'criterion': ['gini', 'entropy']
}
grid_search_rfc = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=10)
grid_search_rfc.fit(X_train.iloc[:, chosen_columns], y_train)

print(grid_search_rfc.best_params_)

# Training a random forest classifier according to the best parameters

rfc_model = RandomForestClassifier(criterion='gini', max_depth=None, max_features='sqrt', n_estimators=100)
rfc_model.fit(X_train.iloc[:, chosen_columns], y_train)

y_pred = rfc_model.predict(X_test.iloc[:, chosen_columns])

# Testing f1 score on train and test sets

print('Random forest f1 score train: ', f1_score(y_train, rfc_model.predict(X_train.iloc[:, chosen_columns])))
print('Random forest f1 score test: ', f1_score(y_test, y_pred))
print('----------------------------')

# Looking at the importance the RandomForest algorithm gave to each feature

pd.Series(rfc_model.feature_importances_, index=X_train.iloc[:, chosen_columns].columns).nlargest(20).plot(kind='barh')
plt.tight_layout()
plt.title('Feature Importance : Random Forest')
plt.show()


lr = LogisticRegression(max_iter=500)

# Define the parameter grid for grid search
param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'C': [0.1, 1, 10],
    'solver': ['liblinear', 'saga']
}

# Perform grid search
grid_search_lr = GridSearchCV(lr, param_grid, cv=5)
grid_search_lr.fit(X_train.iloc[:, chosen_columns], y_train)



# Testing f1 score on train and test sets

print('Logistic Regression f1 score train: ', f1_score(y_train, grid_search_lr.best_estimator_.predict(X_train.iloc[:, chosen_columns])))
print('Logistic Regression f1 score test: ', f1_score(y_test, grid_search_lr.best_estimator_.predict(X_test.iloc[:, chosen_columns])))
print('----------------------------')
column_names = X_train.iloc[:, chosen_columns].columns.tolist()
coefficients = grid_search_lr.best_estimator_.coef_[0]

# Plot the feature importance
plt.bar(range(len(coefficients)), coefficients, tick_label=column_names)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance : Logistic Regression')
plt.show()

#Looking at the distribution of the variables

plt.hist(X_test['Bilirubin_total_missing_percent'], bins=100)
plt.title('Bilirubin Missing ration')
plt.show()

plt.hist(X_test['Platelets_missing_percent'], bins=100)
plt.title('Platelets Missing ratio')
plt.show()

np.percentile(X_test['Platelets_missing_percent'], 50)

plt.hist(X_test['ICULOS'], bins=100)
plt.title('Time in the hospital')
plt.show()


def exploaring_model_results(model, name):
    """
    The function will explore how the model performs across different groups of patients.
    :param model: the model which its results will be analyzed
    :return:
    """
    print(name)
    print('******************')
    # splitting the Bilirubin_total_missing_percent to two groups which the first one is patients how it was checked
    # and the second one is patients who their Bilirubin wasn't checked
    condition = X_test['Bilirubin_total_missing_percent'] == 1.00
    mask = condition
    prediction = model.predict(X_test[mask].iloc[:, chosen_columns])
    disp = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test[mask].iloc[:, chosen_columns],
        y_test[mask],
        cmap='plasma'
    )
    disp.ax_.set_title(name + ': Confusion Matrix for patients who their Bilirubin wasnt tested', fontsize=10)
    plt.tight_layout()
    plt.show()
    print('F1 for the group who their Bilirubin wasnt tested:', f1_score(y_test[mask], prediction))
    prediction = model.predict(X_test[~mask].iloc[:, chosen_columns])
    disp = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test[~mask].iloc[:, chosen_columns],
        y_test[~mask],
        cmap='plasma'

    )
    disp.ax_.set_title(name + ': Confusion Matrix for patients who their Bilirubin was tested', fontsize=10)
    plt.tight_layout()
    plt.show()
    print('F1 for the group who their Bilirubin was tested:', f1_score(y_test[~mask], prediction))

    # Split the ICULIOS varibale into two groups based on the median
    condition = X_test['ICULOS'] <= np.percentile(X_test['ICULOS'], 50)
    mask = condition
    prediction = model.predict(X_test[mask].iloc[:, chosen_columns])
    disp = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test[mask].iloc[:, chosen_columns],
        y_test[mask],
        cmap='plasma'
    )
    disp.ax_.set_title(name + ': Confusion Matrix for patients who been in the hospital more than 20 hours',
                       fontsize=10)
    plt.tight_layout()
    plt.show()
    print('-------------------------------------------')

    print('F1 for the group who been in the hospital more than 20 hours:', f1_score(y_test[mask], prediction))
    prediction = model.predict(X_test[~mask].iloc[:, chosen_columns])
    disp = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test[~mask].iloc[:, chosen_columns],
        y_test[~mask],
        cmap='plasma'

    )
    disp.ax_.set_title(name + ': Confusion Matrix for patients who been in the hospital less than 20 hours',
                       fontsize=10)
    plt.tight_layout()
    plt.show()
    print('F1 for the group who been in the hospital less than 20 hours:', f1_score(y_test[~mask], prediction))
    print('-------------------------------------------')


exploaring_model_results(grid_search_xgb.best_estimator_, 'XGBoost')

exploaring_model_results(rfc_model, 'Random Forest')

exploaring_model_results(grid_search_lr.best_estimator_, 'Logistic Regression')
