import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fancyimpute import KNN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

# Plotting Style
plt.style.use('fivethirtyeight')

def load_data():
    """
    Loads the training and testing data from either respective .csv files in
    the data directory

    Args:
        None

    Returns:
        train: Training dataset
        test: testing dataset
    """
    train = pd.read_csv('data/train.csv', index_col=0)
    test = pd.read_csv('data/test.csv', index_col=0)
    return train, test

def clean_data(df, dummy=True):
    """
    Removes all features made up of 25% or more null values. Utilizes
    fancyimpute's KNN to impute missing values in all continuous features. Imputes the
    mode for all missing values in categorical features. Creates dummy variables
    for all categorical features.

    Args:
        df: Dataframe of combined train and test datasets.
        dummy: True if categorical variables are to be dummied. False to
        leave categorical variables as is for plotting purposes.

    Returns:
        df: Dataframe of combined train and test datasets.
    """
    # Remove ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'] features
    # as they are made up of >= 25 % null values
    df = df[df.columns[df.isnull().mean() < .25]]

    # Impute continuous features using fancy impute's KNN method with k=3
    continuous_nans = ['LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
                        'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',
                        'GarageArea']
    df[continuous_nans] = KNN(k=3).complete(df[continuous_nans])

    # Impute mode for all missing values in categorical features.
    # Converts 'GarageYrBlt' and 'GarageCars' to object type as to avoid imputing
    # float value. Converts both features back to int type.
    df['GarageCars'] = df['GarageCars'].astype(object)
    df['GarageYrBlt'] = df['GarageYrBlt'].astype(object)
    fill = pd.Series([df[i].value_counts().index[0] if df[i].dtype == np.dtype('O')
        else df[i] for i in df], index=df.columns)
    df = df.fillna(fill)
    df['GarageYrBlt'] = df['GarageYrBlt'].astype(int)
    df['GarageCars'] = df['GarageCars'].astype(int)

    # Convert categorical features to dummy variables
    if dummy==True:
        df = pd.get_dummies(df, drop_first=True)
    return df

def scale_date(train, test):
    """
    Scales all continuous features.

    Args:
        train: unscaled training data
        test: unscaled testing data

    Returns:
        train: scaled training data
        test: scaled testing data
    """
    scaler = StandardScaler()
    continuous_features = [col for col in train.select_dtypes(include='int64')]
    train[continuous_features] = scaler.fit_transform(train[continuous_features])
    test[continuous_features] = scaler.transform(test[continuous_features])
    return train, test

def model_selection(train):
    """
    Performs a test/train split on the training data. Gridsearches over Lasso,
    Ridge, ElasticNet, and XGBoost models. Run top-performing model (XGBoost) on
    validation set.

    Args:
        train: training dataset

    Returns:
        Root-Mean-Squared Error (RMSE) between the logarithm of the predicted
        value and the logarithm of the observed sales price of the validation
        data set using the best gridsearch model (XGBoost).
    """
    # Test/Train split training data
    y = train['SalePrice']
    X = train.drop('SalePrice', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Gridsearch Lasso Model
    lasso = Lasso()
    param_list = {'alpha': [0.5, 0.1, 0.001, 0.0001]}
    lasso_grid = GridSearchCV(lasso, param_list, scoring='neg_mean_squared_error',
                     cv=5)
    lasso_grid.fit(X_train, y_train)
    print('Model: {}, Best Params: {}, Best Score: {}'\
        .format(lasso, lasso_grid.best_params_, np.sqrt(abs(lasso_grid.best_score_))))

    # Gridsearch Ridge Model
    ridge = Ridge()
    param_list = {'alpha': [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75],
                    'solver': ['auto', 'svd', 'lsqr', 'cholesky']}
    ridge_grid = GridSearchCV(ridge, param_list, scoring='neg_mean_squared_error',
                     cv=5)
    ridge_grid.fit(X_train, y_train)
    print('Model: {}, Best Params: {}, Best Score: {}'
        .format(ridge, ridge_grid.best_params_, np.sqrt(abs(ridge_grid.best_score_))))

    # Gridsearch ElasticNet
    elastic = ElasticNet()
    param_list = {'alpha': np.linspace(0.1, 1.0, 20),
                  'l1_ratio': np.linspace(0.1, 1.0, 20)}
    elastic_grid = GridSearchCV(elastic, param_list, scoring='neg_mean_squared_error',
                     cv=5)
    elastic_grid.fit(X_train, y_train)
    print('Model: {}, Best Params: {}, Best Score: {}'\
        .format(elastic, elastic_grid.best_params_, abs(elastic_grid.best_score_)))

    # Gridsearch XGBoost
    xgb = XGBRegressor()
    param_list = {'max_depth':[2, 4, 6], 'min_child_weight': [1, 3, 5],
                'n_estimators':[200, 600, 1000], 'learning_rate':[ 0.3 ,
                0.24444444, 0.18888889, 0.13333333, 0.07777778,  0.05], 'reg_alpha':
                [1e-5, 0.1, 100]}
    xgb_grid = GridSearchCV(xgb, param_list, scoring='neg_mean_squared_error',
                     cv=5)
    xgb_grid.fit(X_train, y_train)
    print('Model: {}, Best Params: {}, Best Score: {}'\
        .format(xgb, xgb_grid.best_params_, abs(xgb_grid.best_score_)))

    # Final XGBoost model on validation data set
    xgb = XGBRegressor(learning_rate=0.05, max_depth=2, min_child_weight=3,
                        n_estimators=1000, reg_alpha=0.1)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    log_diff = np.log(np.expm1(y_pred) + 1) - np.log(np.expm1(y_test) + 1)
    score = np.sqrt(np.mean(log_diff**2))
    print(score)
    # 0.163980400948

if __name__=='__main__':
    train, test = load_data()

    # Clean Data
    train_saleprice = train.pop('SalePrice')
    cutoff = len(train)
    # Temporarily combine training and test data to more easily clean both
    combined = pd.concat(objs=[train, test], axis=0)
    combined = clean_data(combined)
    # Split training and test data
    train = combined[:cutoff]
    test = combined[cutoff:]

    #Scale Continuous Features
    train, test = scale_date(train, test)

    # Add saleprice back to train
    train.insert(0, column='SalePrice', value=train_saleprice)

    # Log-Transform target feature 'SalePrice'
    train['SalePrice'] = np.log1p(train["SalePrice"])

    # Model Selection
    model_selection(train)
    # Lasso: Best Params: {'alpha': 0.001},
        # Best Score: Best Score: 0.163827
    # Ridge: Best Params: {'alpha': 5, 'solver': 'svd'},
        # Best Score: 0.159123
    # ElasticNet: Best Params: Best Params: {'alpha': 0.1, 'l1_ratio': 0.1},
        # Best Score: 0.130522
    # XGBoost: Best Params: {'learning_rate': 0.05, 'max_depth': 2,
        # 'min_child_weight': 3, 'n_estimators': 1000, 'reg_alpha': 0.1},
        # Best Score: 789749045.7137355

    # Final Model
    y = train['SalePrice']
    X = train.drop('SalePrice', axis=1)
    xgb = XGBRegressor(learning_rate=0.05, max_depth=2, min_child_weight=3,
                        n_estimators=1000, reg_alpha=0.1)
    xgb.fit(X, y)
    final_predictions = np.expm1(xgb.predict(test))
    submission = pd.concat([pd.Series(test.index),
                    pd.Series(final_predictions)], axis=1)
    submission.columns = ['Id', 'SalePrice']
    submission.to_csv('data/submission.csv', index=False)
