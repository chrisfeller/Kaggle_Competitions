import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plotting import eda_plots, plot_correlation_matrix, distribution_plot
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

def load_data(train=True):
    """
    Load Titanic dataset from .csv file.

    Args:
        train: True if training data; False if testing data.

    Returns:
        Titanic dataset
    """
    if train:
        df = pd.read_csv('data/train.csv')
        # Shuffle data
        df = df.iloc[np.random.permutation(df.shape[0])]
    else:
        df = pd.read_csv('data/test.csv')
        # Shuffle data
        df = df.iloc[np.random.permutation(df.shape[0])]
    return df

def clean_data(train, test):
    """
    Clean data by imputing for null values, turning categorical features into
    dummy variables, and standardizing continuous features.

    Args:
        train: training dataframe
        test: test dataframe

    Returns:
        train: training dataframe
        test: test dataframe
    """
    train, test = handle_nulls(train, test)
    train, test = make_dummies(train, test)
    train, test = standardize_df(train, test)
    return train, test

def handle_nulls(train, test):
    """
    Fill missing data in Age feature with median age of all passengers.
    Drop Cabin feature because of high-number of nulls (77.2% of feature)
    Fill missing data in Embarked features withmode ('S').

    Args:
        train: training dataframe
        test: test dataframe

    Returns:
        train: training dataframe
        test: test dataframe
    """
    test['Fare'].fillna(train['Fare'].median(), inplace=True)
    test['Age'].fillna(train['Age'].median(), inplace=True)
    test['Embarked'].fillna('S')
    test.drop('Cabin', axis=1, inplace=True)

    train['Fare'].fillna(train['Fare'].median(), inplace=True)
    train['Age'].fillna(train['Age'].median(), inplace=True)
    train['Embarked'].fillna('S')
    train.drop('Cabin', axis=1, inplace=True)

    return train, test

def make_dummies(train, test):
    """
    Make dummy columns for categorical varfiables.

    Args:
        train: training dataframe
        test: test dataframe

    Returns:
        train: training dataframe
        test: test dataframe
    """
    to_dummy = ['Sex', 'Embarked', 'Pclass']
    for col in to_dummy:
        train = pd.concat([train, pd.get_dummies(train[col], prefix=col,
                        drop_first=True)], axis=1)
        test = pd.concat([test, pd.get_dummies(test[col],
                        prefix=col, drop_first=True)], axis=1)
        train.drop(col, inplace=True, axis=1)
        test.drop(col, inplace=True, axis=1)
    return train, test

def standardize_df(train, test):
    """
    Standardize Fare, Age, SibSp, Parch features.

    Args:
        train: training dataframe
        test: test dataframe

    Returns:
        train: training dataframe
        test: test dataframe
    """
    to_standardize = ['Fare', 'Age', 'SibSp', 'Parch']
    scaler = StandardScaler()
    train[to_standardize] = scaler.fit_transform(train[to_standardize])
    test[to_standardize] = scaler.transform(test[to_standardize])
    return train, test

def family_size(train, test):
    """
    Combine SibSp (Number of siblings/spouses aboard) and Parch
    (Number of parents/children aboard) into one feature Family_Size. Then
    create dummy variables for Famkily_Size.

    Args:
        train: training dataframe
        test: test dataframe

    Returns:
        train: training dataframe
        test: test dataframe
    """
    for data in [train, test]:
        data['Family_Size'] = data['SibSp'] + data['Parch'] + 1
        data['Single'] = data['Family_Size'].map(lambda x: 1 if x == 1 else 0)
        data['Small_Family'] = data['Family_Size'].map(lambda x: 1 if x == 2 else 0)
        data['Medium_Family'] = data['Family_Size'].map(lambda x: 1 if 3 <= x <=4 else 0)
        data['Large_Family'] = data['Family_Size'].map(lambda x: 1 if x >= 5 else 0)
    return train, test

def title_extraction(train, test):
    """
    Extract the title from each passenger's name. Create dummy variables for the
    top three most common, Mr, Mrs, and Miss, in addition to an 'Other' category.

    Args:
        train: training dataframe
        test: test dataframe

    Returns:
        train: training dataframe
        test: test dataframe
    """

    for data in [train, test]:
        data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        data['Title'] = data["Title"].replace(['Dr', 'Rev', 'Mlle', 'Col',
                        'Major', 'Lady', 'Capt', 'Sir', 'Mme', 'Ms', 'Jonkheer',
                         'Don', 'Countess'], 'Other')
    train = pd.concat([train, pd.get_dummies(train['Title'], prefix='Title',
                                            drop_first=True)], axis=1)
    test = pd.concat([test, pd.get_dummies(test['Title'], prefix='Title',
                                            drop_first=True)], axis=1)
    train.drop('Title', inplace=True, axis=1)
    test.drop('Title', inplace=True, axis=1)
    return train, test

def plot_cross_validation(X, y, models, scoring):
    """
    Return 10-Fold Cross Validation scores for various models in addition to
    box plots for each of the 10 fold models.

    Args:
        X: Feature matrix
        y: Target vector
        models: Dictionary of models with the model name as the key and the
        instantiated model as the value.
        scoring: Str of the scoring to use (i.e., 'accuracy')

    Returns:
        Scores: 10-Fold Cross Validation scores for all models.
        Plot: Boxplot of all 10-fold model scores.
    """
    seed = 123
    results = []
    names = []
    all_scores = []
    print('Mod - Avg - Std Dev')
    print('---   ---   -------')
    for name, model in models.items():
        kfold = KFold(n_splits=10, random_state=seed)
        cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        all_scores.append(cv_results.mean())
        print('{}: {:.2f} ({:2f})'.format(name, cv_results.mean(), cv_results.std()))
    print('Avg of all: {:.3f}'.format(np.mean(all_scores)))
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle('Algorithm Comparison of CrossVal Scores')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names, rotation=20, fontsize=10)
    ax.set_ylim([0.5,1])
    ax.set_ylabel('K-Fold CV Accuracy')
    ax.set_xlabel('Model')
    plt.tight_layout()
    plt.show()

def test_params(model, train_X, train_y, param_list, scoring):
    """
    Use grid search to discover optimal parameters for each tested model

    Args:
        model: fitted model
        train_X: training data containing all features
        train_y: training data containing target
        param_list: dictionary of parameters to test and test values
                    (e.g., {'alpha': np.logspace(-1, 1, 50)})

    Returns :
        Best parameter for the model and its score
    """
    g = GridSearchCV(model, param_list, scoring=scoring, cv=10, n_jobs=-1, verbose=1)
    g.fit(train_X, train_y)
    print('Model: {}, Best Params: {}, Best Score: {}'\
        .format(model, g.best_params_, g.best_score_))


if __name__=='__main__':
    # Load Data
    train = load_data(train=True)
    test = load_data(train=False)

    # EDA Plots
    eda_plots(train)

    # Clean Data
    train, test = clean_data(train, test)

    # Correlation Matrix and Distribution Plot
    plot_correlation_matrix(train)
    distribution_plot(train)

    # Feature Engineering
    train, test = family_size(train, test)
    train, test = title_extraction(train, test)

    # Train/Test Split Training Data
    X = train[['Age', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S', 'Pclass_2',
                'Pclass_3', 'Single', 'Small_Family', 'Medium_Family', 'Large_Family',
                'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Other']]
    y = train['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state=0)

    # Final Model
    rf = RandomForestClassifier(bootstrap=True, criterion='gini', max_depth=None,
                                max_features=8, min_samples_leaf=2,
                                min_samples_split=2, n_estimators=200)
    rf.fit(X, y)

    # Submission
    X_test = test[['Age', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S',
                    'Pclass_2', 'Pclass_3', 'Single', 'Small_Family',
                    'Medium_Family', 'Large_Family', 'Title_Miss', 'Title_Mr',
                    'Title_Mrs', 'Title_Other']]
    final_predictions = rf.predict(X_test)
    submission = pd.concat([test['PassengerId'], pd.Series(final_predictions)], axis=1)
    submission.columns = ['PassengerId', 'Survived']
    submission.to_csv('submission.csv', index=False)


    # Cross Validate/Test All Models
    models = {'Logistic Regression': LogisticRegression(), 'K-Nearest Neighbors':
    KNeighborsClassifier(),'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': GaussianNB(), 'Random Forest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(), 'Gradient Boosting': GradientBoostingClassifier(),
    'SVC': SVC()}
    plot_cross_validation(X_train, y_train, models, 'accuracy')

    # Random Forest
    param_list = {"max_depth": [None],
              "max_features": np.arange(1,9, 1),
              "min_samples_split": np.arange(2,100,10),
              "min_samples_leaf": np.arange(2,100, 100),
              "bootstrap": [True, False],
              "n_estimators" :[100,200, 300],
              "criterion": ["gini"]}
    # test_params(RandomForestClassifier(), X_train, y_train, param_list, 'accuracy')
    # Best Params: {'bootstrap': True, 'criterion': 'gini', 'max_depth': None,
    # 'max_features': 8, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 200}

    # Gradient Boosting
    param_list = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1]
              }
    # test_params(GradientBoostingClassifier(), X_train, y_train, param_list, 'accuracy')
    # Best Params: {'learning_rate': 0.1, 'loss': 'deviance', 'max_depth': 4,
    #'max_features': 0.3, 'min_samples_leaf': 100, 'n_estimators': 300}

    # SVC
    param_list = {'kernel': ['rbf'],
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}
    # test_params(SVC(), X_train, y_train, param_list, 'accuracy')
    # Best Params: {'C': 50, 'gamma': 0.01, 'kernel': 'rbf'}
