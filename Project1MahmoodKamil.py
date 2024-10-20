import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import svm

# STEP 1: Data Loading and Preprocessing
def load_data(path):
    df = pd.read_csv(path)
    print(df.info())
    return df

# Stratified Sampling for train/test split
def stratified_sampling(df, target_col="Step"):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=69)
    for train_idx, test_idx in splitter.split(df, df[target_col]):
        strat_train = df.loc[train_idx].reset_index(drop=True)
        strat_test = df.loc[test_idx].reset_index(drop=True)
    return strat_train, strat_test

# Feature and target separation
def separate_features_target(df, target_col="Step"):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return X, y

# STEP 2: Data Visualization
def visualize_data(df):
    pd.plotting.scatter_matrix(df)
    plt.figure()
    corr_matrix = df.corr()
    sns.heatmap(np.abs(corr_matrix), annot=True, cmap='coolwarm')
    plt.show()

# STEP 3: Data Scaling
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train.iloc[:, :-1])  # Avoid scaling the target column
    X_train_scaled = pd.DataFrame(scaler.transform(X_train.iloc[:, :-1]), columns=X_train.columns[:-1])
    X_test_scaled = pd.DataFrame(scaler.transform(X_test.iloc[:, :-1]), columns=X_test.columns[:-1])
    return X_train_scaled.join(X_train.iloc[:, -1:]), X_test_scaled.join(X_test.iloc[:, -1:])

# STEP 4: Model Training and Evaluation

# Logistic Regression with GridSearchCV
def logistic_regression(X_train, y_train, X_test, y_test):
    param_grid = {'solver': ['lbfgs', 'newton-cg', 'sag'], 'penalty': [None, 'l2'], 'C': [0.01, 0.1, 1, 10]}
    model = LogisticRegression(multi_class='multinomial', random_state=69, max_iter=100)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted', n_jobs=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    evaluate_model(best_model, X_train, y_train, X_test, y_test, 'Logistic Regression')

# Support Vector Machine with GridSearchCV
def support_vector_machine(X_train, y_train, X_test, y_test):
    param_grid = {'gamma': ['scale', 'auto', 10, 100], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'C': [0.001, 0.01, 0.1, 1, 10]}
    model = svm.SVC(random_state=69)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted', n_jobs=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    evaluate_model(best_model, X_train, y_train, X_test, y_test, 'SVM')

# Random Forest with RandomizedSearchCV
def random_forest(X_train, y_train, X_test, y_test):
    param_grid = {'n_estimators': [5, 10, 15, 20, 25, 30, 50], 'max_depth': [None, 5, 10, 15, 25, 45],
                  'min_samples_split': [2, 5, 10, 15], 'min_samples_leaf': [1, 2, 4, 6, 12],
                  'max_features': [None, 0.1, 'sqrt', 'log2'], 'criterion': ['gini', 'entropy']}
    model = RandomForestClassifier(random_state=69)
    rand_search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, scoring='f1_weighted', n_jobs=1)
    rand_search.fit(X_train, y_train)

    best_model = rand_search.best_estimator_
    evaluate_model(best_model, X_train, y_train, X_test, y_test, 'Random Forest')

# STEP 5: Model Evaluation
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy Score (Test): {accuracy_score(y_test, y_pred)}")
    print(f"Accuracy Score (Train): {accuracy_score(y_train, y_pred_train)}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=np.unique(y_test), columns=np.unique(y_test))
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_df, annot=True, cmap='coolwarm', cbar=False)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

# MAIN EXECUTION
data_path = 'C:/Users/mahmo/Downloads/Project_1_Data.csv'
df = load_data(data_path)

# Stratified sampling
strat_train, strat_test = stratified_sampling(df)

# Feature and target separation
X_train, y_train = separate_features_target(strat_train)
X_test, y_test = separate_features_target(strat_test)

# Data visualization
visualize_data(strat_train)

# Data scaling
X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

# Train and evaluate models
logistic_regression(X_train_scaled, y_train, X_test_scaled, y_test)
support_vector_machine(X_train_scaled, y_train, X_test_scaled, y_test)
random_forest(X_train_scaled, y_train, X_test_scaled, y_test)