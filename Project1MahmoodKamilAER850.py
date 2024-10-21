import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import svm
import joblib

# Step 1: Data Loading and Preprocessing
def load_and_prepare_data(path, target_col="Step"):
    """Load the data, apply stratified sampling, and return training and test sets."""
    df = pd.read_csv(path)
    print(df.info())

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in splitter.split(df, df[target_col]):
        train_data = df.loc[train_idx].reset_index(drop=True)
        test_data = df.loc[test_idx].reset_index(drop=True)
    
    return train_data, test_data

# Separate Features and Target
def split_features_target(df, target_col="Step"):
    """Separate the features and target variable."""
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return X, y

# Data Visualization
def visualize_data(df):
    """Visualize data correlations and relationships."""
    pd.plotting.scatter_matrix(df)
    plt.figure()
    corr_matrix = df.corr()
    sns.heatmap(np.abs(corr_matrix), annot=True, cmap='coolwarm')
    plt.show()

# Data Scaling
def scale_features(X_train, X_test):
    """Apply feature scaling."""
    scaler = StandardScaler()
    scaler.fit(X_train.iloc[:, :-1])
    
    X_train_scaled = pd.DataFrame(scaler.transform(X_train.iloc[:, :-1]), columns=X_train.columns[:-1])
    X_test_scaled = pd.DataFrame(scaler.transform(X_test.iloc[:, :-1]), columns=X_test.columns[:-1])
    
    return X_train_scaled.join(X_train.iloc[:, -1:]), X_test_scaled.join(X_test.iloc[:, -1:])

# Model Evaluation
def evaluate_and_report_model(model, X_train, y_train, X_test, y_test, model_name):
    """Evaluate the model performance and print metrics."""
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred)}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=np.unique(y_test), columns=np.unique(y_test))
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_df, annot=True, cmap='coolwarm', cbar=False)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Logistic Regression Model
def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train and evaluate Logistic Regression with GridSearch."""
    param_grid = {'solver': ['lbfgs', 'newton-cg', 'sag'], 'penalty': [None, 'l2'], 'C': [0.01, 0.1, 1, 10]}
    model = LogisticRegression(multi_class='multinomial', random_state=42, max_iter=100)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    evaluate_and_report_model(best_model, X_train, y_train, X_test, y_test, "Logistic Regression")
    
    return best_model

# Support Vector Machine Model
def train_svm(X_train, y_train, X_test, y_test):
    """Train and evaluate SVM with GridSearch."""
    param_grid = {'gamma': ['scale', 'auto', 10, 100], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'C': [0.001, 0.01, 0.1, 1, 10]}
    model = svm.SVC(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    evaluate_and_report_model(best_model, X_train, y_train, X_test, y_test, "SVM")
    
    return best_model

# Random Forest Model
def train_random_forest(X_train, y_train, X_test, y_test):
    """Train and evaluate Random Forest with RandomizedSearch."""
    param_grid = {'n_estimators': [5, 10, 20, 50], 'max_depth': [None, 5, 10, 25], 'min_samples_split': [2, 5, 10],
                  'min_samples_leaf': [1, 2, 4], 'max_features': [None, 'sqrt', 'log2'], 'criterion': ['gini', 'entropy']}
    
    model = RandomForestClassifier(random_state=42)
    rand_search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, scoring='f1_weighted')
    rand_search.fit(X_train, y_train)
    
    best_model = rand_search.best_estimator_
    evaluate_and_report_model(best_model, X_train, y_train, X_test, y_test, "Random Forest")
    
    return best_model

# Stacking Classifier
def stacking_classifier(X_train, y_train, X_test, y_test, rf_model, svm_model):
    """Train and evaluate Stacking Classifier."""
    stacked_model = StackingClassifier(estimators=[('rf', rf_model), ('svm', svm_model)], cv=5)
    stacked_model.fit(X_train, y_train)

    evaluate_and_report_model(stacked_model, X_train, y_train, X_test, y_test, "Stacking Classifier")
    
    return stacked_model

# Save and Load Model
def save_model(model, filename):
    """Save model to disk."""
    joblib.dump(model, filename)

def load_model(filename):
    """Load model from disk."""
    return joblib.load(filename)

# Prediction on new data
def predict_new_data(model, new_data):
    """Make predictions on new data using a loaded model."""
    predictions = model.predict(new_data)
    print(f"New Data Predictions: {predictions}")
    return predictions

# Main Execution
if __name__ == '__main__':
    data_path = 'C:/Users/mahmo/Downloads/Project_1_Data.csv'
    
    # Load and preprocess data
    train_data, test_data = load_and_prepare_data(data_path)
    X_train, y_train = split_features_target(train_data)
    X_test, y_test = split_features_target(test_data)

    # Visualize data
    visualize_data(train_data)

    # Scale features
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # Train models
    lr_model = train_logistic_regression(X_train_scaled, y_train, X_test_scaled, y_test)
    svm_model = train_svm(X_train_scaled, y_train, X_test_scaled, y_test)
    rf_model = train_random_forest(X_train_scaled, y_train, X_test_scaled, y_test)

    # Stacking classifier
    stacked_model = stacking_classifier(X_train_scaled, y_train, X_test_scaled, y_test, rf_model, svm_model)

    # Save stacked model
    save_model(stacked_model, 'stacked_model.pkl')

    # Load model and predict new data
    loaded_model = load_model('stacked_model.pkl')
    new_data = pd.DataFrame([[9.375, 3.0625, 1.51], [6.995, 5.125, 0.3875], [0, 3.0625, 1.93]], columns=['X', 'Y', 'Z'])
    predict_new_data(loaded_model, new_data)
