
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

# Load dataset
data = pd.read_csv("C:/Users/hassa/Documents/GitHub/Projects/Project_1_Data.csv")
print(data.info())

# Perform stratified sampling based on income categories
data["income_groups"] = pd.cut(data["median_income"],
                               bins=[0, 2, 4, 6, np.inf],
                               labels=[1, 2, 3, 4])

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in splitter.split(data, data["income_groups"]):
    train_set = data.loc[train_idx].reset_index(drop=True)
    test_set = data.loc[test_idx].reset_index(drop=True)

# Drop income_groups after splitting
train_set = train_set.drop(columns=["income_groups"], axis=1)
test_set = test_set.drop(columns=["income_groups"], axis=1)

# Prepare training and testing sets
X_train = train_set.drop("Step", axis=1)
y_train = train_set["Step"]
X_test = test_set.drop("Step", axis=1)
y_test = test_set["Step"]

# Correlation matrix visualization
corr_matrix = X_train.iloc[:, :-5].corr()
plt.figure()
sns.heatmap(np.abs(corr_matrix), annot=False, cmap="coolwarm")

# Alternative correlation matrix visualization
corr_matrix_alt = X_train.iloc[:, :-5].corr()
plt.figure()
sns.heatmap(np.abs(corr_matrix_alt), annot=False, cmap="magma")

# Train first model: Linear Regression
# model1 = LinearRegression()
# model1.fit(X_train, y_train)
# predictions_train1 = model1.predict(X_train)

# for i in range(5):
#     print(f"Predicted: {predictions_train1[i]:.2f}, Actual: {y_train.iloc[i]:.2f}")

# mae_train1 = mean_absolute_error(y_train, predictions_train1)
# print(f"Linear Regression Training MAE: {mae_train1:.2f}")

# Cross-validation for first model
# cv_scores = cross_val_score(model1, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
# cv_mae = -cv_scores.mean()
# print(f"Cross-validated MAE (Linear Regression): {cv_mae:.2f}")
# # NOTE: Potential data leakage in CV implementation.

# Train second model: Random Forest Regressor
# model2 = RandomForestRegressor(n_estimators=30, random_state=42)
# model2.fit(X_train, y_train)
# predictions_train2 = model2.predict(X_train)
# mae_train2 = mean_absolute_error(y_train, predictions_train2)
# print(f"Random Forest Training MAE: {mae_train2:.2f}")

# Compare predictions of both models
# for i in range(5):
#     print(f"Linear Model Prediction: {round(predictions_train1[i], 2)}, "
#           f"Random Forest Prediction: {round(predictions_train2[i], 2)}, "
#           f"Actual Value: {round(y_train.iloc[i], 2)}")

# Hyperparameter tuning using GridSearchCV
# param_grid = {
#     'n_estimators': [10, 30, 50],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['sqrt', 'log2']
# }
# model3 = RandomForestRegressor(random_state=42)
# grid_search = GridSearchCV(model3, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
# grid_search.fit(X_train, y_train)

# Best model selection
# best_params = grid_search.best_params_
# print("Optimal Parameters:", best_params)
# best_model3 = grid_search.best_estimator_
