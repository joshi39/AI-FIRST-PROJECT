
# Import required libraries
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

# Loading the dataset
data = pd.read_csv("C:\\Users\\pk540\\Downloads\\heart.csv")

# Preprocessing the dataset
scaler = StandardScaler()
data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])

# Split the dataset into training and testing sets
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Logistic Regression classifier object
lr = LogisticRegression(random_state=42)

# Create a dictionary of hyperparameters to tune
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2']
}

# Using GridSearch to tune hyperparameters
grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# hyperparameters to train the model
lr_best = LogisticRegression(**grid_search.best_params_, random_state=42)
lr_best.fit(X_train, y_train)

# feature selection to select the most important features
sfm = SelectFromModel(lr_best, threshold='median')
sfm.fit(X_train, y_train)

# Transform the datasets
X_train_transformed = sfm.transform(X_train)
X_test_transformed = sfm.transform(X_test)

# Train the model on the transformed datasets
lr_best.fit(X_train_transformed, y_train)

# Make predictions on the testing set
y_pred = lr_best.predict(X_test_transformed)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)
sensitivity = recall_score(y_test, y_pred)

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Sensitivity:", sensitivity)
print("Precision:", precision)
print("F1 Score:", f1)
print("Specificity:", specificity)
print("AUC:", roc_auc_score(y_test, y_pred))

