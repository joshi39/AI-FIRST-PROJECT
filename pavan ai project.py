# Import required libraries
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

# Loading the dataset
data = pd.read_csv("C:\\Users\\PAVAN SAITEJA\\OneDrive\\Desktop\\heart.csv")

# Split the dataset into features and target variable
X = data.drop('target', axis=1)
y = data['target']

# Scale the data to have mean of 0 and variance of 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the scaled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create a Decision Tree classifier object
dtc = DecisionTreeClassifier(random_state=42)

# Create a dictionary of hyperparameters to tune
param_grid = {
    'max_depth': [3, 5, 7, 10, 15],
    'min_samples_split': [2, 5, 10, 15],
    'max_features': ['sqrt', 'log2']
}

# Using GridSearch to tune hyperparameters
grid_search = GridSearchCV(estimator=dtc, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# hyperparameters to train the model
dtc_best = DecisionTreeClassifier(**grid_search.best_params_, random_state=42)
dtc_best.fit(X_train, y_train)

# feature selection to select the most important features
sfm = SelectFromModel(dtc_best, threshold='median')
sfm.fit(X_train, y_train)

# Transform the datasets
X_train_transformed = sfm.transform(X_train)
X_test_transformed = sfm.transform(X_test)

# Train the model on the transformed datasets
dtc_best.fit(X_train_transformed, y_train)

# Make predictions on the testing set
y_pred = dtc_best.predict(X_test_transformed)

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
