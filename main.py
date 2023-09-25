import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

# Load the Iris dataset
iris_data = load_iris()

# Split features and target
X = iris_data.data
y = iris_data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #if you run your code now and then later again with the same random_state value of 42, you'll get the same split of data into training and testing sets. This helps in debugging, sharing code, and ensuring consistent comparisons of different models.

# Standardize the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train a Logistic Regression model
logreg_model = LogisticRegression(max_iter=1000)
logreg_model.fit(X_train_scaled, y_train)

# Predict using the model
logreg_predictions = logreg_model.predict(X_test_scaled)

# Calculate accuracy
logreg_accuracy = accuracy_score(y_test, logreg_predictions)
print("Logistic Regression Accuracy:", logreg_accuracy)

# Print classification report
print("Logistic Regression Classification Report:\n", classification_report(y_test, logreg_predictions, target_names=iris_data.target_names))

# Create and train a K-Nearest Neighbors model
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train_scaled, y_train)

# Predict using the model
knn_predictions = knn_model.predict(X_test_scaled)

# Calculate accuracy
knn_accuracy = accuracy_score(y_test, knn_predictions)
print("K-Nearest Neighbors Accuracy:", knn_accuracy)

# Print classification report
print("K-Nearest Neighbors Classification Report:\n", classification_report(y_test, knn_predictions, target_names=iris_data.target_names))


'''
note:
Precision: The ratio of correctly predicted instances of a class to the total predicted instances of that class. It measures the accuracy of positive predictions.

Recall: The ratio of correctly predicted instances of a class to the total actual instances of that class. It measures the ability of the model to capture all positive instances.

F1-Score: The harmonic mean of precision and recall. It provides a balanced measure between precision and recall.

Support: The number of actual occurrences of the class in the test dataset.
'''