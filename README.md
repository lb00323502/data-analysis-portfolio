# Raman Spectroscopy Simulation with PCA and LDA

"""
This notebook demonstrates a sample workflow for classifying spectral data (e.g., from Raman spectroscopy) using PCA for dimensionality reduction and LDA for classification.
Although the dataset is simulated, it reflects the general structure of real-world spectral signals.

Steps:
1. Generate synthetic Raman-like data with 100 features
2. Standardize features
3. Apply PCA and visualize the first two principal components
4. Apply LDA to classify the data
5. Evaluate model performance using classification report and confusion matrix
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 1. Generate synthetic dataset (simulate Raman spectral features)
X, y = make_classification(n_samples=300, n_features=100, n_informative=20, 
                           n_redundant=10, n_classes=2, random_state=42)

# 2. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply PCA to reduce to 2 components for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 4. Visualize PCA result
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='bwr', alpha=0.7)
plt.title("PCA Projection of Simulated Raman Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()

# 5. Split PCA data for classification
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# 6. Apply LDA classifier
lda = LDA()
lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)

# 7. Evaluation metrics
print("Classification Report:\n")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

