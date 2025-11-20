import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer




df = pd.read_excel("all_arabic_names_with_duplicates.xlsx")

X = df['Name']
y = df['Gender']


y_labels = y.astype(str)
X_names = X.astype(str)

vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3))
X_vectors = vectorizer.fit_transform(X_names)
X_train, X_test, y_train, y_test = train_test_split(X_vectors, y_labels, test_size=0.25, random_state=42)

# Hyperparameter tuning using GridSearchCV
logreg = LogisticRegression(solver="liblinear", max_iter=50)

param_grid = {
    "C": [0.001,0.01, 0.1, 1, 10, 100], 
    "penalty": ["l1", "l2"]        
}

grid = GridSearchCV(logreg, param_grid, cv=5, scoring="accuracy")
grid.fit(X_train, y_train)

enhanced_model = grid.best_estimator_
preds = enhanced_model.predict(X_test)

print("Best Parameters:", grid.best_params_)
print("Training Accuracy:", grid.best_score_)
print("Test Accuracy:", accuracy_score(y_test, preds))
print("\nClassification Report:\n", classification_report(y_test, preds))
