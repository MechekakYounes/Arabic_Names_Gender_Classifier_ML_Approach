from sklearn import svm
import pandas as pd 
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_excel("all_arabic_names_with_duplicates.xlsx")
X = df['Name']
y = df['Gender']

y_labels = y.astype(str)
X_names = X.astype(str)

vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 4))
X_vectors = vectorizer.fit_transform(X_names)
X_train, X_test, y_train, y_test = train_test_split(X_vectors, y_labels, test_size=0.2, random_state=42)

# Hyperparameter tuning using GridSearchCV
svm_model = svm.SVC(probability=True)
param_grid = {
    "C": [0.1, 1, 10, 100], 
    "kernel": ["linear", "rbf", "poly"],        
    "gamma": ["scale", "auto"]  
}
grid = GridSearchCV(svm_model, param_grid, cv=5, scoring="accuracy")
grid.fit(X_train, y_train)

svm_model = grid.best_estimator_
y_pred = svm_model.predict(X_test)

print("✅ Best Parameters:", grid.best_params_)
print("✅ Training Accuracy:", grid.best_score_)
print("✅ Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model & vectorizer locally
joblib.dump(svm_model, "svm_gender_model.pkl")
print("SVM Model is successfully done!")
joblib.dump(vectorizer, "svm_count_vectorizer.pkl")
print("SVM Vectorizer is successfully done!")


