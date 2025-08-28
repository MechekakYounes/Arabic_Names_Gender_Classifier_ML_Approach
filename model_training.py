import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import re

def normalize_name(name):
    
    # Take only the first word
    name = name.split()[0]
    # Ensure name is a string and strip whitespace
    name = str(name).strip()

    # Remove diacritics
    diacritics_pattern = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    name = re.sub(diacritics_pattern, '', name)

    # Remove Tatweel (ــ)
    name = re.sub(r'ـ', '', name)

    # Unify Arabic letters
    name = name.replace("ى", "ي")
    name = name.replace("ة", "ه")
    name = name.replace("ئ", "ي")
    name = name.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")

    # Remove numbers and special characters (keep Arabic and English letters and spaces)
    name = re.sub(r'[^ء-ي\s]', '', name)

    # Remove extra spaces
    name = re.sub(r'\s+', ' ', name)

    

    return name


# Load the mapping file and normalize names
mapping_df = pd.read_excel("mapping_file.xlsx")
mapping_df["Name"] = mapping_df["Name"].apply(normalize_name)
#unifying the gender labels and remving any leading/trailing spaces 
mapping_df["Gender"] = mapping_df["Gender"].str.strip().str.lower()
mapping_df = mapping_df.dropna(subset=["Gender"])



X_names = mapping_df["Name"]
y_labels = mapping_df["Gender"]

# Ensure labels and names are in string format
y_labels = y_labels.astype(str)
X_names = X_names.astype(str)

# Vectorize names using TF-IDF with character n-grams     
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4)) 
X_vectors = vectorizer.fit_transform(X_names)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectors, y_labels, test_size=0.2, random_state=42)

# Hyperparameter tuning using GridSearchCV
logreg = LogisticRegression(solver="liblinear", max_iter=100)

param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],  # Regularization strength
    "penalty": ["l1", "l2"]        # Regularization type
}

grid = GridSearchCV(logreg, param_grid, cv=5, scoring="accuracy")
grid.fit(X_train, y_train)
#Evaluate the best model from grid search
enhanced_model = grid.best_estimator_
y_pred = enhanced_model.predict(X_test)

print("✅ Best Parameters:", grid.best_params_)
print("✅ Training Accuracy:", grid.best_score_)
print("✅ Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model & vectorizer locally
joblib.dump(enhanced_model, "gender_model_enhanced.pkl")
print("Model is successfully done!")
joblib.dump(vectorizer, "vectorizer_enhanced.pkl")
print("Vectorizer is successfully done!")