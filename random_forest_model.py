from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, make_scorer,f1_score, precision_score, recall_score, confusion_matrix
import pandas as pd
import seaborn as sns
import joblib
from sklearn.feature_extraction.text import CountVectorizer
df = pd.read_excel("all_arabic_names_preprocessed.xlsx")

X = df['Name']
y = df['Gender']

y_labels = y.astype(str)
X_names = X.astype(str)

vectorizer = CountVectorizer(analyzer='char', ngram_range=(1,3))
X_vectors = vectorizer.fit_transform(X_names)   
X_train, X_test, y_train, y_test = train_test_split(X_vectors, y_labels, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)


print("Test Accuracy:", accuracy_score(y_test, y_pred))
