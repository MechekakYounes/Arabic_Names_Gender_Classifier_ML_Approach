import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from matplotlib.patches import Patch
from sklearn.feature_extraction.text import CountVectorizer


df = pd.read_excel("all_arabic_names_preprocessed.xlsx")
df = df.dropna(subset=['Name', 'Gender'])
print(df.duplicated().sum(), "duplicate rows found.")

X_names = df['Name'].astype(str).reset_index(drop=True)
y_labels = df['Gender'].astype(str).reset_index(drop=True)

vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 4))
X_vectors = vectorizer.fit_transform(X_names)

n_males = sum(y_labels == 'male')
n_females = sum(y_labels == 'female')
print(f"Number of males is {n_males}, Number of females is {n_females}")

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_vectors.toarray())

colors = ['red' if g == 'female' else 'blue' if g == 'male' else 'yellow' for g in y_labels]

plt.figure(figsize=(10, 6))
plt.scatter(X_reduced[:,0], X_reduced[:, 1], c=colors, alpha=0.3, s=20)

plt.title("Arabic Name Vectors by Gender (Count Vectorizer)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

legend_elements = [
    Patch(facecolor='red', label='Female'),
    Patch(facecolor='blue', label='Male')
]
plt.legend(handles=legend_elements)
plt.grid(True)
plt.show()
plt.savefig("names_gender_pca_visualization.png", dpi=300)
