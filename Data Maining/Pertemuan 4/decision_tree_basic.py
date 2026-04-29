#IMPLEMENTASI DECISION TREE DENGAN SCIKIT-LEARN, LISTING 1: decision_tree_basic.py
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# 1. Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Split data
# Stratify=y memastikan distribusi kelas di data train dan test tetap seimbang
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. Buat model Decision Tree
dt = DecisionTreeClassifier(
    criterion='gini',            # 'gini' atau 'entropy'
    max_depth=3,                 # batasi kedalaman untuk hindari overfitting
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

# 4. Training
dt.fit(X_train, y_train)

# 5. Prediksi
y_pred = dt.predict(X_test)

# 6. Evaluasi
print(f"Akurasi: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 7. Visualisasi Pohon (Tambahan agar lebih jelas)
plt.figure(figsize=(12, 8))
tree.plot_tree(
    dt,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True
)
plt.title("Visualisasi Decision Tree pada Dataset Iris")
plt.show()