#tugas rumah pertemuan 5
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from collections import Counter

# 1. Fungsi Jarak Euclidean
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# 2. Implementasi Kelas k-NN
class KNNFromScratch:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        # Hitung jarak antara titik x dengan semua titik di data training
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # Ambil indeks k tetangga terdekat
        k_indices = np.argsort(distances)[:self.k]

        # Ambil label dari k tetangga tersebut
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Voting suara terbanyak
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# --- Persiapan Data Iris ---
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalisasi (Sangat penting untuk k-NN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

k_list = [1, 3, 5, 7, 9, 11]
results = []

for k in k_list:
    # Model dari Scratch
    knn_scratch = KNNFromScratch(k=k)
    knn_scratch.fit(X_train, y_train)
    pred_scratch = knn_scratch.predict(X_test)
    acc_scratch = accuracy_score(y_test, pred_scratch)

    # Model Scikit-Learn
    knn_sklearn = KNeighborsClassifier(n_neighbors=k)
    knn_sklearn.fit(X_train, y_train)
    pred_sklearn = knn_sklearn.predict(X_test)
    acc_sklearn = accuracy_score(y_test, pred_sklearn)

    results.append({
        'k': k,
        'Akurasi Scratch': acc_scratch,
        'Akurasi Sklearn': acc_sklearn
    })

# Tampilkan Hasil
df_res = pd.DataFrame(results)
print(df_res.to_string(index=False))
