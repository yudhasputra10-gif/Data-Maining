#pertemuan 3 LATIHAN: KLASIFIKASI SEDERHANA DENGAN K-NN
import numpy as np
from collections import Counter

def knn_predict(X_train, y_train, X_test, k=3):
    """ Prediksi dengan k-NN menggunakan Euclidean distance """
    predictions = []

    for test_point in X_test:
        # 1. Hitung jarak ke semua training point
        distances = []
        for i, train_point in enumerate(X_train):
            # Rumus Euclidean Distance: akar dari jumlah kuadrat selisih
            dist = np.sqrt(np.sum((test_point - train_point) ** 2))
            distances.append((dist, y_train[i]))

        # 2. Urutkan berdasarkan jarak terkecil
        distances.sort(key=lambda x: x[0])

        # 3. Ambil k tetangga terdekat
        k_neighbors = distances[:k]
        k_labels = [label for _, label in k_neighbors]

        # 4. Majority voting (Suara terbanyak)
        most_common = Counter(k_labels).most_common(1)[0][0]
        predictions.append(most_common)

    return np.array(predictions)

# --- Contoh Penggunaan ---
# Fitur: [Sepal Length, Sepal Width]
X_train = np.array([[5.1, 3.5], [4.9, 3.0], [6.2, 3.4], [5.9, 3.0]])
y_train = np.array([0, 0, 1, 1]) # 0=Setosa, 1=Versicolor

X_test = np.array([[5.0, 3.2], [6.0, 3.2]])

# Menjalankan fungsi prediksi
predictions = knn_predict(X_train, y_train, X_test, k=3)

print(f"Hasil prediksi k-NN: {predictions}")
# Output yang diharapkan: [0 1]
