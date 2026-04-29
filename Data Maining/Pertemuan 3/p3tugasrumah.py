#tugas rumah pertemuan 3
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Fungsi Jarak Manual
def calculate_distances(p1, p2, p=2):
    p1, p2 = np.array(p1), np.array(p2)

    # 1. Euclidean (Minkowski dengan p=2)
    euclidean = np.sqrt(np.sum((p1 - p2)**2))

    # 2. Manhattan (Minkowski dengan p=1)
    manhattan = np.sum(np.abs(p1 - p2))

    # 3. Minkowski (dengan p kustom, misal p=3)
    minkowski = np.sum(np.abs(p1 - p2)**p)**(1/p)

    return euclidean, manhattan, minkowski

# Load Dataset Iris
iris = load_iris()
X = iris.data  # 4 fitur numerik
y = iris.target

# Normalisasi sangat penting agar fitur dengan angka besar tidak mendominasi
X_scaled = StandardScaler().fit_transform(X)

# Ambil titik sampel (data pertama)
sample_idx = 0
sample_point = X_scaled[sample_idx]

results = []
for i in range(1, len(X_scaled)):
    euc, man, mink = calculate_distances(sample_point, X_scaled[i], p=3)
    results.append({
        'index': i,
        'target': y[i],
        'euclidean': euc,
        'manhattan': man,
        'minkowski_p3': mink
    })

import pandas as pd
df_dist = pd.DataFrame(results)

print(f"Data Sampel (Index 0) - Kelas: {iris.target_names[y[sample_idx]]}")
print("\n3 Data Terdekat berdasarkan Euclidean:")
print(df_dist.nsmallest(3, 'euclidean')[['index', 'target', 'euclidean']])

print("\n3 Data Terdekat berdasarkan Manhattan:")
print(df_dist.nsmallest(3, 'manhattan')[['index', 'target', 'manhattan']])
