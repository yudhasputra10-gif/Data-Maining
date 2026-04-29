#pertemuan 3 PENTINGNYA NORMALISASI
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.spatial.distance import euclidean

# 1. Data dengan skala sangat berbeda (Fitur 1: Satuan, Fitur 2: Jutaan)
data = np.array([
    [2, 2000000],
    [5, 7000000],
    [3, 5000000],
    [8, 12000000]
])

# 2. Min-Max Normalization (Mengubah skala ke rentang 0 sampai 1)
scaler_minmax = MinMaxScaler()
data_minmax = scaler_minmax.fit_transform(data)

# 3. Z-Score Normalization / Standardization (Mean = 0, Std Dev = 1)
scaler_zscore = StandardScaler()
data_zscore = scaler_zscore.fit_transform(data)

# 4. Output Hasil
print("Data Asli:")
print(data)

print("\nSetelah Min-Max (Rentang 0-1):")
print(data_minmax)

print("\nSetelah Z-Score (Mean=0, StdDev=1):")
print(data_zscore)

# 5. Perbandingan Jarak Euclidean
dist_asli = euclidean(data[0], data[1])
dist_minmax = euclidean(data_minmax[0], data_minmax[1])

print("-" * 50)
print(f"Jarak Euclidean (Data Asli)   : {dist_asli:.2f}")
print(f"Jarak Euclidean (Min-Max)     : {dist_minmax:.2f}")