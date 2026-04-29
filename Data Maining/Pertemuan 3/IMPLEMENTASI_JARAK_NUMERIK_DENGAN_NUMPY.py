#pertemuan 3 IMPLEMENTASI JARAK NUMERIK DENGAN NUMPY
import numpy as np
from scipy.spatial.distance import euclidean, cityblock, minkowski

# 1. Data contoh sederhana (2D)
A = np.array([2, 3])
B = np.array([5, 7])

# 2. Perhitungan berbagai metrik jarak
euc_dist = euclidean(A, B)
man_dist = cityblock(A, B)
min_dist = minkowski(A, B, p=3)  # Parameter p sama dengan r pada rumus

print(f"Euclidean Distance : {euc_dist:.2f}") # Hasil: 5.00
print(f"Manhattan Distance : {man_dist:.2f}") # Hasil: 7.00
print(f"Minkowski (r=3)    : {min_dist:.2f}") # Hasil: 4.50

# 3. Masalah Skala: Contoh data dengan perbedaan rentang yang jauh
# Misalnya Fitur 1 adalah Jumlah Kamar (skala kecil) dan Fitur 2 adalah Harga/Pendapatan (skala besar)
A_scaled = np.array([2, 2000000])
B_scaled = np.array([5, 7000000])

print("\nTanpa normalisasi:")
print(f"Euclidean: {euclidean(A_scaled, B_scaled):.2f}")
# Output akan didominasi oleh fitur pendapatan karena angkanya jauh lebih besar!