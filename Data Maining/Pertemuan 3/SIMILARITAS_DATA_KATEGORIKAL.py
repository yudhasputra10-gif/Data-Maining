#pertemuan 3 SIMILARITAS DATA KATEGORIKAL
import numpy as np
from sklearn.metrics import jaccard_score

def simple_matching_coefficient(p, q):
    """ Menghitung SMC untuk dua vektor biner """
    p = np.array(p)
    q = np.array(q)

    # m11: keduanya bernilai 1 (kehadiran bersama)
    # m00: keduanya bernilai 0 (ketidakhadiran bersama)
    m11 = np.sum((p == 1) & (q == 1))
    m00 = np.sum((p == 0) & (q == 0))
    total = len(p)

    return (m11 + m00) / total

# Data contoh biner
p = [1, 0, 1, 1, 0]
q = [1, 0, 0, 1, 1]

# 1. Hitung SMC
smc = simple_matching_coefficient(p, q)
print(f"SMC (Simple Matching) : {smc:.2f}") # Output: 0.60

# 2. Hitung Jaccard secara manual
# Jaccard mengabaikan m00 (ketidakhadiran bersama dianggap tidak informatif)
m11 = np.sum((np.array(p) == 1) & (np.array(q) == 1))
m10 = np.sum((np.array(p) == 1) & (np.array(q) == 0))
m01 = np.sum((np.array(p) == 0) & (np.array(q) == 1))

jaccard_manual = m11 / (m11 + m10 + m01)
print(f"Jaccard (Manual)      : {jaccard_manual:.2f}") # Output: 0.50

# 3. Hitung Jaccard dengan scikit-learn
jaccard_sklearn = jaccard_score(p, q, average='binary')
print(f"Jaccard (Sklearn)     : {jaccard_sklearn:.2f}") # Output: 0.50