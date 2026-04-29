# pertemuan 6 CROSS-VALIDATION UNTUK PERBANDINGAN YANG FAIR
#KODE: CROSS-VALIDATION COMPARISON
# LISTING 4: cv_comparison.py

from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
import time
import pandas as pd

# --- 1. PERSIAPAN DATA ---
iris = load_iris()
X = iris.data
y = iris.target

# --- 2. DEFINISI STACKING (Karena dipanggil di dict models) ---
base_models = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5))
]
stacking = StackingClassifier(
    estimators=base_models, 
    final_estimator=LogisticRegression(),
    cv=5
)

# --- 3. LIST MODEL ---
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Stacking': stacking
}

# --- 4. CROSS-VALIDATION ---
# 5-fold artinya data dibagi 5 bagian, dites 5 kali secara bergantian
cv = KFold(n_splits=5, shuffle=True, random_state=42)

results = []
for name, model in models.items():
    start_time = time.time()
    # Menggunakan X dan y (seluruh dataset) untuk Cross Validation
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    elapsed_time = time.time() - start_time

    results.append({
        'Model': name,
        'Mean Accuracy': scores.mean(),
        'Std': scores.std(),
        'Training Time (s)': elapsed_time
    })

# --- 5. OUTPUT HASIL ---
results_df = pd.DataFrame(results).round(4)
print("=" * 60)
print("HASIL PERBANDINGAN MODEL DENGAN CROSS-VALIDATION")
print("=" * 60)
print(results_df.to_string(index=False))

# Kesimpulan
best_model = results_df.loc[results_df['Mean Accuracy'].idxmax(), 'Model']
print(f"\nModel terbaik berdasarkan CV: {best_model}")
print("=" * 60)