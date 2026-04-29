#tugas rumah pertemuan 6
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. LOAD DAN PREPROCESSING DATA TITANIC
# Menggunakan dataset titanic dari seaborn
df = sns.load_dataset('titanic')

# Pilih fitur yang relevan & hapus data kosong (cleaning sederhana)
df = df[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']]
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df = df.dropna()

X = df.drop('survived', axis=1)
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. IMPLEMENTASI MODEL (Sesuai Permintaan Tugas)
# A. Single Decision Tree
dt = DecisionTreeClassifier(max_depth=5, random_state=42)

# B. Random Forest (Tuning n_estimators)
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

# C. Gradient Boosting (Tuning learning rate)
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# D. Stacking (3 Base Models: RF, SVC, DT)
base_models = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('svc', SVC(probability=True, random_state=42)),
    ('dt', DecisionTreeClassifier(max_depth=5, random_state=42))
]
stacking = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression())

# 3. TRAINING DAN EVALUASI SEMUA METRIK
models = {
    'Single Decision Tree': dt,
    'Random Forest': rf,
    'Gradient Boosting': gb,
    'Stacking': stacking
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred)
    })

# 4. TAMPILKAN TABEL HASIL UNTUK LAPORAN
results_df = pd.DataFrame(results).round(4)
print("=" * 65)
print("HASIL EVALUASI MODEL - DATASET TITANIC")
print("=" * 65)
print(results_df.to_string(index=False))

# 5. VISUALISASI UNTUK LAPORAN
results_df.set_index('Model')[['Accuracy', 'F1-Score']].plot(kind='bar', figsize=(10, 5))
plt.title('Perbandingan Accuracy dan F1-Score pada Dataset Titanic')
plt.ylabel('Score')
plt.xticks(rotation=15)
plt.legend(loc='lower right')
plt.show()