#pertemuan 4 PERBANDINGAN DECISION TREE DENGAN RULE-BASED, LISTING 5: compare_models.py
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import seaborn as sns

# 1. Load data (Contoh: Titanic)
titanic = sns.load_dataset('titanic')

# Memilih fitur yang relevan
titanic = titanic[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'survived']]

# Membersihkan data dari nilai kosong (NaN)
titanic = titanic.dropna()

# Mengubah kategori 'sex' menjadi numerik (male: 0, female: 1)
titanic['sex'] = titanic['sex'].map({'male': 0, 'female': 1})

# 2. Menentukan Fitur (X) dan Target (y)
X = titanic.drop('survived', axis=1)
y = titanic['survived']

# 3. Split data menjadi Training dan Testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Inisialisasi Berbagai Model
models = {
    'Decision Tree (Gini)': DecisionTreeClassifier(criterion='gini', max_depth=5),
    'Decision Tree (Entropy)': DecisionTreeClassifier(criterion='entropy', max_depth=5),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    'k-NN (k=5)': KNeighborsClassifier(n_neighbors=5)
}

# 5. Training dan Evaluasi
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

# 6. Menampilkan Hasil dalam DataFrame
results_df = pd.DataFrame(results)
print("=" * 60)
print("PERBANDINGAN PERFORMA MODEL")
print("=" * 60)
print(results_df.round(4).to_string(index=False))