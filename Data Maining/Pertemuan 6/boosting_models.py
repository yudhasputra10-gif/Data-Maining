# pertemuan 6 IMPLEMENTASI ADABOOST DAN GRADIENT BOOSTING
# KODE: ADABOOST DAN GRADIENT BOOSTING
# LISTING 2: boosting_models.py

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd

# --- PERSIPAN DATA (Tambahan supaya tidak error) ---
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)
# ---------------------------------------------------

# 1. AdaBoost (base estimator = decision stump)
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1), # decision stump
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
ada.fit(X_train, y_train)
ada_pred = ada.predict(X_test)

# 2. Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)

# 3. Feature Importance dari Gradient Boosting
gb_importance = pd.DataFrame({
    'feature': data.feature_names,
    'importance': gb.feature_importances_
}).sort_values('importance', ascending=False)

# 4. Evaluasi
print("=" * 50)
print("Perbandingan Boosting Methods:")
print(f"AdaBoost          : {accuracy_score(y_test, ada_pred):.4f}")
print(f"Gradient Boosting : {accuracy_score(y_test, gb_pred):.4f}")
print("-" * 50)
print("Feature Importance (Gradient Boosting):")
print(gb_importance.to_string(index=False))
print("=" * 50)

# 5. Learning Curve
train_scores = []
test_scores = []
estimators = [1, 10, 25, 50, 75, 100, 150, 200]

for n in estimators:
    gb_temp = GradientBoostingClassifier(
        n_estimators=n,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    gb_temp.fit(X_train, y_train)
    train_scores.append(gb_temp.score(X_train, y_train))
    test_scores.append(gb_temp.score(X_test, y_test))

# 6. Visualisasi Learning Curve
plt.figure(figsize=(8, 5))
plt.plot(estimators, train_scores, label='Train', marker='o')
plt.plot(estimators, test_scores, label='Test', marker='s')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('Gradient Boosting Learning Curve')
plt.legend()
plt.grid(True)
plt.show()