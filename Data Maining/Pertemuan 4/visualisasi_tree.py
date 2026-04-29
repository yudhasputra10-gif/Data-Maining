#VISUALISASI DECISION TREE, LISTING 2: visualisasi_tree.py
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.datasets import load_iris
import pandas as pd

# --- BAGIAN PERSIAPAN (Supaya 'dt' dan 'iris' terdefinisi) ---
iris = load_iris()
dt = DecisionTreeClassifier(max_depth=3)
dt.fit(iris.data, iris.target)
# -----------------------------------------------------------

# Visualisasi pohon keputusan (Kode yang lo minta rapihkan)
plt.figure(figsize=(15, 10))
plot_tree(
    dt,
    feature_names=iris.feature_names,
    class_names=list(iris.target_names),
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree - Iris Dataset")
plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
plt.show()

# Interpretasi : node menampilkan
# - samples : jumlah data di node
# - value : [ jumlah kelas 0 , kelas 1 , kelas 2]
# - class : kelas mayoritas
# - gini / entropy : impurity

# Feature importance
importance = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': dt.feature_importances_
}).sort_values('importance', ascending=False)

print("Feature Importance :")
print(importance)