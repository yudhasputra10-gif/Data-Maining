# pertemuan 4 EKSTRAKSI ATURAN DARI DECISION TREE, LISTING 4: extract_rules_from_tree.py
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

# --- PERSIAPAN DATA (Wajib ada supaya tidak Error) ---
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)
# -----------------------------------------------------

def extract_rules(tree, feature_names, class_names, node_id=0, current_rule=None):
    # Ekstrak aturan IF-THEN dari decision tree secara rekursif
    if current_rule is None:
        current_rule = []

    rules = []

    # Cek apakah node adalah daun (leaf)
    if tree.children_left[node_id] == tree.children_right[node_id]:
        # Node daun: tentukan kelas dengan nilai terbanyak
        class_id = np.argmax(tree.value[node_id][0])
        rule_str = "IF " + " AND ".join(current_rule) + f" THEN {class_names[class_id]}"
        rules.append(rule_str)
    else:
        # Node internal: ambil fitur dan ambang batas (threshold) pemisahnya
        feature = feature_names[tree.feature[node_id]]
        threshold = tree.threshold[node_id]

        # Rekursif ke cabang kiri (kondisi: <= threshold)
        left_rule = current_rule + [f"({feature} <= {threshold:.2f})"]
        rules.extend(extract_rules(tree, feature_names, class_names,
                                   tree.children_left[node_id], left_rule))

        # Rekursif ke cabang kanan (kondisi: > threshold)
        right_rule = current_rule + [f"({feature} > {threshold:.2f})"]
        rules.extend(extract_rules(tree, feature_names, class_names,
                                   tree.children_right[node_id], right_rule))

    return rules

# --- Contoh penggunaan ---
# 1. Training model sederhana
dt = DecisionTreeClassifier(max_depth=2, random_state=42)
dt.fit(X_train, y_train)

# 2. Ekstraksi aturan dari objek dt.tree_
all_rules = extract_rules(dt.tree_, iris.feature_names, iris.target_names)

# 3. Cetak hasil
print("Aturan yang diekstrak dari Decision Tree:")
print("=" * 50)
for i, rule in enumerate(all_rules):
    print(f"{i + 1}. {rule}")