# ==============================================================
# Крок 7 – Порівняння моделей (ФІНАЛЬНА ВЕРСІЯ)
# ==============================================================

import pandas as pd
import sqlite3
import warnings
warnings.filterwarnings('ignore')

# ---------- 1. Підключення до БД ----------
print("Підключення до 000.db ...")
conn = sqlite3.connect('lab3.db')
df = pd.read_sql_query("SELECT * FROM obesity_data", conn)
print(f"Дані завантажено: {df.shape}")

# ---------- 2. X / y ----------
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

# ---------- 3. Стратифікований поділ ----------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"train: {X_train.shape[0]}, test: {X_test.shape[0]}")

# ---------- 4. Моделі ----------
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score

models = {
    'RandomForest': RandomForestClassifier(
        random_state=42, class_weight='balanced', n_estimators=200
    ),
    'LogisticRegression': LogisticRegression(
        random_state=42, class_weight='balanced', max_iter=1000
    ),
    'KNeighbors': KNeighborsClassifier(n_neighbors=5),
    'XGBoost': XGBClassifier(
        random_state=42, eval_metric='mlogloss', use_label_encoder=False
    )
}

# ---------- 5. Навчання + метрики ----------
metrics_list = []
preds_list   = []

print("\nТренування моделей...")
for name, clf in models.items():
    print(f"  → {name}")

    clf.fit(X_train, y_train)

    train_pred = clf.predict(X_train)
    test_pred  = clf.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    train_f1  = f1_score(y_train, train_pred, average='macro')
    train_rec = recall_score(y_train, train_pred, average='macro')

    test_acc = accuracy_score(y_test, test_pred)
    test_f1  = f1_score(y_test, test_pred, average='macro')
    test_rec = recall_score(y_test, test_pred, average='macro')

    # === Метрики в список ===
    metrics_list.extend([
        {'model': name, 'split': 'train', 'accuracy': train_acc, 'f1_macro': train_f1, 'recall_macro': train_rec},
        {'model': name, 'split': 'test',  'accuracy': test_acc,  'f1_macro': test_f1,  'recall_macro': test_rec}
    ])

    # === Передбачення ===
    preds_df = pd.DataFrame({
        'true_label': y_train.values,
        'predicted_label': train_pred,
        'source': 'train',
        'model': name
    })
    preds_list.append(preds_df)

    print(f"    train → acc:{train_acc:.4f}  f1:{train_f1:.4f}  rec:{train_rec:.4f}")
    print(f"    test  → acc:{test_acc:.4f}  f1:{test_f1:.4f}  rec:{test_rec:.4f}")

# ==============================================================
# 6. СТВОРЕННЯ ТАБЛИЦЬ (гарантовано!)
# ==============================================================

print("\nІніціалізація таблиць у БД...")

# --- Таблиця predictions ---
conn.execute("DROP TABLE IF EXISTS predictions")
conn.execute('''
CREATE TABLE predictions (
    true_label INTEGER,
    predicted_label INTEGER,
    source TEXT,
    model TEXT
)
''')
print("Таблиця 'predictions' створена")

# --- Таблиця model_metrics (ОКРЕМА!) ---
conn.execute("DROP TABLE IF EXISTS model_metrics")
conn.execute('''
CREATE TABLE model_metrics (
    model TEXT,
    split TEXT,
    accuracy REAL,
    f1_macro REAL,
    recall_macro REAL
)
''')
print("Таблиця 'model_metrics' створена (окремо)")

# ==============================================================
# 7. ЗАПИС У БД
# ==============================================================

print("\nЗапис результатів у БД...")

# --- Запис передбачень ---
preds_all = pd.concat(preds_list, ignore_index=True)
preds_all.to_sql('predictions', conn, if_exists='append', index=False)
print(f"   predictions → {len(preds_all)} рядків збережено")

# --- Запис МЕТРИК у ОКРЕМУ таблицю ---
metrics_all = pd.DataFrame(metrics_list)
metrics_all.to_sql('model_metrics', conn, if_exists='append', index=False)
print(f"   model_metrics → {len(metrics_all)} рядків збережено")

# ==============================================================
# 8. Підсумкова таблиця
# ==============================================================

print("\n" + "="*80)
print(f"{'ПОРІВНЯННЯ МОДЕЛЕЙ':^80}")
print("="*80)

summary = metrics_all.pivot_table(
    index='model',
    columns='split',
    values=['accuracy','f1_macro','recall_macro'],
    aggfunc='first'
).round(4)

print(summary.to_string())
print("="*80)


print("\nПеревірка вмісту таблиць:")
print("predictions (перші 3):")
print(pd.read_sql_query("SELECT * FROM predictions LIMIT 3", conn))
print("\nmodel_metrics (всі):")
print(pd.read_sql_query("SELECT * FROM model_metrics", conn))

conn.close()
print("\nЗ'єднання закрито. Усе збережено!")