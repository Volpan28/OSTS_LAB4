# step_11_train_optimized.py
# КРОК 11: Навчання XGBoost з найкращими параметрами з Кроку 10

import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# --- Опис кроку ---
"""
Мета: Навчити XGBoost з найкращими гіперпараметрами, знайденими в Кроці 10.
Метрика: F1-macro — чутлива до всіх класів.
Результат: Збереження метрик у БД для порівняння з базовою моделлю.
"""

# --- Підключення до БД ---
conn = sqlite3.connect('lab3.db')
df = pd.read_sql_query("SELECT * FROM obesity_data", conn)
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Найкращі параметри з Кроку 10 (вставте з виводу step_10) ---
# Приклад (замініть на свої):
best_params = {
    'n_estimators': 387,
    'max_depth': 7,
    'learning_rate': 0.11234,
    'subsample': 0.823,
    'colsample_bytree': 0.745,
    'min_child_weight': 4
}

print("="*70)
print("КРОК 11: НАВЧАННЯ З ОПТИМАЛЬНИМИ ПАРАМЕТРАМИ")
print("="*70)

# --- Модель ---
best_xgb = XGBClassifier(
    **best_params,
    random_state=42,
    eval_metric='mlogloss'
)

best_xgb.fit(X_train, y_train)

# --- Передбачення ---
train_pred = best_xgb.predict(X_train)
test_pred = best_xgb.predict(X_test)

# --- Метрики ---
train_acc = accuracy_score(y_train, train_pred)
train_f1 = f1_score(y_train, train_pred, average='macro')
test_acc = accuracy_score(y_test, test_pred)
test_f1 = f1_score(y_test, test_pred, average='macro')

print(f"Train → Accuracy: {train_acc:.5f} | F1-macro: {train_f1:.5f}")
print(f"Test  → Accuracy: {test_acc:.5f} | F1-macro: {test_f1:.5f}")

# --- Збереження метрик ---
metrics_df = pd.DataFrame([
    {'model': 'XGBoost_Optimized', 'split': 'train', 'accuracy': train_acc, 'f1_macro': train_f1},
    {'model': 'XGBoost_Optimized', 'split': 'test',  'accuracy': test_acc,  'f1_macro': test_f1}
])
metrics_df.to_sql('model_metrics', conn, if_exists='append', index=False)

conn.close()
print("\nМетрики збережено в 'model_metrics'")
print("Крок 11 завершено. Переходимо до збереження моделі.")