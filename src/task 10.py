# task_10.py — Крок 10: Оптимізація XGBoost (БЕЗ попереджень)
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
import warnings
warnings.filterwarnings('ignore')  # Прибираємо всі попередження

# --- Дані ---
conn = sqlite3.connect('lab3.db')
df = pd.read_sql_query("SELECT * FROM obesity_data", conn)
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Параметри ---
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.29),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'min_child_weight': randint(1, 10)
}

# --- Модель БЕЗ застарілого параметра ---
xgb = XGBClassifier(
    random_state=42,
    eval_metric='mlogloss'  # ТІЛЬКИ цей параметр потрібен
)

search = RandomizedSearchCV(
    xgb,
    param_distributions=param_dist,
    n_iter=30,
    scoring='f1_macro',
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=0  # Вимикаємо детальний вивід
)

print("="*70)
print("КРОК 10: ГІПЕРПАРАМЕТРИЧНА ОПТИМІЗАЦІЯ (RandomizedSearchCV)")
print("="*70)
print("Запуск пошуку (30 ітерацій)...")

search.fit(X_train, y_train)

# --- Результати ---
best_params = search.best_params_
best_score = search.best_score_

print("\nНайкращі параметри:")
for k, v in best_params.items():
    print(f"  {k}: {v}")
print(f"Найкращий F1-macro (CV): {best_score:.5f}")

# --- Збереження ---
pd.DataFrame([best_params]).to_sql('best_params', conn, if_exists='replace', index=False)
conn.close()

print("\nПараметри збережено в таблицю 'best_params'")
print("Крок 10 завершено успішно!")
