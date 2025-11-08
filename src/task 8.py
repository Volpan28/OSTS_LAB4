# task_8.py — Крок 8: Важливість ознак НАЙКРАЩОЇ моделі (XGBoost)
import pandas as pd
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Підключення ---
conn = sqlite3.connect('lab3.db')
df = pd.read_sql_query("SELECT * FROM obesity_data", conn)

X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Навчаємо XGBoost (найкраща модель) ---
from xgboost import XGBClassifier
xgb = XGBClassifier(
    random_state=42,
    eval_metric='mlogloss',
    use_label_encoder=False,
    n_estimators=400,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.8
).fit(X_train, y_train)

# --- Важливість для XGBoost ---
xgb_imp = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb.feature_importances_
}).sort_values('importance', ascending=False)

# --- Збереження в БД ---
xgb_imp.to_sql('feature_importance', conn, if_exists='replace', index=False)

# --- Вивід ---
print("="*70)
print("КРОК 8: ВАЖЛИВІСТЬ ОЗНАК (XGBoost — найкраща модель)")
print("="*70)
print("Топ-5 ознак:")
print(xgb_imp.head(5)[['feature', 'importance']].round(5))

# --- Графік ---
os.makedirs('../../OSTS_Lab3/plots', exist_ok=True)
plot_path = '../../OSTS_Lab3/plots/feature_importance_xgboost.png'

plt.figure(figsize=(10, 6))
sns.barplot(
    data=xgb_imp.head(10),
    x='importance', y='feature',
    hue='feature', palette='viridis', legend=False
)
plt.title('Топ-10 важливих ознак (XGBoost — найкраща модель)')
plt.xlabel('Важливість')
plt.tight_layout()
plt.savefig(plot_path, dpi=200, bbox_inches='tight')
plt.close()

# --- Перевірка файлу ---
if os.path.exists(plot_path):
    print(f"ГРАФІК ЗБЕРЕЖЕНО: {os.path.abspath(plot_path)}")
    print(f"   Розмір: {os.path.getsize(plot_path)/1024:.1f} KB")
else:
    print("ПОМИЛКА: графік не збережено!")

conn.close()
print("Крок 8 завершено.")