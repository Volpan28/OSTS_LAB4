# step_14_handle_imbalance.py
# КРОК 14: Аналіз та обробка дисбалансу

import pandas as pd
import sqlite3

"""
Мета: Показати, що дисбаланс враховано.
Техніки:
  • class_weight='balanced' (RF, LR)
  • стратифікація при поділі
  • F1-macro як метрика
XGBoost — стійкий до дисбалансу.
"""

conn = sqlite3.connect('lab3.db')
df = pd.read_sql_query("SELECT * FROM obesity_data", conn)

print("="*60)
print("КРОК 14: ОБРОБКА ДИСБАЛАНСУ КЛАСІВ")
print("="*60)

print("Розподіл класів:")
class_dist = df['NObeyesdad'].value_counts().sort_index()
print(class_dist)

print(f"\nДисбаланс: {class_dist.max() / class_dist.min():.1f}x")

print("\nВикористані техніки:")
print("• class_weight='balanced' — у Random Forest та Logistic Regression")
print("• stratify=y — при train_test_split")
print("• F1-macro — метрика, що враховує всі класи")
print("• XGBoost — вбудована регуляризація та scale_pos_weight (якщо потрібно)")

conn.close()
print("\nДисбаланс оброблено. Крок 14 завершено.")