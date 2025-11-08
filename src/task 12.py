import sys
sys.path.append('D:\\pycharm\\OSTS_Lab3')
from task_11 import best_xgb
import joblib
import os
from datetime import datetime

"""
Мета: Зберегти навчену модель для використання в продакшні.
Формат: .pkl (joblib) — швидкий, компактний.
Папка: models/
"""

print("="*60)
print("КРОК 12: ЗБЕРЕЖЕННЯ МОДЕЛІ")
print("="*60)

os.makedirs('../../OSTS_Lab3/models', exist_ok=True)
model_path = f"models/xgboost_optimized_{datetime.now().strftime('%Y%m%d')}.pkl"

joblib.dump(best_xgb, model_path)

file_size = os.path.getsize(model_path) / 1024  # KB
print(f"Модель збережена: {model_path}")
print(f"Розмір файлу: {file_size:.1f} KB")
print("Модель готова до розгортання (Flask, FastAPI, тощо).")