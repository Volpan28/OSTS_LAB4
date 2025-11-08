import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split, StratifiedKFold

# Підключення до БД
conn = sqlite3.connect('lab3.db')
df = pd.read_sql_query("SELECT * FROM obesity_data", conn)
conn.close()

print(f"Розмір датасету: {df.shape}")
print(f"Кількість класів у NObeyesdad: {df['NObeyesdad'].nunique()}")
print(f"Розподіл класів:\n{df['NObeyesdad'].value_counts().sort_index()}\n")

# Розділення на X та y
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

# Поділ на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Розмір X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"Розмір y_train: {y_train.shape}, y_test: {y_test.shape}")

# Перевірка стратифікації
print("\nРозподіл класів у y_train:")
print(y_train.value_counts().sort_index() / len(y_train))
print("\nРозподіл класів у y_test:")
print(y_test.value_counts().sort_index() / len(y_test))

# StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(f"\nStratifiedKFold створено: {skf}")