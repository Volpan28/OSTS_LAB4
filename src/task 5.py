import pandas as pd
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Підключення до БД
print("Підключення до бази даних 000.db...")
conn = sqlite3.connect('lab3.db')
df = pd.read_sql_query("SELECT * FROM obesity_data", conn)
print(f"Дані успішно завантажено: {df.shape[0]} рядків, {df.shape[1]} стовпців")

# Розділення на X та y
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']
print(f"Ознаки (X): {X.shape}, Цільова змінна (y): {y.shape}")

# Поділ на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Навчальна вибірка: {X_train.shape[0]} прикладів")
print(f"Тестова вибірка: {X_test.shape[0]} прикладів")
print(f"Стратифікація збережена: {y_train.value_counts().sort_index().to_dict()} (train)")

# Тренування моделі
print("\nТренування Random Forest з class_weight='balanced'...")
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
print("Модель успішно навчилась!")

# Передбачення на train
train_preds = model.predict(X_train)
print(f"Передбачення на train завершено: {len(train_preds)} прогнозів")

# Логування в таблицю predictions
preds_df = pd.DataFrame({
    'true_label': y_train.values,
    'predicted_label': train_preds,
    'source': 'train'
})
preds_df.to_sql('predictions', conn, if_exists='append', index=False)
print(f"Результати збережено в таблицю 'predictions' (source='train') — {len(preds_df)} рядків")

# Закриття з'єднання
conn.close()
print("З'єднання з БД закрито. Готово!")