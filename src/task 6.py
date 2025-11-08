import pandas as pd
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score  # Додано імпорт метрик

# Підключення до БД
print("Підключення до бази даних 000.db...")
conn = sqlite3.connect('lab3.db')
df = pd.read_sql_query("SELECT * FROM obesity_data", conn)
print(f"Дані успішно завантажено: {df.shape[0]} рядків, {df.shape[1]} стовпців")

# Розділення на X та y
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']
print(f"Ознаки (X): {X.shape}, Цільова змінна (y): {y.shape}")

# Поділ на train/test з стратифікацією
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Навчальна вибірка: {X_train.shape[0]} прикладів")
print(f"Тестова вибірка: {X_test.shape[0]} прикладів")
print(f"Стратифікація збережена (train): {dict(y_train.value_counts().sort_index())}")

# Тренування моделі Random Forest
print("\nТренування Random Forest з class_weight='balanced'...")
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
print("Модель успішно навчилась!")

# Передбачення на train
train_preds = model.predict(X_train)
print(f"Передбачення на train: {len(train_preds)} прогнозів")

# Передбачення на test
test_preds = model.predict(X_test)
print(f"Передбачення на test: {len(test_preds)} прогнозів")

# === Обчислення метрик ===
print("\nОбчислення метрик (accuracy, F1-macro, recall-macro)...")

# Метрики для train
train_acc = accuracy_score(y_train, train_preds)
train_f1 = f1_score(y_train, train_preds, average='macro')
train_recall = recall_score(y_train, train_preds, average='macro')

# Метрики для test
test_acc = accuracy_score(y_test, test_preds)
test_f1 = f1_score(y_test, test_preds, average='macro')
test_recall = recall_score(y_test, test_preds, average='macro')

# Вивід метрик у консоль
print(f"\n{'='*50}")
print(f"{'МЕТРИКИ МОДЕЛІ':^50}")
print(f"{'='*50}")
print(f"{'Навчальна вибірка (train)':<30} | Accuracy: {train_acc:.4f} | F1-macro: {train_f1:.4f} | Recall-macro: {train_recall:.4f}")
print(f"{'Тестова вибірка (test)':<30} | Accuracy: {test_acc:.4f} | F1-macro: {test_f1:.4f} | Recall-macro: {test_recall:.4f}")
print(f"{'='*50}")

# === Логування передбачень ===
preds_df = pd.DataFrame({
    'true_label': y_train.values,
    'predicted_label': train_preds,
    'source': 'train'
})
preds_df.to_sql('predictions', conn, if_exists='append', index=False)
print(f"Передбачення на train збережено в таблицю 'predictions' — {len(preds_df)} рядків")

# === Логування метрик ===
metrics_df = pd.DataFrame({
    'model': ['RandomForest', 'RandomForest'],
    'split': ['train', 'test'],
    'accuracy': [train_acc, test_acc],
    'f1_macro': [train_f1, test_f1],
    'recall_macro': [train_recall, test_recall]
})
metrics_df.to_sql('model_metrics', conn, if_exists='append', index=False)
print(f"Метрики збережено в таблицю 'model_metrics' — {len(metrics_df)} рядків")

# Закриття з'єднання
conn.close()
print("\nЗ'єднання з БД закрито. Усі операції завершено успішно!")