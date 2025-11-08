import pandas as pd
import sqlite3

# Читання CSV файлу
df = pd.read_csv('task-d.csv')

# Підключення до БД (створення, якщо не існує)
conn = sqlite3.connect('lab3.db')

# Збереження даних у таблицю 'obesity_data'
df.to_sql('obesity_data', conn, if_exists='replace', index=False)

# Закриття з'єднання
conn.close()