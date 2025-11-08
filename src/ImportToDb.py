import sqlite3
import pandas as pd
import os

# --- Потужне налаштування шляхів ---

# Абсолютний шлях до папки, де знаходиться ЦЕЙ СКРИПТ (тобто .../src/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Шлях до CSV (лежить поруч зі скриптом у 'src/')
CSV_PATH = os.path.join(SCRIPT_DIR, 'task-d.csv')

# Шлях до БД (на один рівень вище, в корені проєкту)
DB_PATH = os.path.join(SCRIPT_DIR, '..', 'lab3.db')

TABLE_NAME = 'obesity_data'


# ------------------------------------

def create_table_from_csv():
    # Перевіряємо, чи існує CSV файл за правильним шляхом
    if not os.path.exists(CSV_PATH):
        print(f"Помилка: Файл {CSV_PATH} не знайдено.")
        print(f"Будь ласка, переконайтеся, що 'task-d.csv' знаходиться в папці 'src'.")
        return

    try:
        print(f"Підключення до бази даних: {os.path.abspath(DB_PATH)}")
        conn = sqlite3.connect(DB_PATH)

        print(f"Читання даних з {CSV_PATH}...")
        df = pd.read_csv(CSV_PATH)

        # Список колонок з вашої Лабораторної 3 (Крок 8)
        required_columns = [
            'Weight', 'FCVC', 'Height', 'CH2O', 'Age',
            'FAF', 'CAEC', 'NCP', 'SCC', 'CALC', 'NObeyesdad'
        ]

        # Перевіряємо, чи всі потрібні колонки є в CSV
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Помилка: у файлі CSV відсутні колонки: {missing_cols}")
            conn.close()
            return

        df_filtered = df[required_columns]

        print(f"Запис {len(df_filtered)} рядків у таблицю '{TABLE_NAME}'...")
        # if_exists='replace' повністю перестворює таблицю
        df_filtered.to_sql(TABLE_NAME, conn, if_exists='replace', index=True, index_label='id')

        print(f"Успішно імпортовано! Таблиця '{TABLE_NAME}' створена/оновлена у {os.path.abspath(DB_PATH)}")

    except Exception as e:
        print(f"Сталася помилка: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()
            print("З'єднання з БД закрито.")


# Цей блок виконає функцію, коли ви запустите скрипт
if __name__ == "__main__":
    create_table_from_csv()