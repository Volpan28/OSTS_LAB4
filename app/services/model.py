import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
import os
from sqlalchemy.orm import Session
from app.database import crud, schemas, models  # <--- Переконайся, що 'models' тут є

# Шлях до папки з моделями
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_optimized.pkl")

# Створюємо папку, якщо її немає
os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------------------------------------------------------
# ВИДАЛЯЄМО АБО КОМЕНТУЄМО 'best_params' (це ти вже зробив)
# ---------------------------------------------------------------
# best_params = { ... }

# ---------------------------------------------------------------
# ВИДАЛЯЄМО АБО КОМЕНТУЄМО 'FEATURE_NAMES'
# ---------------------------------------------------------------
# FEATURE_NAMES = [ ... ]


def train_and_save_model(db: Session):
    """
    Виконується ендпоінтом /train-model (Завдання 3)
    """
    print("Завантаження даних з БД...")
    df = crud.get_all_features(db)

    if df.empty:
        raise ValueError("Таблиця 'obesity_data' порожня. Запустіть src/ImportToDb.py")

    # ВАЖЛИВА ЗМІНА: Видаляємо ID (якщо він є) та цільову змінну.
    # Решта колонок (10+) підуть у тренування.
    if 'id' in df.columns:
        df = df.drop('id', axis=1)

    X = df.drop('NObeyesdad', axis=1)
    y = df['NObeyesdad']

    # Збережемо імена колонок, на яких вчилися, для прогнозу
    trained_features = X.columns.tolist()
    joblib.dump(trained_features, os.path.join(MODEL_DIR, "features.pkl"))  # <--- ЗБЕРІГАЄМО КОЛОНКИ

    # Поділ 90:10
    X_train, X_new_input, y_train, y_new_input = train_test_split(
        X, y, test_size=0.10, random_state=42, stratify=y
    )

    print(f"Тренування моделі на {len(X_train)} зразках та {len(trained_features)} ознаках...")

    # Використовуємо стандартні налаштування (це ти вже зробив)
    model = XGBClassifier(
        random_state=42,
        eval_metric='mlogloss',
        use_label_encoder=False
    )

    model.fit(X_train, y_train)

    print(f"Збереження моделі у файл: {MODEL_PATH}")
    joblib.dump(model, MODEL_PATH)

    # ... (решта функції логування не змінилася) ...
    print("Логування тренувальних прогнозів у БД...")
    train_preds = model.predict(X_train)

    log_count = 0
    db.query(models.Predictions).filter(models.Predictions.source == "train").delete()
    db.commit()

    for true_val, pred_val in zip(y_train, train_preds):
        prediction_data = schemas.PredictionCreate(
            true_label=int(true_val),
            predicted_label=int(pred_val),
            source="train"
        )
        crud.create_prediction(db, prediction_data)
        log_count += 1

    print(f"Залоговано {log_count} тренувальних прогнозів.")
    return model, {"status": "Model trained and saved", "train_samples": len(X_train),
                   "features_used": len(trained_features)}


def load_model_for_inference():
    """
    Виконується ендпоінтом /predict (Завдання 4)
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Модель не знайдена за шляхом: {MODEL_PATH}. Спочатку запустіть POST /api/train-model")

    print(f"Завантаження моделі з файлу: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    # Завантажуємо список колонок, на яких модель вчилася
    features_path = os.path.join(MODEL_DIR, "features.pkl")
    if not os.path.exists(features_path):
        raise FileNotFoundError("Файл 'features.pkl' не знайдено. Будь ласка, перетренуйте модель.")

    trained_features = joblib.load(features_path)

    return model, trained_features  # <--- Повертаємо і модель, і колонки


def predict_single(model, trained_features, input_data: schemas.InferenceInput):
    """
    Робить прогноз для одного JSON-запиту.
    """
    # 1. Конвертуємо Pydantic-схему в DataFrame
    input_df = pd.DataFrame([input_data.model_dump()])

    # 2. ВАЖЛИВО: Впорядковуємо колонки, як у trained_features
    # (Ми беремо з JSON-у тільки ті 10, що є у схемі, але нам треба
    # додати решту колонок, яких очікує модель, і заповнити їх нулями)

    # Створюємо порожній DataFrame з усіма колонками, на яких вчилася модель
    final_input_df = pd.DataFrame(columns=trained_features)
    # Додаємо наш один рядок
    final_input_df = pd.concat([final_input_df, input_df], ignore_index=True)

    # Заповнюємо відсутні колонки (напр. 'Gender_Male', 'SMOKE' і т.д.) нулями
    final_input_df = final_input_df.fillna(0)

    # Залишаємо тільки ті колонки, на яких вчилися, і в правильному порядку
    final_input_df = final_input_df[trained_features]

    # 3. Робимо прогноз
    prediction = model.predict(final_input_df)

    return int(prediction[0])