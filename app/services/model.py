import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
import os
from sqlalchemy.orm import Session
from app.database import crud, schemas

# Визначаємо шлях до папки з моделями
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_optimized.pkl")

# Створюємо папку, якщо її немає
os.makedirs(MODEL_DIR, exist_ok=True)

# Гіперпараметри, знайдені в Лабораторній 3 (Крок 10)
# ВАЖЛИВО: Я вставив ті, що були у вашому файлі .ipynb (Крок 11)
best_params = {
    'n_estimators': 387,
    'max_depth': 7,
    'learning_rate': 0.11234,
    'subsample': 0.823,
    'colsample_bytree': 0.745,
    'min_child_weight': 4
}


def train_and_save_model(db: Session):
    """
    Виконується ендпоінтом /train-model
    """
    print("Завантаження даних з БД...")
    df = crud.get_all_features(db)

    # Видаляємо ID, якщо він є, і цільову змінну
    if 'id' in df.columns:
        df = df.drop('id', axis=1)

    X = df.drop('NObeyesdad', axis=1)
    y = df['NObeyesdad']

    # Використовуємо поділ 90:10, як вказано в Завданні 4
    X_train, X_new_input, y_train, y_new_input = train_test_split(
        X, y, test_size=0.10, random_state=42, stratify=y
    )

    print(f"Тренування моделі на {len(X_train)} зразках...")
    # Ініціалізуємо модель з найкращими параметрами
    model = XGBClassifier(
        **best_params,
        random_state=42,
        eval_metric='mlogloss',
        use_label_encoder=False  # Для сумісності
    )

    model.fit(X_train, y_train)

    print(f"Збереження моделі у файл: {MODEL_PATH}")
    joblib.dump(model, MODEL_PATH)

    # === Логування прогнозів на train-даних (Завдання 3.5) ===
    print("Логування тренувальних прогнозів у БД...")
    train_preds = model.predict(X_train)

    log_count = 0
    for true_val, pred_val in zip(y_train, train_preds):
        prediction_data = schemas.PredictionCreate(
            true_label=int(true_val),
            predicted_label=int(pred_val),
            source="train"
        )
        crud.create_prediction(db, prediction_data)
        log_count += 1

    print(f"Залоговано {log_count} тренувальних прогнозів.")
    return model, {"status": "Model trained and saved", "train_samples": len(X_train)}


def load_model_for_inference():
    """
    Виконується ендпоінтом /predict
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Модель не знайдена за шляхом: {MODEL_PATH}. Спочатку запустіть /train-model.")

    print(f"Завантаження моделі з файлу: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    return model


def predict_single(model, input_data: schemas.InferenceInput):
    """
    Робить прогноз для одного JSON-запиту.
    """
    # Конвертуємо Pydantic-схему в DataFrame, оскільки модель очікує імена ознак
    input_df = pd.DataFrame([input_data.model_dump()])

    # Переконуємося, що порядок колонок правильний
    # (Модель XGBoost чутлива до порядку)
    input_df = input_df[model.feature_names_in_]

    prediction = model.predict(input_df)

    # predict() повертає масив, беремо перший елемент
    return int(prediction[0])