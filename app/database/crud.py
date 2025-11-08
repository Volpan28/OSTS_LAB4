from sqlalchemy.orm import Session
import pandas as pd
from app.database import models, schemas

# Читання ВСІХ даних для тренування
def get_all_features(db: Session):
    # Використовуємо SQLAlchemy модель
    query = db.query(models.ObesityData)
    # Конвертуємо в Pandas DataFrame
    df = pd.read_sql(query.statement, db.bind)
    return df

# Запис вхідних даних (з /predict)
def create_inference_input(db: Session, input_data: schemas.InferenceInputCreate):
    # Pydantic-схема (input_data) конвертується в dict
    db_input = models.InferenceInputs(**input_data.model_dump())
    db.add(db_input)
    db.commit()
    db.refresh(db_input)
    return db_input

# Запис прогнозу (з /train-model або /predict)
def create_prediction(db: Session, prediction_data: schemas.PredictionCreate):
    db_pred = models.Predictions(**prediction_data.model_dump())
    db.add(db_pred)
    db.commit()
    db.refresh(db_pred)
    return db_pred