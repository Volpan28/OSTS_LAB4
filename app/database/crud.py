from sqlalchemy.orm import Session
import pandas as pd
from app.database import models, schemas # <--- Виправлено імпорт

# Читання ВСІХ даних для тренування
def get_all_features(db: Session):
    query = db.query(models.ObesityData)
    df = pd.read_sql(query.statement, db.bind)
    return df

# Запис вхідних даних (з /predict)
def create_inference_input(db: Session, input_data: schemas.InferenceInputCreate):
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