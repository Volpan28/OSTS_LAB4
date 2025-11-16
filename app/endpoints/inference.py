from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import database, crud, schemas
from app.services import model as model_service
import pandas as pd

router = APIRouter()

@router.post("/predict", response_model=schemas.PredictionCreate, tags=["Inference"])
def predict(
        input_data: schemas.InferenceInput,
        db: Session = Depends(database.get_db)
):
    """
    Робить прогноз для нових даних.
    """

    try:
        # 1. Завантажуємо модель І КОЛОНКИ
        model, trained_features = model_service.load_model_for_inference()  # <--- ЗМІНА ТУТ
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # 2. Логуємо вхідні дані
    crud.create_inference_input(db, input_data)

    # 3. Робимо прогноз
    prediction_result = model_service.predict_single(model, trained_features, input_data)  # <--- ЗМІНА ТУТ

    # 4. Логуємо вихідні дані
    prediction_log = schemas.PredictionCreate(
        predicted_label=prediction_result,
        source="inference"
    )
    db_log = crud.create_prediction(db, prediction_log)

    return db_log