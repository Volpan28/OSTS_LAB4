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

    # 1. Завантажуємо модель З ДИСКА при кожному запиті
    # Це гарантує, що ми використовуємо версію, навчену через /train-model
    try:
        model = model_service.load_model_for_inference()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # 2. Логуємо вхідні дані
    crud.create_inference_input(db, input_data)

    # 3. Робимо прогноз (вже з правильним порядком колонок)
    prediction_result = model_service.predict_single(model, input_data)

    # 4. Логуємо вихідні дані
    prediction_log = schemas.PredictionCreate(
        predicted_label=prediction_result,
        source="inference"  # true_label = null (це правильно)
    )
    db_log = crud.create_prediction(db, prediction_log)

    return db_log