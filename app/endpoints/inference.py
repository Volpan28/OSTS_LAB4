from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import database, crud, schemas
from app.services import model as model_service

router = APIRouter()

# Завантажуємо модель 1 раз при старті, щоб не читати з диска при кожному запиті
# Це значно прискорює відповідь
try:
    model = model_service.load_model_for_inference()
except FileNotFoundError:
    print("ПОПЕРЕДЖЕННЯ: Файл моделі не знайдено. Запустіть /train-model для її створення.")
    model = None


@router.post("/predict", tags=["Inference"])
def predict(
        input_data: schemas.InferenceInput,
        db: Session = Depends(database.get_db)
):
    """
    Робить прогноз для нових даних.

    1. Приймає JSON з 10 ознаками.
    2. Завантажує навчену модель.
    3. Зберігає вхідні дані в 'inference_inputs'.
    4. Робить прогноз.
    5. Зберігає прогноз в 'predictions' (source='inference').
    6. Повертає результат.
    """
    global model
    if model is None:
        # Якщо модель не була завантажена при старті, пробуємо знову
        try:
            model = model_service.load_model_for_inference()
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # 1. Логуємо вхідні дані
    crud.create_inference_input(db, input_data)

    # 2. Робимо прогноз
    prediction_result = model_service.predict_single(model, input_data)

    # 3. Логуємо вихідні дані
    prediction_log = schemas.PredictionCreate(
        predicted_label=prediction_result,
        source="inference"  # true_label залишається None
    )
    crud.create_prediction(db, prediction_log)

    return {"predicted_obesity_level": prediction_result}