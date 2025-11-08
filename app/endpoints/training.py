from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import database, crud
from app.services import model as model_service

router = APIRouter()


@router.post("/train-model", tags=["Training"])
def train_model(db: Session = Depends(database.get_db)):
    """
    Запускає процес тренування моделі.

    1. Читає всі дані з таблиці 'obesity_data'.
    2. Ділить на train (90%) та new_input (10%).
    3. Навчає XGBoost на train-даних.
    4. Зберігає модель у 'models/xgboost_optimized.pkl'.
    5. Зберігає прогнози на train-даних у таблицю 'predictions'.
    """
    try:
        _, result = model_service.train_and_save_model(db)
        return result
    except Exception as e:
        # Обробка помилки, якщо БД порожня
        if "n_samples=0" in str(e) or "empty" in str(e):
            raise HTTPException(status_code=400,
                                detail=f"Помилка тренування: {str(e)}. Переконайтеся, що таблиця 'obesity_data' наповнена даними, запустивши src/ImportToDb.py.")
        raise HTTPException(status_code=500, detail=f"Внутрішня помилка сервера: {str(e)}")