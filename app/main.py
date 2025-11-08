from fastapi import FastAPI
from app.database import database, models
from app.endpoints import training, inference

# Створюємо таблиці в БД, якщо їх ще немає
models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(
    title="Obesity Prediction API",
    description="Лабораторна робота 4. Розгортання моделі XGBoost.",
    version="1.0"
)

# Підключення роутерів
app.include_router(training.router, prefix="/api")
app.include_router(inference.router, prefix="/api")

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Ласкаво просимо до API для прогнозування ожиріння!"}