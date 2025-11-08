from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.sql import func
from app.database.database import Base  # <--- Виправлено імпорт


# 1. Існуюча таблиця з даними (для читання)
class ObesityData(Base):
    __tablename__ = "obesity_data"

    id = Column(Integer, primary_key=True, index=True)
    Weight = Column(Float)
    FCVC = Column(Float)
    Height = Column(Float)
    CH2O = Column(Float)
    Age = Column(Float)
    FAF = Column(Float)
    CAEC = Column(Float)
    NCP = Column(Float)
    SCC = Column(Float)
    CALC = Column(Float)
    NObeyesdad = Column(Integer)  # Цільова змінна


# 2. НОВА ТАБЛИЦЯ: Зберігає вхідні дані з /predict
class InferenceInputs(Base):
    __tablename__ = "inference_inputs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    # Ознаки, які прийшли в JSON-запиті
    Weight = Column(Float)
    FCVC = Column(Float)
    Height = Column(Float)
    CH2O = Column(Float)
    Age = Column(Float)
    FAF = Column(Float)
    CAEC = Column(Float)
    NCP = Column(Float)
    SCC = Column(Float)
    CALC = Column(Float)


# 3. НОВА ТАБЛИЦЯ: Зберігає всі прогнози (train + inference)
class Predictions(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    # Може бути NULL, якщо джерело "inference"
    true_label = Column(Integer, nullable=True)
    predicted_label = Column(Integer)

    # 'train' (з /train-model) або 'inference' (з /predict)
    source = Column(String)