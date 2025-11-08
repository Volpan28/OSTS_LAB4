from pydantic import BaseModel
from typing import Optional

# Схема для JSON-тіла запиту на /predict
# Вона повинна містити всі ознаки, які потрібні моделі
class InferenceInput(BaseModel):
    Weight: float
    FCVC: float
    Height: float
    CH2O: float
    Age: float
    FAF: float
    CAEC: float
    NCP: float
    SCC: float
    CALC: float

# Схема для створення запису в таблиці inference_inputs
class InferenceInputCreate(InferenceInput):
    pass

# Схема для створення запису в таблиці predictions
class PredictionCreate(BaseModel):
    true_label: Optional[int] = None
    predicted_label: int
    source: str