from pydantic import BaseModel
from typing import Optional

# Схема для JSON-тіла запиту на /predict
# (на основі 10 ознак з твого task 8.py)
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

# Схема для запису в таблицю inference_inputs
class InferenceInputCreate(InferenceInput):
    pass

# Схема для запису в таблицю predictions
class PredictionCreate(BaseModel):
    true_label: Optional[int] = None
    predicted_label: int
    source: str