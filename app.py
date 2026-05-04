from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
from sklearn.linear_model import LinearRegression
import uvicorn
import os

API_KEY = os.getenv("API_KEY", "brak_klucza")
DATABASE_URL = os.getenv("DATABASE_URL", "brak_bazy")

# basic api
app = FastAPI(
    title="API Apartment price prediction",
    description="API for predicting apartment prices",
    version="1.0.0"
)


@app.get("/")
def read_root():
    """Endpoint główny"""
    return {"message": "Welcome to predicting apartment prices API!"}


# Cechy: [powierzchnia_m2, liczba_pokoi, wiek_budynku_lata]
np.random.seed(42)
X_train = np.array([
    [50, 2, 10],
    [60, 2, 5],
    [70, 3, 15],
    [80, 3, 8],
    [90, 4, 20],
    [100, 4, 12],
    [45, 1, 30],
    [120, 5, 2],
    [55, 2, 25],
    [85, 3, 18],
])

# Ceny: 3000 zł/m2 * powierzchnia + bonus za pokoje - kara za wiek
y_train = (8000 * X_train[:, 0] + 20000 * X_train[:, 1] - 2000 * X_train[:, 2] + np.random.randint(-20000, 20000,
                                                                                                   size=10))

# Trenowanie modelu
model = LinearRegression()
model.fit(X_train, y_train)


class HouseFeatures(BaseModel):
    powierzchnia_m2: float = Field(..., gt=0, description="Powierzchnia w metrach kwadratowych")
    liczba_pokoi: int = Field(..., ge=1, le=10, description="Liczba pokoi (1-10)")
    wiek_budynku_lata: int = Field(..., ge=0, le=200, description="Wiek budynku w latach")

    class Config:
        json_schema_extra = {
            "example": {
                "powierzchnia_m2": 75.5,
                "liczba_pokoi": 3,
                "wiek_budynku_lata": 12
            }
        }

@app.get("/config")
def show_config():
    return {
        "api_key": API_KEY,
        "database_url": DATABASE_URL,
        "message": "Zmienne środowiskowe działają"
    }

@app.post("/predict")
def predict(features: HouseFeatures):
    """
    Endpoint predykcji ceny mieszkania.

    Przyjmuje cechy mieszkania i zwraca przewidywaną cenę.
    """
    try:
        # Przygotowanie danych do predykcji
        input_data = np.array([[features.powierzchnia_m2, features.liczba_pokoi, features.wiek_budynku_lata]])

        # Wykonanie predykcji
        prediction = model.predict(input_data)[0]

        return {
            "predicted_price": round(prediction, 2),
            "currency": "PLN",
            "input_features": features.model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd podczas predykcji: {str(e)}")


@app.get("/info")
def get_model_info():
    return {
        "model_type": "LinearRegression",
        "algorithm": "Regresja liniowa",
        "features": ["powierzchnia_m2", "liczba_pokoi", "wiek_budynku_lata"],
        "n_features": 3,
        "coefficients": {
            "powierzchnia_m2": round(model.coef_[0], 2),
            "liczba_pokoi": round(model.coef_[1], 2),
            "wiek_budynku_lata": round(model.coef_[2], 2)
        },
        "intercept": round(model.intercept_, 2),
        "description": "Model predicts apartment prices based on 3 attributes",
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "API predykcji cen mieszkań"}


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # tryb deweloperski
    )