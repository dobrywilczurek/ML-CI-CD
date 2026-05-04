import numpy as np
from model import train_and_predict, get_accuracy

def test_predictions_not_none():
    preds, _ = train_and_predict()
    assert preds is not None, "Predictions should not be None."

def test_predictions_length():
    preds, _ = train_and_predict()
    # Liczba próbek testowych = 30% z 10 próbek = 3 (po podziale z random_state=42)
    expected_len = 3   # ponieważ test_size=0.3, a zbiór ma 10 elementów
    assert len(preds) > 0, "Lista predykcji jest pusta."
    assert len(preds) == expected_len, f"Oczekiwano {expected_len} predykcji, otrzymano {len(preds)}."

def test_predictions_value_range():
    preds, _ = train_and_predict()
    for val in preds:
        assert val > 0, f"Predykcja {val} nie jest dodatnia."
        assert val < 1_500_000, f"Predykcja {val} jest podejrzanie wysoka."

def test_model_accuracy():
    r2 = get_accuracy()
    assert r2 >= 0.7, f"R^2 = {r2:.3f} jest mniejsze niż wymagane 0.7."