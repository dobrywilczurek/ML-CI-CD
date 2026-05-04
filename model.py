import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def prepare_data():
    np.random.seed(42)
    X = np.array([
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
    y = (8000 * X[:, 0] + 20000 * X[:, 1] - 2000 * X[:, 2] +
         np.random.randint(-20000, 20000, size=10))
    # Podział na train (70%) i test (30%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def train_and_predict():
    """
    Trenuje model regresji liniowej na zbiorze treningowym,
    a następnie zwraca predykcje dla zbioru testowego oraz wytrenowany model.
    """
    X_train, X_test, y_train, _ = prepare_data()
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions, model

def get_accuracy():
    """
    Dla regresji 'dokładność' zastępujemy współczynnikiem R^2 (im bliżej 1, tym lepiej).
    Funkcja zwraca wartość R^2 na zbiorze testowym.
    """
    X_train, X_test, y_train, y_test = prepare_data()
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return r2_score(y_test, y_pred)