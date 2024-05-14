# random_forest_model.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def train_random_forest(X, y, task='regression', n_estimators=100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if task == 'regression':
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, X_test, y_test, predictions
