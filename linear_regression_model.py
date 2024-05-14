import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from data_loader import load_data, preprocess_data

def train_linear_regression(data_path, target_column):
    data = load_data(data_path)
    X, y = preprocess_data(data, target_column)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return model, X_test, y_test, predictions, mse, r2

if __name__ == "__main__":
    model, X_test, y_test, predictions, mse, r2 = train_linear_regression('data/all_period.csv', 'Systematic Risk')
    print(f"MSE: {mse}, R2 Score: {r2}")
