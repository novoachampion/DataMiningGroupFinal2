# export_metrics.py
import pandas as pd

def save_to_excel(data, metrics, correlation_matrix, filename):
    """Saves datasets and results to an Excel file with different sheets."""
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        data.to_excel(writer, sheet_name='Data', index=False)
        pd.DataFrame([metrics]).to_excel(writer, sheet_name='Metrics')
        correlation_matrix.to_excel(writer, sheet_name='Correlations')
        print(f"Excel file saved at {filename}")

def calculate_regression_metrics(model, X_test, y_test):
    """Calculates and returns regression metrics."""
    from sklearn.metrics import mean_squared_error, r2_score
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return {'MSE': mse, 'R2 Score': r2}
