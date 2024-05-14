import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data, target_column):
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    return X, y

def train_knn(X, y, n_neighbors=5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X_train_scaled, y_train)
    return knn, X_test_scaled, y_test

def evaluate_and_visualize(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    explained_variance = explained_variance_score(y_test, predictions)

    # Creating DataFrame for metrics
    metrics_length = len(y_test)
    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': predictions,
        'MSE': [mse] * metrics_length,
        'MAE': [mae] * metrics_length,
        'R2': [r2] * metrics_length,
        'Explained Variance': [explained_variance] * metrics_length
    })
    results_df.to_excel('EDA/knn_results.xlsx', index=False)
    
    # Generate and save a scatter plot
    output_file('EDA/knn_scatter_plot.html')
    source = ColumnDataSource(data=dict(actual=y_test, predicted=predictions))
    p = figure(title="KNN Test vs Predictions", x_axis_label='Actual', y_axis_label='Predicted')
    p.circle('actual', 'predicted', source=source, size=10, color='navy', alpha=0.5)
    save(p)

def main():
    data = load_data('data/all_period.csv')
    X, y = preprocess_data(data, 'Systematic Risk')
    model, X_test, y_test = train_knn(X, y)
    evaluate_and_visualize(model, X_test, y_test)

if __name__ == "__main__":
    main()
