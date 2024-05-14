# knn_model.py
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train_knn(X, y, n_neighbors=5):
    """
    Trains a K Nearest Neighbors regression model and predicts on the test set.
    Args:
        X: Features dataset.
        y: Target dataset (continuous).
        n_neighbors: Number of neighbors to use by default for k neighbors queries.

    Returns:
        model: Trained KNN regression model.
        X_test: Test set features.
        y_test: Test set target.
        predictions: Predictions on the test set.
    """
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and fit the KNN regression model
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X_train_scaled, y_train)

    # Make predictions
    predictions = model.predict(X_test_scaled)

    return model, X_test_scaled, y_test, predictions
