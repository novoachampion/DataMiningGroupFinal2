# config.py
MODELS = {
    'linear_regression': {
        'module': 'model_trainer',
        'model_function': 'train_linear_regression',
        'target': 'Systematic Risk'
    },
    'random_forest': {
        'module': 'random_forest_model',
        'model_function': 'train_random_forest',
        'target': 'Systematic Risk',
        'task': 'regression',
        'n_estimators': 100
    },
    'knn': {
        'module': 'knn_model',
        'model_function': 'train_knn',
        'target': 'Systematic Risk',
        'n_neighbors': 5
    }
}

PLOTS_OUTPUT_DIR = 'EDA'
DATA_FILE = 'data/all_period.csv'
