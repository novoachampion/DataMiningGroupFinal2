# main.py
from data_loader import load_data, preprocess_data
import model_trainer, random_forest_model, knn_model
from visualizations import generate_scatter_plots
from export_metrics import save_to_excel, calculate_regression_metrics
import config
import pandas as pd

def main():
    data = load_data(config.DATA_FILE)
    features = data.drop(config.MODELS['linear_regression']['target'], axis=1).columns.tolist()  # Get features from the full dataset
    for model_key, model_info in config.MODELS.items():
        print(f"Processing model: {model_key}")
        X, y = preprocess_data(data, model_info['target'])
        
        # Dynamically access the module and function
        module = globals()[model_info['module']]
        model_function = getattr(module, model_info['model_function'])
        model, X_test, y_test, predictions = model_function(X, y, **{k: v for k, v in model_info.items() if k not in ['module', 'model_function', 'target']})

        # Check if X_test is a DataFrame or a numpy array
        if isinstance(X_test, pd.DataFrame):
            feature_names = X_test.columns.tolist()
        else:  # Assuming it's a numpy array, use the earlier derived feature list
            feature_names = features
        
        generate_scatter_plots(X_test, feature_names, f"{config.PLOTS_OUTPUT_DIR}/{model_key}_scatter_plots.html")

        metrics = calculate_regression_metrics(model, X_test, y_test)
        print(f"Metrics for {model_key}: {metrics}")
        save_to_excel(data, metrics, data.corr(), f"{config.PLOTS_OUTPUT_DIR}/{model_key}_model_analysis.xlsx")

if __name__ == '__main__':
    main()
