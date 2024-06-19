import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from preprocessing import get_preprocessing_employee_data, get_employee_timesheet_data, save_clean_data
from train_evaluate_model import (
    load_and_preprocess_data,
    train_and_evaluate_models,
    perform_grid_search,
    save_feature_importances,
    plot_feature_importances,
    calculate_vif,
    plot_confusion_matrix,
    plot_roc_curve,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    # Preprocess data
    logging.info('Starting data preprocessing...')
    general_data = pd.read_csv("input_data/general_data.csv")
    employee_survey = pd.read_csv("input_data/employee_survey_data.csv")
    manager_survey = pd.read_csv("input_data/manager_survey_data.csv")
    in_time_data = pd.read_csv("input_data/in_time.csv")
    out_time_data = pd.read_csv("input_data/out_time.csv")

    employee_data = get_preprocessing_employee_data(general_data, employee_survey, manager_survey)
    employee_timesheet_data = get_employee_timesheet_data(in_time_data, out_time_data)
    save_clean_data(employee_data, employee_timesheet_data, 'preprocessed_data/clean_data_01.csv')
    logging.info('Data preprocessing completed.')

    logging.info("Starting model training and evaluation...")
    all_features_df, y = load_and_preprocess_data("preprocessed_data/clean_data_01.csv")

    # Since some of the features have high colinearity, we would like to make sure to remove those features not to impact the performance of the model and overfitting.
    X_reduced, all_features_names = calculate_vif(all_features_df, threshold=10)
    X_preprocessed_reduced = X_reduced.values

    X_train, X_test, y_train, y_test = train_test_split(
        X_preprocessed_reduced, y, test_size=0.3, random_state=35, stratify=y
    )

    model_performance = train_and_evaluate_models(X_train, y_train, X_test, y_test)
    best_model_name = max(model_performance, key=model_performance.get)

    logging.info(
        f"\nBest model: {best_model_name} with accuracy: {model_performance[best_model_name]:.4f}"
    )
    logging.info(f"Performing grid search to tune best model performance")
    best_model = perform_grid_search(X_train, y_train)

    y_test_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_report = classification_report(y_test, y_test_pred)

    logging.info(f"Test Accuracy of Best Model: {test_accuracy:.2f}")
    logging.info("Test Classification Report of Best Model:")
    logging.info("\n" + test_report)

    feature_importance_df = save_feature_importances(
        best_model, all_features_names, "results/feature_importances.csv"
    )
    plot_feature_importances(feature_importance_df)
    plot_confusion_matrix(
        best_model, X_test, y_test, output_file="results/confusion_matrix.png"
    )
    plot_roc_curve(best_model, X_test, y_test, output_file="results/roc_curve.png")
    logging.info("Model training and evaluation completed.")


if __name__ == "__main__":
    main()
