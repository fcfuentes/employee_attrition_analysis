import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_and_preprocess_data(file_path):
    employee_data = pd.read_csv(file_path)

    X = employee_data.drop("Attrition", axis=1)
    y = employee_data["Attrition"]

    ordered_categorical_features = [
        "Education",
        "JobLevel",
        "StockOptionLevel",
        "JobSatisfaction",
        "EnvironmentSatisfaction",
        "WorkLifeBalance",
        "JobInvolvement",
        "PerformanceRating",
    ]

    unordered_categorical_features = X.select_dtypes(include=["object"]).columns
    continuous_features = X.select_dtypes(
        include=["float64", "int64"]
    ).columns.difference(ordered_categorical_features)

    assert len(ordered_categorical_features) + len(continuous_features) + len(
        unordered_categorical_features
    ) == (employee_data.shape[1] - 1)

    scaler = StandardScaler()
    ordinal_encoder = OrdinalEncoder()
    encoder = OneHotEncoder(drop="first", sparse=False)

    X_continuous = scaler.fit_transform(X[continuous_features])
    X_ordered_categorical = ordinal_encoder.fit_transform(
        X[ordered_categorical_features]
    )
    X_categorical = encoder.fit_transform(X[unordered_categorical_features])
    X_preprocessed = np.hstack((X_continuous, X_ordered_categorical, X_categorical))
    all_features_names = (
        list(continuous_features)
        + list(ordered_categorical_features)
        + list(encoder.get_feature_names_out(unordered_categorical_features))
    )
    all_features_df = pd.DataFrame(X_preprocessed, columns=all_features_names)

    return all_features_df, y


def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    smote = SMOTE(random_state=35)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    classifiers = {
        "RandomForest": RandomForestClassifier(random_state=35),
        "GradientBoosting": GradientBoostingClassifier(random_state=35),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=35),
        "SVC": SVC(random_state=35),
        "KNeighbors": KNeighborsClassifier(),
        "DecisionTree": DecisionTreeClassifier(random_state=35),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=35)
    model_performance = {}

    for model_name, model in classifiers.items():
        scores = cross_val_score(
            model, X_resampled, y_resampled, cv=cv, scoring="accuracy"
        )
        model_performance[model_name] = scores.mean()
        logging.info(f"{model_name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
        evaluate_model(model_name, model, X_resampled, y_resampled, X_test, y_test)

    return model_performance


def evaluate_model(model_name, model, X_resampled, y_resampled, X_test, y_test):
    model.fit(X_resampled, y_resampled)
    y_train_pred = model.predict(X_resampled)
    y_test_pred = model.predict(X_test)

    logging.info(f"Model performance for {model_name}")
    logging.info(f"Training Accuracy: {accuracy_score(y_resampled, y_train_pred):.2f}")
    logging.info("Training Classification Report:")
    logging.info("\n" + classification_report(y_resampled, y_train_pred))

    logging.info(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.2f}")
    logging.info("Test Classification Report:")
    logging.info("\n" + classification_report(y_test, y_test_pred))
    logging.info("----------------------------------------------------------------")


def perform_grid_search(X_resampled, y_resampled):
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10, 15],
        "min_samples_leaf": [1, 2, 4, 6],
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=35),
        param_grid=param_grid,
        cv=10,
        n_jobs=-1,
        scoring="accuracy",
    )
    grid_search.fit(X_resampled, y_resampled)

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    logging.info(f"Best parameters: {best_params}")
    logging.info(f"Best model: {best_model}")

    return best_model


def plot_feature_importances(feature_importance_df):
    top_features_df = feature_importance_df.head(10)
    plt.figure(figsize=(14, 10))
    bars = plt.barh(
        top_features_df["Feature"], top_features_df["Importance"], color="dodgerblue"
    )

    plt.xlabel("")
    plt.ylabel("")

    plt.title("Top 10 Feature Importances", fontsize=18, fontweight="bold")
    plt.gca().invert_yaxis()

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)

    plt.gca().tick_params(
        axis="y", which="both", left=False, right=False, labelleft=True
    )
    plt.gca().tick_params(
        axis="x", which="both", top=False, bottom=False, labelbottom=False
    )

    for bar in bars:
        plt.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.4f}",
            va="center",
            fontsize=12,
            color="black",
        )
    plt.savefig("results/top_features_importance.png", bbox_inches="tight", dpi=300)
    plt.show()


def save_feature_importances(
    model, all_features_names, output_file="feature_importances.csv"
):
    feature_importances = model.feature_importances_
    print(f"{len(feature_importances)} and {len(all_features_names)}")
    feature_importance_df = pd.DataFrame(
        {"Feature": all_features_names, "Importance": feature_importances}
    )
    feature_importance_df = feature_importance_df.sort_values(
        by="Importance", ascending=False
    )

    logging.info(feature_importance_df)
    feature_importance_df.to_csv(output_file, index=False)

    logging.info(f"Successfully saved feature importances to {output_file}")
    return feature_importance_df


def calculate_vif(X, threshold):

    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i) for i in range(X.shape[1])
    ]

    while vif_data["VIF"].max() > threshold:
        feature_to_remove = vif_data.sort_values("VIF", ascending=False).iloc[0]
        logging.info(
            f"Removing feature '{feature_to_remove.Feature}' with VIF {feature_to_remove.VIF:.2f}"
        )

        # Remove the feature from the DataFrame
        X = X.drop(columns=[feature_to_remove.Feature])

        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [
            variance_inflation_factor(X.values, i) for i in range(X.shape[1])
        ]

    return X, X.columns


def plot_confusion_matrix(model, X_test, y_test, output_file):
    plt.figure(figsize=(10, 7))
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["No Attrition", "Attrition"]
    )
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(output_file)
    logging.info(f"Confusion matrix saved to {output_file}")
    plt.show()


def plot_roc_curve(model, X_test, y_test, output_file):
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = roc_auc_score(y_test, y_score)

    plt.figure(figsize=(10, 7))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig(output_file)
    logging.info(f"ROC curve saved to {output_file}")
    plt.show()
