import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from loguru import logger
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from halo import Halo

spinner = Halo(text='Loading', spinner='dots')


def load_data():
    """
    Load the California Housing and Iris datasets.

    Returns:
        tuple: A tuple containing the California Housing DataFrame and the Iris DataFrame.
    """
    logger.info("Starting to load data...")
    try:
        with spinner:
            logger.info("Loading datasets...")
            california = fetch_california_housing()
            iris = load_iris()

            california_df = pd.DataFrame(california.data, columns=california.feature_names)
            california_df['TARGET'] = california.target

            iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
            iris_df['TARGET'] = iris.target

            logger.info("Datasets loaded.")
            logger.info(f"California dataset shape: {california_df.shape}")
            logger.info(f"Iris dataset shape: {iris_df.shape}")
        return california_df, iris_df
    except Exception as error:
        logger.error(f"Failed to load datasets: {error}")
        return None, None
    finally:
        logger.info("Finished loading data.")


def model_validation(X_train_california, y_train_california, X_test_california, y_test_california, X_train_iris,
                     y_train_iris, X_test_iris, y_test_iris):
    """Train Decision Tree models and evaluate performance"""
    logger.info("Starting model validation...")
    try:
        logger.info("Training Decision Tree models...")
        california_model = DecisionTreeRegressor(random_state=42)
        california_model.fit(X_train_california, y_train_california)
        y_california_pred = california_model.predict(X_test_california)

        iris_model = DecisionTreeClassifier(random_state=42)
        iris_model.fit(X_train_iris, y_train_iris)
        y_iris_pred = iris_model.predict(X_test_iris)

        california_mae = mean_absolute_error(y_test_california, y_california_pred)
        iris_accuracy = accuracy_score(y_test_iris, y_iris_pred)

        logger.info(f"California Model Mean Absolute Error: {california_mae}")
        logger.info(f"Iris Model Accuracy: {iris_accuracy}")
    finally:
        logger.info("Finished model validation.")


def plot_model_complexity(model, X_train, y_train, X_test, y_test, param_name, param_values):
    """Plot the effect of a hyperparameter on model performance"""
    logger.info("Plotting model complexity...")
    try:
        train_scores = []
        test_scores = []

        for param_value in param_values:
            model.set_params(**{param_name: param_value})
            model.fit(X_train, y_train)
            train_scores.append(model.score(X_train, y_train))
            test_scores.append(model.score(X_test, y_test))

        plt.figure(figsize=(10, 5))
        plt.plot(param_values, train_scores, label="Training Score")
        plt.plot(param_values, test_scores, label="Testing Score")
        plt.xlabel(param_name)
        plt.ylabel("Score")
        plt.legend()
        plt.title(f"{param_name} vs Performance")
        plt.show()
    except Exception as error:
        logger.error(f"Failed to plot model complexity: {error}")


def random_forest_model(X_train_california, y_train_california, X_test_california, y_test_california, X_train_iris,
                        y_train_iris, X_test_iris, y_test_iris):
    """Train Random Forest models and evaluate performance"""
    logger.info("Training Random Forest models...")
    try:
        # Create a pipeline
        pipeline_california = Pipeline([
            ('scaler', StandardScaler()),  # Standardize features
            ('rf', RandomForestRegressor(random_state=42))  # RandomForestRegressor model
        ])

        # Fit the pipeline
        pipeline_california.fit(X_train_california, y_train_california)

        # Save the entire pipeline
        joblib.dump(pipeline_california, "california_rf_model.pkl")

        y_california_rf_pred = pipeline_california.predict(X_test_california)

        iris_rf_classifier = RandomForestClassifier(random_state=42)
        iris_rf_classifier.fit(X_train_iris, y_train_iris)
        y_iris_rf_pred = iris_rf_classifier.predict(X_test_iris)

        california_rf_mae = mean_absolute_error(y_test_california, y_california_rf_pred)
        iris_rf_accuracy = accuracy_score(y_test_iris, y_iris_rf_pred)

        logger.info(f"Random Forest California Model Mean Absolute Error: {california_rf_mae}")
        logger.info(f"Random Forest Iris Model Accuracy: {iris_rf_accuracy}")
    except Exception as error:
        logger.error(f"Failed to train Random Forest models: {error}")


def xgboost_model(X_train_california, y_train_california, X_test_california, y_test_california, X_train_iris,
                  y_train_iris, X_test_iris, y_test_iris):
    """Train XGBoost models and evaluate performance"""
    logger.info("Training XGBoost models...")
    try:
        # Create a pipeline to prevent data leakage
        pipeline_california = Pipeline([
            ('scaler', StandardScaler()),  # Standardize features
            ('xgb', xgb.XGBRegressor(objective='reg:squarederror', random_state=42))
        ])

        pipeline_california.fit(X_train_california, y_train_california)
        y_california_xgb_pred = pipeline_california.predict(X_test_california)

        # Save the fitted scaler
        scaler = pipeline_california.named_steps['scaler']
        joblib.dump(scaler, 'scaler.pkl')

        pipeline_iris = Pipeline([
            ('scaler', StandardScaler()),  # Standardize features
            ('xgb', xgb.XGBClassifier(random_state=42))
        ])

        pipeline_iris.fit(X_train_iris, y_train_iris)
        y_iris_xgb_pred = pipeline_iris.predict(X_test_iris)

        california_xgb_mae = mean_absolute_error(y_test_california, y_california_xgb_pred)
        iris_xgb_accuracy = accuracy_score(y_test_iris, y_iris_xgb_pred)

        logger.info(f"XGBoost California Model Mean Absolute Error: {california_xgb_mae}")
        logger.info(f"XGBoost Iris Model Accuracy: {iris_xgb_accuracy}")
    except Exception as error:
        logger.error(f"Failed to train XGBoost models: {error}")


def feature_engineering(california_df):
    """Create a new feature by multiplying two existing features"""
    logger.info("Performing feature engineering...")
    try:
        if 'AveRooms' in california_df.columns and 'AveBedrms' in california_df.columns:
            california_df['NEW_FEATURE'] = california_df['AveRooms'] * california_df['AveBedrms']
            logger.info("Feature engineering completed.")
        else:
            logger.info("AveRooms or AveBedrms not found in DataFrame.")
        logger.info(f"Updated California dataset columns: {california_df.columns}")
    except Exception as error:
        logger.error(f"Feature engineering failed: {error}")


def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning using GridSearchCV"""
    logger.info("Performing hyperparameter tuning...")
    try:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5,
                                   scoring='neg_mean_absolute_error')
        grid_search.fit(X_train, y_train)

        logger.info("Best Hyperparameters for Random Forest:")
        logger.info(grid_search.best_params_)
    except Exception as error:
        logger.error(f"Hyperparameter tuning failed: {error}")


def cross_validation(X, y):
    """Perform cross-validation to evaluate model performance"""
    logger.info("Performing cross-validation...")
    try:
        rf_scores = cross_val_score(RandomForestRegressor(random_state=42), X, y, cv=5,
                                    scoring='neg_mean_absolute_error')
        logger.info(f"Mean Absolute Error (Cross-Validation) - Random Forest: {-rf_scores.mean()}")
    except Exception as error:
        logger.error(f"Cross-validation failed: {error}")


def ensemble_methods(X_train, y_train, X_test, y_test):
    """Train an ensemble of Random Forest models and evaluate performance"""
    logger.info("Training ensemble of Random Forest models...")
    try:
        n_estimators = 5
        models = [RandomForestRegressor(random_state=42) for _ in range(n_estimators)]
        for i, model in enumerate(models):
            model.fit(X_train, y_train)

        ensemble_predictions = np.array([model.predict(X_test) for model in models])
        ensemble_mean = np.mean(ensemble_predictions, axis=0)
        ensemble_mae = mean_absolute_error(y_test, ensemble_mean)

        logger.info(f"Ensemble Mean Absolute Error: {ensemble_mae}")
    except Exception as error:
        logger.error(f"Ensemble methods failed: {error}")


def model_interpretability(model_rf, data):
    """Print feature importances and plot them"""
    logger.info("Printing feature importances for the Random Forest model...")
    try:
        feature_importances = model_rf.feature_importances_
        for feature, importance in zip(data.columns, feature_importances):
            logger.info(f"{feature}: {importance}")

        # Plotting feature importances
        plt.figure(figsize=(10, 5))
        sorted_indices = np.argsort(feature_importances)[::-1]
        plt.bar(range(data.shape[1]), feature_importances[sorted_indices], align='center')
        plt.xticks(range(data.shape[1]), data.columns[sorted_indices], rotation=90)

        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.title('Feature Importances')
        plt.show()
    except Exception as error:
        logger.error(f"Failed to interpret model: {error}")


def deployment(final_model, X, y, filename):
    """Deploy the final model and save it to a file"""
    logger.info("Deploying the final model...")
    try:
        final_model.fit(X, y)
        joblib.dump(final_model, filename)
        logger.info("Final model deployed and saved to file.")
    except Exception as error:
        logger.error(f"Failed to deploy the final model: {error}")


def log_time(time_start, process_name):
    elapsed_time = time.time() - time_start
    logger.info(f"Time taken for {process_name}: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    logger.add("logfile.log", rotation="10 MB", level="INFO", enqueue=True)
    logger.info("Starting Machine Learning pipeline...")

    try:
        start_time = time.time()
        logger.info("Loading data...")
        california_data, iris_data = load_data()
        log_time(start_time, "loading data")

        X_california = california_data.drop(columns=['TARGET'])
        y_california = california_data['TARGET']

        X_iris = iris_data.drop(columns=['TARGET'])
        y_iris = iris_data['TARGET']

        logger.info("Splitting data...")
        X_california_train, X_california_test, y_california_train, y_california_test = train_test_split(X_california,
                                                                                                        y_california,
                                                                                                        test_size=0.2,
                                                                                                        random_state=42)
        X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(X_iris, y_iris, test_size=0.2,
                                                                                random_state=42)
        start_time = time.time()
        logger.info("Validating model...")
        with spinner:
            model_validation(X_california_train, y_california_train, X_california_test, y_california_test, X_iris_train,
                             y_iris_train, X_iris_test, y_iris_test)
        log_time(start_time, "model validation")

        logger.info("Plotting model complexity...")
        with spinner:
            plot_model_complexity(DecisionTreeRegressor(random_state=42), X_california_train, y_california_train,
                                  X_california_test, y_california_test, 'max_depth', range(1, 11))

        logger.info("Training Random Forest model...")
        with spinner:
            random_forest_model(X_california_train, y_california_train, X_california_test, y_california_test,
                                X_iris_train,
                                y_iris_train, X_iris_test, y_iris_test)

        logger.info("Training XGBoost model...")
        with spinner:
            xgboost_model(X_california_train, y_california_train, X_california_test, y_california_test, X_iris_train,
                          y_iris_train, X_iris_test, y_iris_test)

        logger.info("Performing feature engineering...")
        with spinner:
            feature_engineering(california_data)

        logger.info("Tuning hyperparameters...")
        with spinner:
            hyperparameter_tuning(X_california_train, y_california_train)

        logger.info("Performing cross validation...")
        with spinner:
            cross_validation(X_california, y_california)

        logger.info("Training ensemble methods...")
        with spinner:
            ensemble_methods(X_california_train, y_california_train, X_california_test, y_california_test)

        logger.info("Interpreting model...")
        with spinner:
            california_rf_model = RandomForestRegressor(random_state=42)
            california_rf_model.fit(X_california_train, y_california_train)
            model_interpretability(california_rf_model, X_california)

        start_time = time.time()
        logger.info("Deploying model...")
        with spinner:
            deployment(california_rf_model, X_california, y_california, "california_rf_model.pkl")
        log_time(start_time, "deployment")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

    logger.info("Machine Learning pipeline completed.")
