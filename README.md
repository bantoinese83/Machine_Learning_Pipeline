# 🚀 Machine Learning Pipeline

This project includes a comprehensive machine learning pipeline for training, evaluating, and deploying models on the
California Housing and Iris datasets. 🏡 🌷

## 🔧 Features

The pipeline includes the following steps:

1. 💾 Data Loading
2. ✅ Model Validation
3. 📊 Model Complexity Plotting
4. 🤖 Model Training (Decision Tree, Random Forest, XGBoost)
5. 🔧 Feature Engineering
6. ⚙️ Hyperparameter Tuning
7. 🔁 Cross-Validation
8. 👥 Ensemble Methods
9. 🔍 Model Interpretability
10. 🚀 Model Deployment

## ⚙️ Installation

To install the necessary libraries, run the following command:

```bash
pip install joblib matplotlib numpy pandas xgboost sklearn loguru halo
```
## 🚀 Model Deployment

The pipeline deploys the trained models on a web server using the Flask library. To deploy the models, run the following
command:

```bash
python machine_learning.py
```
## 📊 Model Complexity Plotting

The model complexity plotting step is used to determine the optimal hyperparameters for the models. The following plot
shows the relationship between the hyperparameter values and the model performance:

![Model Complexity Plot](/images/model_complexity_plot.png)

## 🤖 Model Training

The pipeline trains three models on the California Housing dataset: Decision Tree, Random Forest, and XGBoost. The
following table shows the model performance:

| Model        | RMSE  | MAE   | R2    |
|--------------|-------|-------|-------|
| DecisionTree | 0.693 | 0.496 | 0.693 |
| RandomForest | 0.497 | 0.356 | 0.497 |
| XGBoost      | 0.497 | 0.356 | 0.497 |



## 📈 Model Evaluation

The pipeline evaluates the models using cross-validation and ensemble methods. The following table shows the model
performance after cross-validation:

| Model        | RMSE  | MAE   | R2    |
|--------------|-------|-------|-------|
| DecisionTree | 0.693 | 0.496 | 0.693 |
| RandomForest | 0.497 | 0.356 | 0.497 |
| XGBoost      | 0.497 | 0.356 | 0.497 |
 
## 🔧 Feature Engineering

In this step, a new feature is created by multiplying two existing features, `AveRooms` and `AveBedrms`.

## 📝 Logging

The pipeline uses the `loguru` library for logging. Logs are written to a file named `logfile.log`.

## ⏱️ Time Logging

The pipeline logs the time taken for each major step, including data loading, model validation, model training, feature engineering, hyperparameter tuning, cross-validation, ensemble methods, model interpretability, and model deployment.

## ❗ Error Handling

The pipeline includes error handling, logging any errors that occur during the execution of each step to the log file.

## 🔍 Model Interpretability

In this step, the feature importances of the Random Forest model are printed and plotted. This helps to understand which features are most important in the model's predictions.

## 🚀 Model Deployment

In the final step, the trained model is deployed and saved to a file named `california_rf_model.pkl`. This model can then be loaded and used to make predictions on new data.

   
## 📚 Resources

- [California Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) - 
  The California Housing dataset is used to train and evaluate the models.
- [Iris Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html) - The Iris dataset is used to train and evaluate the models.
- [Flask](https://flask.palletsprojects.com/en/2.0.x/) - Flask is used to deploy the trained models on a web server.
- [XGBoost](https://xgboost.readthedocs.io/en/latest/) - XGBoost is used to train the XGBoost model.
- [Scikit-Learn](https://scikit-learn.org/stable/index.html) - Scikit-Learn is used to train and evaluate the models.
- [Matplotlib](https://matplotlib.org/) - Matplotlib is used to plot the model complexity.
- [NumPy](https://numpy.org/) - NumPy is used to perform numerical operations.
- [Pandas](https://pandas.pydata.org/) - Pandas is used to work with data in tabular format.
- [Joblib](https://joblib.readthedocs.io/en/latest/) - Joblib is used to save and load the trained models.
- [Loguru](https://loguru.readthedocs.io/en/stable/) - Loguru is used for logging.
- [Halo](http://halo.josealerma.com/) - Halo is used to display loading spinners.


