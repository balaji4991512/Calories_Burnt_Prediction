# 🧠 Calories Burnt Prediction — Machine Learning Regression Project

## 📘 Project Overview

This project develops a **machine learning regression model** to predict the number of calories burnt during physical activities using biometric and exercise-related features such as gender, age, weight, height, heart rate, body temperature, and workout duration.

The primary goal is to demonstrate a **complete end-to-end ML workflow**, including data preparation, feature engineering, model comparison, and evaluation.

---

## 🎯 Objectives

* Predict calories burnt based on exercise and physiological parameters.
* Perform **Exploratory Data Analysis (EDA)** to understand feature relationships.
* Apply systematic **data preprocessing and feature scaling**.
* Compare and tune multiple regression algorithms using **GridSearchCV**.
* Evaluate the final model using standard regression metrics.
* Ensure no data leakage and maintain a reproducible workflow.

---

## 🧩 Dataset Information

The project uses two datasets:

| File           | Description                                                                                                    |
| -------------- | -------------------------------------------------------------------------------------------------------------- |
| `exercise.csv` | Contains user exercise records (User ID, Gender, Age, Height, Weight, Duration, Heart Rate, Body Temperature). |
| `calories.csv` | Contains the corresponding calorie values for each user (User ID, Calories).                                   |

The datasets are merged using the common key `User_ID`.

**Target variable:** `Calories`
**Feature variables:** `Gender`, `Age`, `Height`, `Weight`, `Duration`, `Heart_Rate`, `Body_Temp` (plus engineered feature `BMI`).

---

## 🔍 Exploratory Data Analysis (EDA)

* Checked dataset dimensions, column types, and summary statistics.
* Identified missing values and anomalies.
* Visualized feature distributions and correlations using heatmaps and pairplots.
* Observed strong relationships between workout duration, heart rate, and calorie burn.
* Created additional derived features such as **BMI (Body Mass Index)** for improved representation.

---

## ⚙️ Data Preprocessing Steps

| Step                          | Description                                                                                        |
| ----------------------------- | -------------------------------------------------------------------------------------------------- |
| **1. Data Cleaning**          | Removed irrelevant columns (e.g., `User_ID`) and handled whitespace or inconsistent string values. |
| **2. Encoding**               | Converted categorical columns (e.g., `Gender`) into numerical form (0 for male, 1 for female).     |
| **3. Feature Engineering**    | Created new features such as `BMI = Weight / (Height/100)^2`.                                      |
| **4. Missing Value Handling** | Filled missing numerical data using median imputation.                                             |
| **5. Feature Scaling**        | Standardized features using `StandardScaler` to normalize magnitude differences across columns.    |

Feature scaling ensures that all variables contribute equally to model training and prevents dominance of large-valued features in distance-based algorithms.

---

## 🧠 Model Selection and Hyperparameter Tuning

A variety of regression algorithms were compared using **GridSearchCV** with 5-fold cross-validation.

### Models Considered:

1. **Linear Regression**
2. **Ridge Regression**
3. **Lasso Regression**
4. **Decision Tree Regressor**
5. **Random Forest Regressor**
6. **XGBoost Regressor**

### Hyperparameter Tuning:

Each model was optimized using a parameter grid (e.g., `alpha`, `max_depth`, `n_estimators`, `learning_rate`) to find the configuration that minimizes **Mean Absolute Error (MAE)** on cross-validation folds.

A custom function `ModelSelection()` was used to:

* Iterate through models and parameter grids
* Perform GridSearchCV for each model
* Record the best hyperparameters and scores in a summary DataFrame

---

## 🧪 Model Evaluation

After model selection, the best-performing model was evaluated on a **held-out test set**.

### Evaluation Metrics:

* **Mean Absolute Error (MAE):** Average magnitude of errors (lower is better).
* **Root Mean Squared Error (RMSE):** Penalizes larger errors more strongly (lower is better).
* **R² Score:** Proportion of variance in target explained by features (closer to 1 indicates better fit).

A residual analysis and scatterplot (`Actual vs Predicted Calories`) were used to visualize model accuracy and detect bias.

---

## 🧱 Final Model Workflow

1. Split the dataset into **training (80%)** and **testing (20%)** before tuning.
2. Run **GridSearchCV** on the training data only.
3. Identify the **best model and hyperparameters**.
4. Retrain the selected model on the full training set.
5. Evaluate the model using unseen test data to ensure generalization.
6. (Optional) Retrain the final model on the **entire dataset** before deployment to maximize learning.

This approach ensures proper separation between training, tuning, and evaluation phases — preventing data leakage and giving an unbiased performance estimate.

---

## 🧰 Tools and Libraries

* **Language:** Python 3.8+
* **Core Libraries:** pandas, numpy, matplotlib, seaborn
* **ML Libraries:** scikit-learn, xgboost
* **Development Environment:** Jupyter Notebook / VS Code

---

## 🚀 How to Run

1. Clone the repository or download the project files.
2. Install required dependencies:

   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn xgboost
   ```
3. Execute the main script:

   ```bash
   python calories_burnt_prediction.py
   ```
4. Follow the printed logs and outputs:

   * EDA insights
   * Model comparison summary
   * Final evaluation metrics
   * Visualizations (correlation heatmap, actual vs predicted plot)

---

## 🧾 Best Practices Demonstrated

* Proper **train/test split before tuning** to avoid data leakage.
* **Cross-validation** for robust model comparison.
* Use of **scaling** to normalize feature ranges.
* Application of **GridSearchCV** for systematic hyperparameter optimization.
* **Interpretability** via metrics and residual analysis.
* **Reproducible workflow** with consistent random seeds and clear stages.

---

## 🔮 Future Enhancements

* Integrate the trained model into a **Streamlit or Flask app** for real-time calorie predictions.
* Extend the dataset to include exercise types (e.g., running, cycling, yoga).
* Explore **deep learning regression** (e.g., MLP Regressor) for non-linear relationships.
* Add **explainability methods** like SHAP or LIME for feature importance visualization.
