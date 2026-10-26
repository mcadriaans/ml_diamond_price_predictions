# Predicting Diamond Prices: A Comprehensive Regression Analysis
<img width="479" height="329" alt="image" src="https://github.com/user-attachments/assets/fc710385-24f6-4095-9b47-11ca6963952e" />


## Project Overview

This project aims to develop a machine learning model that predicts diamond prices with both high accuracy and clear interpretability. Using a rich Kaggle dataset of 53,940 entries, the analysis explores how quality factors like cut, color, and clarity interact with size-related features such as carat, depth, and dimensions. The workflow includes thorough data cleaning, deep exploratory analysis, smart feature engineering, a comparison of regression models, Bayesian hyperparameter tuning, and transparent model interpretation using SHAP. The end goal is a reliable, explainable model that reveals what truly drives diamond pricing.

## Key Features

*   **End-to-End ML Pipeline:** Covers data ingestion, cleaning, EDA, preprocessing, model building, tuning, and interpretation.
*   **Advanced Feature Engineering:** Implemented log transformations for skewed numericals, target encoding for categoricals, and VIF-driven composite features (e.g., `volume`) to mitigate multicollinearity.
*   **Diverse Model Exploration:** Compared performance across Linear Regression (baseline, VIF-adjusted, Polynomial), Lasso, Ridge, ElasticNet, Random Forest, Gradient Boosting, and XGBoost.
*   **Hyperparameter Tuning:** Utilized Optuna for Bayesian Optimization to fine-tune the best-performing model (XGBoost) for optimal real-world RMSE.
*   **In-depth Model Interpretation:** Performed detailed residual analysis on both log-transformed and original scales, and leveraged SHAP values for nuanced feature importance, especially in clarifying multicollinear impacts.

## Dataset

The project utilizes the [Kaggle Diamonds Dataset](https://www.kaggle.com/datasets/nancyalaswad90/diamonds-prices).
It contains 53,940 observations and 10 features, with `price` as the target variable.
Key features include: `carat`, `cut`, `color`, `clarity`, `depth`, `table`, `length`, `width`, `height`.

## Methodology

### 1. Data Cleaning & Exploratory Data Analysis (EDA)

*   Identified and removed 20 rows with invalid zero dimensions (`x, y, z`) and dropped the `Unnamed: 0` index column.
*   Renamed `x, y, z` to `length, width, height` for clarity.
*   Converted `cut, color, clarity` to ordered categorical types.
*   Conducted univariate analysis (skewness, kurtosis) revealing highly skewed distributions for most features, necessitating transformations.
*   Analyzed bivariate relationships (price vs. categorical, price vs. carat) revealing complex interactions and counter-intuitive trends (e.g., higher quality cuts/colors sometimes correlate with lower *absolute* prices due to smaller carat sizes).
*   Used ANOVA and Cramér's V to assess categorical feature significance and association strength, confirming all three (`cut, color, clarity`) significantly impact price with negligible-to-weak inter-associations.
*   Outlier analysis revealed significant outlier rates for `price` (6.55%), `depth` (4.72%), and `carat` (3.49%).

### 2. Feature Engineering & Preprocessing

*   **Log Transformations:** Applied `np.log1p` to highly right-skewed numerical features and the `price` target to normalize distributions and stabilize variance.
*   **Target Encoding:** Employed for categorical features (`cut, color, clarity`) to capture non-linear relationships with the target variable (on a log scale) and reduce dimensionality.
*   **Standardization:** Applied `StandardScaler` to all numerical features to ensure equal contribution to distance-based models.
*   **VIF-driven Feature Creation:** Identified extreme multicollinearity among `length, width, height` (VIF > 190). Engineered a composite `volume` feature (`height * width * length`) to consolidate their impact, successfully reducing VIF for all remaining features to acceptable levels. `depth` was removed due to its negligible predictive value.

### 3. Model Building & Evaluation

*   **Linear Models:** Evaluated Baseline Linear Regression, VIF-selected Linear Regression, Lasso, Ridge, ElasticNet, and Polynomial Regression (with interactions). These models struggled with interpretability (e.g., counter-intuitive `carat` coefficients) and capturing complex non-linear patterns.
*   **Ensemble Models:** Built Random Forest, Gradient Boosting, and XGBoost models. Random Forest initially showed severe overfitting. Gradient Boosting and XGBoost demonstrated significantly superior performance, highlighting their ability to capture non-linearities and handle feature interactions.

### 4. Model Optimization

*   **Bayesian Optimization with Optuna:** Tuned XGBoost's hyperparameters (`n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_alpha`, `reg_lambda`), optimizing for Root Mean Squared Error (RMSE) on the *original dollar scale*. This significantly improved predictive accuracy over the non-optimized version.

### 5. Model Interpretation

*   **Residual Analysis:** Conducted detailed analysis on both log-transformed (model's training domain, exhibiting homoscedastic residuals) and original dollar scales. This revealed heteroscedasticity on the original scale (prediction errors grow with price), explaining the difference between RMSE and MedAE, and confirmed the model's unbiased nature.
*   **SHAP Values:** Leveraged SHAP (SHapley Additive exPlanations) to gain granular, local and global feature importance. This was critical in:
    *   Correctly attributing `carat` as the **dominant price driver**, overcoming multicollinearity's masking effect seen in simpler models.
    *   Explaining the nuances of `color` and `clarity` impacts, showing how lower quality grades in these attributes can correlate with higher prices due to their strong interaction with `carat` size.
    *   Confirming the negligible individual contribution of `length`, `width`, and `height` once `carat` is accounted for.

## Key Findings & Insights

*   **Carat Dominance:** `carat` weight is the most influential factor, fundamentally driving diamond value, a fact clarified by robust SHAP analysis that cuts through multicollinearity.
*   **Complex Interactions:** Non-linear relationships and intricate feature interactions are paramount in diamond pricing, making tree-based ensemble models (especially XGBoost) far superior to linear approaches.
*   **Quality vs. Size Trade-offs:** Counter-intuitive trends for quality attributes (e.g., D-color diamonds sometimes having lower average absolute prices than J-color) are explained by their inverse correlation with `carat` weight.
*   **Robustness of XGBoost:** The optimized XGBoost model is highly accurate, unbiased, and provides excellent practical precision for a wide range of diamond prices, delivering valuable insights for fair pricing.

## Model Performance

The **Optimized XGBoost Regressor** achieved the best overall performance:

| Model                               | RMSE(train) | RMSE(test) | RMSE (gap_pct) | RMSE(test) / Mean Price (pct) | MedAE(test) | R²(train)(log) | R²(test)(log) |
| :---------------------------------- | :---------- | :--------- | :------------- | :---------------------------- | :---------- | :------------- | :------------ |
| **XGBoost Regressor (Optimized)**   | **428.95**  | **526.71** | **22.79**      | **13.37**                     | **93.00**   | **1.00**       | **0.99**      |

*   **Lowest Test RMSE:** $526.71 (approx. 13.37% of the mean price).
*   **Lowest MedAE:** $93.00, meaning half of predictions are within $93 of actual price.
*   **High R² (log-transformed):** 1.00 (train) and 0.99 (test), indicating excellent variance explanation.

## Getting Started

To explore the project, clone this repository and open the Jupyter Notebook. It is recommended to use a virtual environment.

```bash
# Clone the repository
git clone https://github.com/mcadriaans/ml_diamond-price-prediction.git

# Navigate into the project directory
cd ml_diamond-price-prediction

# Install required Python packages
pip install -r requirements.txt

# Launch the notebook interface
jupyter notebook
