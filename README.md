# 💎Predicting Diamond Prices: A Comprehensive Regression Analysis

## Project Overview

This project aims to develop a machine learning model that predicts diamond prices with both high accuracy and clear interpretability. Using a rich dataset of 53,940 entries, the analysis cleans and explores the data, examining relationships between the target variable `price` and the diamond attribute predictors, and comparing linear and tree-based models.

We do not only focus on predictive performance but also aim to understand model behaviour. We observe how correlated size-related features affect linear model coeffcients, whether feature reduction helps. and how prediction errors behave on both log-transformed and original price scales.

The final model is an <b>Optuna-tuned XGBoost regressor</b>, selected using validation performance and evaluated once on a held-out test set.

## Project Highlights

*   Cleaned the dataset by removing duplicate observations and records with physically invalid zero dimensions.
*   Explored numerical distributions, categorical relationships, outliers, and associations among features.
*   Split the data into <b>64% training</b>, <b>16% validation</b>, and <b>20% test</b> sets.
*   Applied <b>log1p</b> transformations to skewed numerical features and the `price` target.
*   Target encoded the ordinal categorical features `cut`, `color` and `clarity`, fitting the encoder on the training data.
*   Standardized predictor features for modelling.
*   Compared baseline and regularized linear regression , polynomial interaction features, Random Forest, Gradient Boosting and XGBoost Regressor.
*   Investigated multicollinearity with variance inflation factors (VIF) and examined built-in XGBoost feature importance and SHAP explanations.
*   Used Optuna'S Bayesian optimization approach to tune XGBoost hyperparameters against validation RMSE on the original price scale.
*   Evaluated the selected model using RMSE, MAE, median absoulte error (MedAE), log-scale R² and residual analysis.


## Dataset

Data is comprised of  53,940 observations and 10 features, with `price` as the target variable.
Key features include: `carat`, `cut`, `color`, `clarity`, `depth`, `table`, `x`, `y`, `z`. 
x, y, z we renamed as `length`, `width`, `height` for improved readability.

### Data Cleaning 
* Dropped the `Unnamed: 0` row-identifier column before checking for duplicates.
* Removed <b>148 exact duplicate rows</b>, checking the diamond features rather than the unique row identifier.
* Removed <b>3 records with physically invalid zero dimensions</b>.
* Set the categorical quality features in their intended ordinal order.

## Methodology
### 1. Exploratory Data Analysis 
The EDA examines the distributions of numerical features, including skewness and kurtosis; price relationships with categorical and numerical features; outliers; and correlations among predictors.
ANOVA and Cramér's V are used to investigate categorical feature relationships.

The analysis shows that diamond `price` is strongly related to `carat` weight and other size-related measurements. Its also highlights wha average prieces across quality grades need careful interpretation. Diamonds in different grades can have different carat weight distributions.

### 2. Preprocessing and data split
The dataset is split into training(64%), validation(16%) and test(20%) subsets. The validation set is used to compare models and tune hyperparameters. The test set is held back for final evaluation.

Preprocessing includes:
* Applying `np.lop1p`to skewed numerical features and to `price`.
* Target encoding categorical features and fitting the encoder to the training data.
* Applying `StandardScaler` to predictor features.

Models are trained to predict `log1p(price)`. For metrics in dollars, predictions are transformed back to the original price scale.

### 3. Model Comparision
The notebook compares several approaches:
* Baseline Linear Regression
* VIF-informed linear model investigation
* Lasso, Ridge and Elastic Net Regression
* Linear Regression with pairwise polynomial interaction features
* Random Forest Regressor
* Gradient Boosting Regressor
* XGBoost Regressor

The linear model experiments reveal strong correlations among size-related features can make individual(independent) coefficients difficult to interpret. Adding interaction terms improves on the baseline linear model, while tree-based models better capture non-linear relationships and feature interactions.

### 4. Feature importance  and feature reduction experiments
The notebook examines XGBoost's built-in feature importance and SHAP values. These are used to explore how the model relies on the predictors and how feature values contribute to predictions.

Several reduced feature experiments test whether a smaller model can retain performance:
* Using only the five highest ranked builtin features substantially worsens validation performance.
* Removing `length`, `width` and `height` produces a small improvement over the original XGBoost model, but does not outperform the tuned model.
* Removing `cut` and `table` degrades validation performance.

These experiments support retaining all eight predictors in the final model. Feature importance and SHAP describe model behaviour; they do not establish that a feature has an independent causal effect.

### 5. XGBoost optimization and residual analysis
Optuna is used to tune XGBoost hyperparameters, with validation RMSE on the original dollar scale as the optimization objection objective. The tuned model improves validation RMSE from <b>$552.97</b> for the original XGBoost model to <b>$532.46</b>, <b>an improvement of approximately 3.7%</b>.

Residuals are examined on both the log-transformed and original price scales. On the original scale, absolute errors tend to increase for more expensive diamonds, meaning predictions are less precise in dollar terms at the upper end of the price range.

## Final Model Performance

The <b>Optimized XGBoost Regressor</b> using all eight predictors, was selected based on validation performance and evaluated on the held-out test set.

| Metric                             | Train| Validation | Test| 
| :---------------------------------- | :---------- | :--------- | :------------- | 
| <b>RMSE</b>  | $413.29  | $532.46 | $522.62 | 
|<b>RMSE (% of mean price)</b>| 10.53% | 13.35% | 13.39% |
|<b>MAE</b>|	$212.60 | $264.45 | $262.04 |
|<b>MedAE</b>| $80.45 | $94.45 |	$95.31 |
|<b>R² (log-transformed target)</b>|	0.995 | 0.993 | 0.992 |

### Interpreting the test results
* RMSE gives greater weight to larger errors, so it is sensitive to occassional large misses.
* The mean absolute test error is about $262 on the held-out test observations.
* Half of the absolute test errors are no greater than approximately $95.31
* The model explains a high proportion of variation in the transformed target on this split.

Test performance is close to validation performance, which is encouraging for this data split. It is not a guarantee that the same performance will hold on other samples or real-world diamond listings.

## Key Findings

1. <font color= #0000FF><b>Size-related features are strong predictors</b></font>. Carat weight and physical dimensions rank highly in the EDA and XGBoost interpretation.
2. <font color= #0000FF><b>Multicollinearity complicates linear model interpretation</b></font>. Correlated size-related predictors make it difficult to isolate their individual coefficient effects.
3. <font color= #0000FF><b>Non-linear models perform better in this project</b></font>. The interaction model improves on baseline linear regression and tree-based models further reduce prediction error.
4. <font color= #0000FF><b>Tuning improves XGBoost's validation performance</b></font>. Optuna reduces validation RMSE by approximately 3.7% compared to the original XGBoost model.
5. <font color= #0000FF><b>Feature reduction has trade-offs</b></font>. The tested smaller feature sets do not outperform the tuned model suing all eight features.
6. <font color= #0000FF><b>Prediction errors are uneven across the price range</b></font>. Residual analysis indicates larger absolute errors for more expensive diamonds.

## Limiations and Next Steps
* Model comparisons use a single train-validation-test split. Cross-validation could help assess how stable the results are across different samples.
* Errors tend to increase in dollar terms for more expensive diamonds.
* Correlated predictors make individual feature contributions difficult to isolate. SHAP and built-in importance should be interpreted as explanations of model behaviour, not causal evidence.
* Alternative encodings for `cut`, `color` and `clarity` could be compared.
* Additional justified features such as `length-to-width` ratio, could be investigated.
* Preprocessing and prediction could be packaged into a reusable pipeline for consistent future predictions.

## Repository Contents
* `Diamond_Price_Prediction.ipynb` - the notebook containing data cleaning, EDA, preprocessing, model comparison, feature interpretation, tuning and final evaluation.
* `requirements.txt` - Python package dependencies, if included in the repository.

## Getting started
Clone the repository and open the notebook in a Python environment with Jupyter installed:

```bash
# Clone the repository
git clone https://github.com/mcadriaans/ml_diamond-price-prediction.git

# Navigate into the project directory
cd ml_diamond-price-prediction

# Launch the notebook interface
pip install requirements.txt
jupyter notebook
```
