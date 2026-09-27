# DSN Bootcamp Machine Learning — Sales Prediction

A machine learning regression project developed as part of the **DSN Bootcamp** practice/hackathon workflow. The goal is to predict `total_sales` for unseen product–store observations using product, pricing, visibility, and store-level information.

## Project Objective

Given historical retail records, build a regression model that predicts:

```text
total_sales
```

The notebook currently explores three modelling approaches:

1. Decision Tree Regressor
2. XGBoost Regressor
3. Voting Regressor combining Decision Tree and XGBoost

Model performance is evaluated using **Root Mean Squared Error (RMSE)**, where a lower value is better.

## Dataset Overview

The training dataset contains **6,818 rows and 13 columns**.

| Feature | Description / Role |
|---|---|
| `id` | Unique row identifier |
| `product_code` | Product identifier |
| `product_weight_kg` | Product weight in kilograms |
| `fat_content` | Product fat-content category |
| `shelf_visibility` | Product shelf-visibility measure |
| `product_category` | Product category |
| `product_price` | Product price |
| `store_code` | Store identifier |
| `store_age_years` | Store age |
| `store_size` | Store-size category |
| `store_location_tier` | Store location tier |
| `store_format` | Store format/type |
| `total_sales` | Regression target |

### Missing values observed in the training data

- `product_weight_kg`: 1,225 missing values
- `store_size`: 1,919 missing values

The current modelling pipeline handles missing numeric values with an imputer and missing categorical values using the most frequent category.

## Modelling Workflow

### 1. Train/validation split

The notebook uses:

```python
train_test_split(
    X,
    y,
    test_size=0.10,
    random_state=21
)
```

### 2. Preprocessing

A `ColumnTransformer` is used to preprocess different feature types.

**Numeric features**

- `product_weight_kg`
- `shelf_visibility`
- `product_price`
- `store_age_years`

Missing numeric values are imputed using either the **mean or median**, selected during grid search.

**Categorical features**

- `fat_content`
- `store_code`
- `store_size`
- `store_location_tier`
- `store_format`

Categorical preprocessing uses:

- most-frequent-value imputation
- one-hot encoding

### 3. Feature selection

`SelectFromModel` with a `DecisionTreeRegressor` is used to retain up to 10 selected features.

### 4. Hyperparameter tuning

`GridSearchCV` with **10-fold cross-validation** is used for model selection.

## Current Results

Results recorded in the uploaded notebook:

| Model | Validation RMSE |
|---|---:|
| Decision Tree Regressor | **1127.894** |
| XGBoost Regressor | **1146.769** |
| Voting Regressor | **1123.471** |

Among the models currently tested in the notebook, the **Voting Regressor produced the lowest holdout RMSE: 1123.471**.

> These are local validation results from the notebook and are not necessarily the same as Kaggle leaderboard scores.

## Submission Files

The notebook generates the following files:

```text
dtree.csv
xgb.csv
dtree_xgb.csv
```

Each submission should contain exactly:

```text
id,total_sales
```

Example:

```csv
id,total_sales
row_00001,2150.43
row_00002,1847.20
```

## Repository Structure

A recommended repository layout is:

```text
.
├── DSN_BOOTCAMP_CODE.ipynb
├── README.md
├── PROJECT_NOTES.md
├── SUBMISSION_GUIDE.md
├── REPO_STRUCTURE.md
├── requirements.txt
├── .gitignore
└── data/
    └── README.md
```

The competition datasets should generally **not** be committed if redistribution is restricted. Add them locally or through the competition platform.

## Installation

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

Core dependencies include:

- NumPy
- pandas
- scikit-learn
- XGBoost
- Jupyter

## Running the Notebook

1. Place the competition files in the expected data location, or update the file paths.
2. Open `DSN_BOOTCAMP_CODE.ipynb`.
3. Run the notebook from top to bottom.
4. Review validation RMSE.
5. Generate a submission CSV.
6. Validate its columns and row IDs before uploading it to the competition.

## Important Code Issues to Fix

The uploaded notebook currently contains a few issues worth correcting before treating it as the final project version.

### 1. Initial file assignment is reversed

The notebook currently contains:

```python
train = pd.read_csv('/content/sample_submission.csv')
test = pd.read_csv('/content/test.csv')
sample_submission = pd.read_csv('/content/train.csv')
```

The correct logical assignment is:

```python
train = pd.read_csv('/content/train.csv')
test = pd.read_csv('/content/test.csv')
sample_submission = pd.read_csv('/content/sample_submission.csv')
```

### 2. Submission IDs depend on a separate `test` variable

The modelling data is loaded into `test_dsn_mart`, but submission code uses:

```python
test['id']
```

A safer approach is:

```python
solution_df = pd.DataFrame({
    'id': test_dsn_mart['id'],
    'total_sales': solution
})
```

or to copy the official sample submission and replace only its target column.

### 3. Product information is currently dropped

The notebook drops:

```python
product_code
product_category
```

before modelling.

These variables may contain predictive information. Future experiments should test proper categorical handling rather than automatically discarding them.

### 4. Category consistency should be reviewed

The raw data contains examples of inconsistent capitalization in `product_category`, such as:

```text
Frozen Foods
HEALTH AND HYGIENE
soft drinks
meat
```

If `product_category` is later used as a feature, these labels should be standardized before encoding.

## Possible Improvements

Future iterations can explore:

- stronger categorical preprocessing
- CatBoost for native categorical handling
- LightGBM / additional gradient-boosting models
- cross-validated out-of-fold evaluation
- feature engineering
- product/store interaction features
- target-derived business features where leakage is avoided
- model blending with optimized weights
- systematic comparison of preprocessing choices
- reproducible training scripts outside the notebook

## Metric

The project uses **Root Mean Squared Error (RMSE)**:

```text
RMSE = sqrt(mean((actual - predicted)^2))
```

A lower RMSE indicates better predictive performance.

## Status

**Current stage:** baseline modelling and ensemble experimentation.

The notebook already produces valid model predictions, but preprocessing, feature engineering, validation design, and submission generation can still be improved.

---

Built as part of a DSN Bootcamp machine-learning learning/practice workflow.
