# Project Notes — DSN Bootcamp Sales Prediction

## What the current notebook does

The uploaded notebook contains 13 code cells and implements the following workflow:

1. Imports NumPy and pandas.
2. Loads training and test data.
3. Inspects the training dataframe with `info()`, `head()`, and `describe()`.
4. Splits the training data into features and target.
5. Drops `id`, `product_code`, and `product_category` from the model features.
6. Builds a preprocessing pipeline.
7. Trains a tuned Decision Tree model.
8. Creates a Decision Tree submission.
9. Trains a tuned XGBoost model.
10. Creates an XGBoost submission.
11. Builds a Voting Regressor from the two fitted pipelines.
12. Creates an ensemble submission.

## Dataset observations

Training dataframe:

- Rows: 6,818
- Columns: 13
- Target: `total_sales`

Observed dtypes:

- float64: 4 columns
- int64: 1 column
- object/string: 8 columns

### Missing data

`product_weight_kg`

- 5,593 non-null
- 1,225 missing

`store_size`

- 4,899 non-null
- 1,919 missing

### Numeric summary recorded in the notebook

| Variable | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| `product_weight_kg` | 12.8509 | 12.6470 | 4.482 | 21.883 |
| `shelf_visibility` | 0.0655 | 0.0532 | 0.0000 | 0.3201 |
| `product_price` | 140.4736 | 142.0650 | 31.11 | 273.04 |
| `store_age_years` | 34.1782 | 33 | 23 | 47 |
| `total_sales` | 2174.7566 | 1790.8900 | 32.70 | 12996.82 |

## Current feature set

The following columns are deliberately excluded before modelling:

```python
'id'
'product_code'
'product_category'
'total_sales'
```

The model therefore receives:

### Numeric

```text
product_weight_kg
shelf_visibility
product_price
store_age_years
```

### Categorical

```text
fat_content
store_code
store_size
store_location_tier
store_format
```

## Preprocessing

### Numeric variables

`SimpleImputer` is applied.

The imputation strategy is selected by grid search:

```text
mean
median
```

### Categorical variables

The pipeline uses:

```python
SimpleImputer(strategy='most_frequent')
OneHotEncoder(drop='first', sparse_output=False)
```

### Feature selection

The pipeline applies:

```python
SelectFromModel(
    DecisionTreeRegressor(random_state=22),
    max_features=10,
    threshold=-float('inf')
)
```

This limits the transformed model input to at most 10 selected features.

## Models

### Decision Tree

Grid:

```python
{
    'preprocessor__flint__imputer__strategy': ['mean', 'median'],
    'model__max_depth': [None, 5, 7],
    'model__min_samples_leaf': [10, 20]
}
```

Search:

```text
10-fold GridSearchCV
scoring = negative mean squared error
```

Recorded holdout RMSE:

```text
1127.894
```

### XGBoost

Grid:

```python
{
    'preprocessor__flint__imputer__strategy': ['mean', 'median'],
    'model__n_estimators': [50, 100, 150],
    'model__max_depth': [None, 5, 10]
}
```

Recorded holdout RMSE:

```text
1146.769
```

### Voting Regressor

Models combined:

```text
Decision Tree pipeline
XGBoost pipeline
```

Recorded holdout RMSE:

```text
1123.471
```

This is the lowest RMSE among the three results stored in the notebook.

## Important bugs / presentation issues

### File-loading bug

The first cell currently assigns:

```python
train = sample_submission.csv
test = test.csv
sample_submission = train.csv
```

This should be corrected.

### Two separate train/test naming systems

The notebook uses both:

```text
train / test / sample_submission
```

and:

```text
train_dsn_mart / test_dsn_mart
```

This can easily cause `NameError`s or mismatched IDs. Pick one naming convention and use it throughout.

Recommended:

```python
train
test
sample_submission
```

### Submission IDs

Submission code should use the same test dataframe that produced the prediction matrix.

Prefer:

```python
submission = sample_submission[['id']].copy()
submission['total_sales'] = predictions
```

Then validate:

```python
assert len(submission) == len(test)
assert submission['id'].equals(test['id'])
assert submission['total_sales'].notna().all()
```

### Category normalization

Examples in the raw dataset show inconsistent `product_category` case:

```text
Frozen Foods
HEALTH AND HYGIENE
soft drinks
meat
```

The current model drops this column, but future models that use it should normalize the labels.

## Suggested next experiments

1. Establish a cross-validated baseline without feature selection.
2. Compare inclusion vs exclusion of `product_category`.
3. Compare inclusion vs exclusion of `product_code`.
4. Try CatBoost.
5. Tune XGBoost more carefully.
6. Test interaction/aggregate features.
7. Use K-fold out-of-fold predictions instead of relying only on one 90/10 split.
8. Blend models using validation-driven weights rather than equal VotingRegressor weights.
9. Save preprocessing and model configuration for reproducibility.
10. Keep a results table for every experiment.

## Experiment log template

| Experiment | Features | Model | CV | RMSE | Kaggle Score | Notes |
|---|---|---|---|---:|---:|---|
| Baseline 01 | Current | Decision Tree | GridSearch 10-fold + holdout | 1127.894 | — | Notebook result |
| Baseline 02 | Current | XGBoost | GridSearch 10-fold + holdout | 1146.769 | — | Notebook result |
| Ensemble 01 | Current | Voting DT + XGB | Holdout | 1123.471 | — | Best notebook holdout |
