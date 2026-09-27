# Submission Guide

## Expected submission format

The competition submission should contain exactly two columns:

```text
id
total_sales
```

## Recommended submission code

Use the official sample submission to preserve row order:

```python
predictions = model.predict(test_features)

submission = sample_submission[['id']].copy()
submission['total_sales'] = predictions

assert len(submission) == len(test)
assert submission['id'].equals(test['id'])
assert submission['id'].is_unique
assert submission['total_sales'].notna().all()

submission.to_csv('submission.csv', index=False)
```

## Why this is safer

The uploaded notebook currently uses:

```python
pd.DataFrame({
    'id': test['id'],
    'total_sales': solution
})
```

but the prediction matrix is built from `test_dsn_mart`. Mixing dataframe names makes the code fragile.

## Existing notebook outputs

The notebook generates:

```text
dtree.csv
xgb.csv
dtree_xgb.csv
```

These correspond to:

- Decision Tree
- XGBoost
- Voting ensemble

## Pre-upload checklist

Before submitting:

```python
print(submission.shape)
print(submission.columns.tolist())
print(submission.isna().sum())
print(submission.head())
```

Verify:

- exactly two columns
- correct target name: `total_sales`
- no extra dataframe index column
- no missing predictions
- same number of rows as test
- IDs in the same order as the official test/sample-submission file

Always save with:

```python
index=False
```
