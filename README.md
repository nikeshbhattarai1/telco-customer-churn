# Telco Customer Churn: Linear Models End to End

This project uses linear models on the Telco Customer Churn dataset. The notebook covers customer churn classification, customer tenure prediction and Customer Lifetime Value (CLV) estimation. It also includes validation and data leakage checks to make sure the final results are reliable.

## What's in the notebook

The notebook is split into six parts.

1. **Getting a first model running.** EDA, cleaning the TotalCharges column, encoding the target, one hot encoding the categorical features, splitting into train and validation sets, scaling and building a dummy baseline. The baseline gets 73% accuracy just by always predicting "no churn" which sets the floor any real model needs to beat.
2. **Compare models and evaluate honestly.** Logistic Regression, Ridge Classifier and SGD Classifier are compared on accuracy, precision, recall, F1, ROC-AUC and PR-AUC. It plots ROC and PR curves and tries three ways to pick a decision threshold: a fixed budget approach, an F1 maximizing approach and a cost based approach using assumed costs for false positives and false negatives along with revenue per retained customer. It also looks at the logistic regression coefficients to see which features matter most.
3. **Regression models and Customer Lifetime Value.** Predicts customer tenure using Linear, Ridge, Lasso and Elastic Net regression, checks residuals, plots a Lasso regularization path and builds a simple CLV estimate from predicted tenure times monthly charge.
4. **Making sure results are actually honest.** Runs 5 fold cross validation and compares it against the holdout split, plots learning curves to check for overfitting or underfitting and runs a deliberate data leakage experiment. A fake feature derived directly from the target is added on purpose, the inflated score is measured and then the feature is removed to show why leakage checks matter.
5. **Ship it.** Builds the final scikit-learn Pipeline (scaler plus classifier) and evaluates it on a held out test set. A comparison table lines up cross validation mean, validation score, and test score side by side to check the model generalizes.
6. **Model card.** A short model card documents intended use, performance and limitations of the final logistic regression classifier.

## Key results

| Stage | Metric | Value |
|---|---|---|
| Dummy baseline | Accuracy | 0.730 |
| Logistic Regression (val) | Accuracy / F1 / ROC-AUC | 0.806 / 0.605 / 0.842 |
| Logistic Regression (5 fold CV) | ROC-AUC / F1 | 0.845 ± 0.013 / 0.600 ± 0.028 |
| Best regression model (tenure) | MAE | about 7 months |
| Final deployment threshold | n/a | 0.266, chosen for cost optimization instead of the default 0.5 |

## Data

The dataset is included in this repository under the `data` folder.

```text
data/
└── Telco-Customer-Churn.csv
```

The notebook loads the dataset using:

```python
df = pd.read_csv('data/Telco-Customer-Churn.csv')
```

The dataset was originally obtained from the [Telco Customer Churn dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

The CSV is included in this repository so the notebook can be run directly after cloning the project.


## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

Install with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Running it

```bash
jupyter notebook customer-churn.ipynb
```

Run the cells from top to bottom. Some later sections (regression, the leakage experiment and the final shipping pipeline) reload and re-split the data themselves, so they do not strictly depend on earlier cell state but the notebook is meant to be read in order since each section builds on the reasoning of the one before it.

## Notes

The leakage section in part 4 is intentional and educational where a fake feature derived from the target is added on purpose to show how leakage inflates metrics and then it is removed.

---
