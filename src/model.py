"""Predict market-level sales from advertising spend with held-out evaluation."""
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split

from .common import ROOT, load_source, plot_style, save_run, write_json

FEATURES = ['TV', 'radio', 'newspaper']
SEED = 42


def validate_data(raw):
    required = FEATURES + ['sales']
    if not set(required).issubset(raw.columns):
        raise ValueError(f'Dataset must contain {required}.')
    data = raw[required].apply(pd.to_numeric, errors='raise').copy()
    if not np.isfinite(data.to_numpy()).all() or (data < 0).any().any():
        raise ValueError('Sales and advertising spend must be finite and nonnegative.')
    if data.duplicated().any():
        raise ValueError('Duplicate observations require review before splitting.')
    return data


def split_data(data):
    return train_test_split(data.index.to_numpy(), test_size=.2, random_state=SEED)


def regression_metrics(actual, predicted):
    return {'r2': float(r2_score(actual, predicted)),
            'mae_thousands_of_units': float(mean_absolute_error(actual, predicted)),
            'rmse_thousands_of_units': float(np.sqrt(mean_squared_error(actual, predicted)))}


def run():
    data = validate_data(pd.read_csv(load_source('advertising.csv')))
    train_ids, test_ids = split_data(data)
    x_train, x_test = data.loc[train_ids, FEATURES], data.loc[test_ids, FEATURES]
    y_train, y_test = data.loc[train_ids, 'sales'], data.loc[test_ids, 'sales']
    model = LinearRegression().fit(x_train, y_train)
    baseline = DummyRegressor(strategy='mean').fit(x_train, y_train)
    predicted = model.predict(x_test)
    cv = cross_validate(LinearRegression(), x_train, y_train,
                        cv=KFold(n_splits=5, shuffle=True, random_state=SEED),
                        scoring={'r2': 'r2', 'mae': 'neg_mean_absolute_error', 'mse': 'neg_mean_squared_error'})
    folds = pd.DataFrame({'fold': range(1, 6), 'r2': cv['test_r2'],
                          'mae': -cv['test_mae'], 'rmse': np.sqrt(-cv['test_mse'])})
    metrics = {'observations': len(data), 'training_observations': len(train_ids),
               'test_observations': len(test_ids), 'random_seed': SEED,
               'test': regression_metrics(y_test, predicted),
               'mean_baseline_test': regression_metrics(y_test, baseline.predict(x_test)),
               'training_only_5fold_cv': {f'{column}_{stat}': float(getattr(folds[column], stat)())
                                          for column in ['r2', 'mae', 'rmse'] for stat in ['mean', 'std']},
               'missing_values': int(data.isna().sum().sum()),
               'duplicate_rows': int(data.duplicated().sum()),
               'target_units': 'thousands of units; advertising inputs are thousands of dollars'}
    reports = save_run(metrics, ['pandas', 'numpy', 'matplotlib', 'scikit-learn'])
    folds.to_csv(reports / 'cross_validation.csv', index=False)
    pd.DataFrame({'row_index': test_ids, 'actual_sales': y_test.to_numpy(), 'predicted_sales': predicted,
                  'residual': y_test.to_numpy() - predicted}).to_csv(reports / 'test_predictions.csv', index=False)
    write_json(reports / 'split.json', {'train_row_indices': train_ids.tolist(), 'test_row_indices': test_ids.tolist()})
    write_json(reports / 'model.json', {'model': 'LinearRegression', 'trained_on': 'training split only (160 observations)',
               'features': FEATURES, 'coefficients': model.coef_.tolist(), 'intercept': float(model.intercept_),
               'training_min': x_train.min().to_dict(), 'training_max': x_train.max().to_dict()})
    pd.DataFrame({'feature': FEATURES, 'coefficient': model.coef_}).to_csv(reports / 'coefficients.csv', index=False)
    plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    axes[0].scatter(y_test, predicted, color='#007f86', alpha=.8, edgecolors='white')
    low, high = min(y_test.min(), predicted.min()), max(y_test.max(), predicted.max())
    axes[0].plot([low, high], [low, high], '--', color='#e48b32')
    axes[0].set(title=f'Held-out sales predictions | R² = {metrics["test"]["r2"]:.3f}', xlabel='Actual sales (thousands)', ylabel='Predicted sales (thousands)')
    axes[1].scatter(predicted, y_test - predicted, color='#007f86', alpha=.8, edgecolors='white')
    axes[1].axhline(0, ls='--', color='#e48b32')
    axes[1].set(title='Residuals reveal model limitations', xlabel='Predicted sales (thousands)', ylabel='Actual minus predicted (thousands)')
    fig.savefig(reports / 'model_evaluation.png')
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout='constrained')
    for ax, feature in zip(axes, FEATURES):
        ax.scatter(x_train[feature], y_train, color='#007f86', s=15, alpha=.6)
        ax.set(xlabel=f'{feature} spend ($ thousands)', ylabel='Sales (thousands)', title=feature)
    fig.suptitle('Training data only | association is not causation')
    fig.savefig(reports / 'training_relationships.png')
    plt.close(fig)
    print(json.dumps(metrics, indent=2))
    return metrics


def predict(spend):
    if len(spend) != 3 or not np.isfinite(spend).all() or any(v < 0 for v in spend):
        raise ValueError('Supply three finite, nonnegative advertising amounts.')
    path = ROOT / 'reports/model.json'
    if not path.exists():
        raise FileNotFoundError('Run python -m src.model first.')
    model = json.loads(path.read_text(encoding='utf-8'))
    outside = [f for f, value in zip(FEATURES, spend) if not model['training_min'][f] <= value <= model['training_max'][f]]
    return {'predicted_sales_thousands': float(np.dot(spend, model['coefficients']) + model['intercept']),
            'outside_training_range': outside,
            'note': 'Educational association model; predictions can be negative and are not causal budget advice.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predict', nargs=3, type=float, metavar=('TV', 'RADIO', 'NEWSPAPER'), help='Advertising spend in thousands of dollars')
    args = parser.parse_args()
    if args.predict is None:
        run()
    else:
        print(json.dumps(predict(args.predict), indent=2))
