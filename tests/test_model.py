import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, r2_score

from src.common import ROOT
from src.model import regression_metrics, split_data, validate_data


def test_partition_is_disjoint_complete_and_repeatable():
    data = pd.DataFrame(index=range(200))
    train, test = split_data(data)
    assert len(train) == 160 and len(test) == 40
    assert not set(train) & set(test)
    assert set(train) | set(test) == set(range(200))
    assert np.array_equal(test, split_data(data)[1])


def test_metrics_have_sales_units_not_percentage_accuracy():
    results = regression_metrics([1, 2, 3], [2, 2, 2])
    assert results['mae_thousands_of_units'] == pytest.approx(2 / 3)
    assert results['rmse_thousands_of_units'] == pytest.approx(np.sqrt(2 / 3))
    assert results['r2'] == 0


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1])
def test_invalid_measurement_rejected(bad):
    raw = pd.DataFrame({'TV': [bad], 'radio': [1], 'newspaper': [2], 'sales': [3]})
    with pytest.raises(ValueError):
        validate_data(raw)


def test_saved_claims_recompute_from_individual_predictions():
    predictions = pd.read_csv(ROOT / 'reports/test_predictions.csv')
    metrics = json.loads((ROOT / 'reports/metrics.json').read_text())
    split = json.loads((ROOT / 'reports/split.json').read_text())
    assert predictions['row_index'].tolist() == split['test_row_indices']
    assert not set(split['train_row_indices']) & set(split['test_row_indices'])
    assert r2_score(predictions.actual_sales, predictions.predicted_sales) == pytest.approx(metrics['test']['r2'])
    assert mean_absolute_error(predictions.actual_sales, predictions.predicted_sales) == pytest.approx(metrics['test']['mae_thousands_of_units'])
