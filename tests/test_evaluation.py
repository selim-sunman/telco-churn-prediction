import pytest
import pandas as pd
from src.evaluation import ModelEvaluator

class MetricConfig:
    def __init__(self, module: str, name: str):
        self.module = module
        self.name = name

def to_result_dict(results):
    if isinstance(results, list):
        assert len(results) > 0
        results = results[0]
    assert isinstance(results, dict)
    return results


def test_evaluation_with_label_and_proba_metrics(dummy_config, mock_logger):

    y_test = pd.Series([0, 1, 0, 1])
    y_pred = pd.Series([0, 1, 0, 1])
    y_prob = pd.Series([0.1, 0.9, 0.2, 0.8])

    metric_list = dummy_config["metrics"]

    metrics_config = [MetricConfig(module=m["module"], name=m["name"]) for m in metric_list]

    evaluator = ModelEvaluator(metrics_config=metrics_config, logger=mock_logger)

    results = evaluator.evaluate_model(y_test=y_test, y_pred=y_pred, y_prob=y_prob)
    results = to_result_dict(results)

    assert "accuracy_score" in results
    assert "roc_auc_score" in results

    assert isinstance(results["accuracy_score"], float)
    assert pytest.approx(results["roc_auc_score"], rel=1e-6) == 1.0




def test_evaluation_without_proba_sets_none_for_proba_metrics(dummy_config, mock_logger):

    y_test = pd.Series([0, 1, 0, 1])
    y_pred = pd.Series([0, 1, 0, 1])

    metric_list = dummy_config["metrics"]

    metrics_config = [MetricConfig(module=m["module"], name=m["name"]) for m in metric_list]

    evaluator = ModelEvaluator(metrics_config=metrics_config, logger=mock_logger)

    results = evaluator.evaluate_model(y_test=y_test, y_pred=y_pred, y_prob=None)
    results = to_result_dict(results)

    assert "accuracy_score" in results
    assert "roc_auc_score" in results

    assert isinstance(results["accuracy_score"], float)
    assert results["roc_auc_score"] is None



def test_evaluation_handles_missing_metric_function_gracefully(dummy_config, mock_logger):

    y_test = pd.Series([0, 1, 0, 1])
    y_pred = pd.Series([0, 1, 0, 1])


    broken_metrics = dummy_config["metrics"].copy()
    broken_metrics.append({
        "module": "sklearn.metrics",
        "name": "this_metric_does_not_exist"
    })


    metrics_config = [MetricConfig(module=m["module"], name=m["name"]) for m in broken_metrics]

    evaluator = ModelEvaluator(metrics_config=metrics_config, logger=mock_logger)

    results = evaluator.evaluate_model(y_test=y_test, y_pred=y_pred)
    results = to_result_dict(results)


    assert "accuracy_score" in results
    assert isinstance(results["accuracy_score"], float)

    assert "this_metric_does_not_exist" not in results