import pandas as pd
import importlib
from typing import Any, List, Dict, Optional





class ModelEvaluator:
    def __init__(self, metrics_config: List[Any], logger):
        
        self.metrics_config = metrics_config
        self.logger = logger

    
    def evaluate_model(self, y_test: pd.Series, y_pred: pd.Series, y_prob: Optional[pd.Series] = None) -> Dict[str, Any]:
        
        self.logger.info("The model evaluation process is being initiated...")
        results = {}

        for metric_info in self.metrics_config:
            module_name = metric_info.module
            metric_name = metric_info.name


            try:
                module = importlib.import_module(module_name)
                metric_func = getattr(module, metric_name)

                proba_metrics = ["roc_auc_score", "log_loss", "brier_score_loss", "average_precision_score"]


            

                if metric_name in proba_metrics:
                    if y_prob is not None:
                        score = metric_func(y_test, y_prob)
                        results[metric_name] = score
                    else:
                        self.logger.warning(
                            f"{metric_name} requires a probability (y_prob). It was skipped because the model doesn't support it."
                        )
                        results[metric_name] = None

                else:

                    try:
                        score = metric_func(y_test, y_pred, zero_division=0)
                    except TypeError:

                        score = metric_func(y_test, y_pred)

                    if metric_name == "classification_report":
                        
                        score = metric_func(y_test, y_pred, output_dict=True)
                        results[metric_name] = score

                    elif metric_name == "confusion_matrix":

                        results[metric_name] = score.tolist() if hasattr(score, 'tolist') else score
                    else:
                        try:
                            results[metric_name] = float(score) if score is not None else None
                        except (TypeError, ValueError):
                            results[metric_name] = score


            except (ImportError, AttributeError) as e:
                self.logger.error(f"Metric library or function not found ({module_name}.{metric_name}): {e}")
            except Exception as e:
                self.logger.error(f"An unexpected error occurred while calculating {metric_name}: {e}")

        self._log_metrics(results)
        return results



    def _log_metrics(self, metrics: Dict[str, float]) -> None:

        self.logger.info("--- MODEL EVALUATION RESULTS ---")

        for metric_name, value in metrics.items():
            if value is not None:
                formatted_name = metric_name.replace('_', ' ').title()
                if isinstance(value, (int, float)):
                    self.logger.info(f"{formatted_name}: {value:.4f}")
                else:
                    self.logger.info(f"{formatted_name}: {value}")

        self.logger.info("-------------------------------------------")