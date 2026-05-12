"""
visualizer.py - Draws plots to help us understand model performance.

This module draws three types of charts:
- Confusion Matrix: Shows correct and incorrect predictions.
- Precision-Recall Curve: Shows the trade-off between precision and recall.
- Feature Importance: Shows which features helped the model the most.

All charts are saved as PNG images. If a chart fails to draw, it just prints
an error and moves on so the rest of the code doesn't crash.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score


class ModelVisualizer:
    """Draws and saves evaluation charts.

    Attributes:
        logger: Logger used to print progress and error messages.
        save_dir (Path): The folder where the PNG images will be saved.
    """

    def __init__(self, logger, save_dir: str):
        self.logger = logger
        self.save_dir = Path(save_dir)

        sns.set_theme(style="whitegrid")

    def plot_confusion_matrix(self, y_test, y_pred, class_names=["No Churn", "Churn"], filename="confusion_matrix.png"):
        """Draws a confusion matrix and saves it as an image.

        A confusion matrix helps us see how many times the model was right or
        wrong for each category (churn vs. no churn).

        Args:
            y_test (array-like): The real answers.
            y_pred (array-like): The model's predictions.
            class_names (list[str]): Names for the categories on the plot.
            filename (str): Name of the saved image file.
        """

        self.logger.info("A confusion matrix graph is being drawn...")

        try:
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8,6))
            sns.heatmap(cm, annot=True, fmt="g", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
            plt.title("Confusion Matrix")
            plt.xlabel("True Label")
            plt.ylabel("Predicted Labels")
            
            save_path = self.save_dir / filename
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            plt.close()
            self.logger.info(f"The graph has been saved: {save_path}")
        except Exception as e:
            self.logger.error(f"An error occurred while drawing the confusion matrix: {e}")

    def plot_precision_recall_curve(self, y_test, y_prob, filename="precision_recall_curve.png"):
        """Draws a precision-recall curve and saves it as an image.

        This is very useful for imbalanced datasets like churn prediction,
        where one class (no churn) is much bigger than the other (churn).

        Args:
            y_test (array-like): The real answers.
            y_prob (array-like): The model's probability scores.
            filename (str): Name of the saved image file.
        """

        self.logger.info("Plotting Precision-Recall curve and Average Precision score...")

        try:
            precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
            ap_score = average_precision_score(y_test, y_prob)

            df_pr = pd.DataFrame({
                "Recall": recall,
                "Precision": precision
            })

            plt.figure(figsize=(8, 6))
            sns.lineplot(data=df_pr, x="Recall", y="Precision", color="b", linewidth=2, label=f"Model (AP = {ap_score:.3f})")
            plt.fill_between(recall, precision, alpha=0.2, color="b")

            baseline = y_test.sum() / len(y_test)
            plt.axhline(y=baseline, color="r", linestyle="--", label=f"Random Prediction (Baseline = {baseline:.2f})")

            plt.title(f"Telco Churn: Precision-Recall Curve\nAverage Precision Score: {ap_score:.3f}", fontsize=14, fontweight="bold")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.legend(loc="upper right")
            plt.tight_layout()

            save_path = self.save_dir / filename
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            plt.close()
            self.logger.info(f"The graph has been saved: {save_path}")
        except Exception as e:
            self.logger.error(f"An error occurred while plotting the Precision-Recall curve and Average Precision score: {e}")

    def plot_feature_importance(self, model, feature_names, top_n, filename="feature_importance.png"):
        """Draws a bar chart showing the most important features.

        Only works if the model has a `feature_importances_` attribute
        (like Random Forest or XGBoost).

        Args:
            model: The trained machine learning model.
            feature_names (list[str]): Names of the features in the dataset.
            top_n (int): How many top features to show on the chart.
            filename (str): Name of the saved image file.
        """

        try:
            importances = model.feature_importances_

            df_importances = pd.DataFrame({
                "feature": feature_names,
                "Importance_Level": importances
            }).sort_values(by="Importance_Level", ascending=False).head(top_n)

            plt.figure(figsize=(10,6))
            sns.barplot(x="Importance_Level", y="feature", data=df_importances, hue="feature", palette="viridis", legend=False)
            plt.title(f"Most Effective {top_n} Attribute", fontweight="bold")
            plt.xlabel("Importance_Level")
            plt.ylabel("Features")

            save_path = self.save_dir / filename
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            plt.close()
            self.logger.info(f"The graph has been saved: {save_path}")
        except Exception as e:
            self.logger.error(f"An error occurred while plotting feature importances: {e}")
