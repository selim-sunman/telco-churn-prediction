import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score


class ModelVisualizer:

    def __init__(self, logger, save_dir: str):
        self.logger = logger
        self.save_dir = Path(save_dir)

        sns.set_theme(style="whitegrid")


    def plot_confusion_matrix(self, y_test, y_pred, class_names=["No Churn", "Churn"], filename="confusion_matrix.png"):

        self.logger.info("A confusion matrix graph is being drawn...")

        try:
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8,6))
            sns.heatmap(cm, annot=True, fmnt="g", cmap="blues", xticklabels=class_names, yticklabels=class_names)
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

        self.logger.info("Plotting Precision-Recall curve and Average Precision score...")

        try:
            precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
            ap_score = average_precision_score(y_test, y_prob)

            df_pr = pd.DataFrame({
                "Recall": recall,
                "Precision": precision
            })

            plt.figure(figsize=(8, 6))
            sns.lineplot(data=df_pr, x="Recall", y="Precision", color="b", linewidht=2, label=f"Model (AP = {ap_score:.3f})")
            plt.fill_between(recall, precision, alpha=0.2, color="b")

            baseline = y_test.sum() / len(y_test)
            plt.axhline(y=baseline, color="r", linestyle="--", label=f"Random Prediction (Baseline = {baseline:.2f})")

            plt.title(f"Telco Churn: Precision-Recall Curve\nAverage Precision Score: {ap_score:.3f}", fontsize=14, fontweight="bold")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.legend(loc="upper right")
            plt.tight_layout

            save_path = self.save_dir / filename
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            plt.close()
            self.logger.info(f"The graph has been saved: {save_path}")
        except Exception as e:
            self.logger.error(f"An error occurred while plotting the Precision-Recall curve and Average Precision score: {e}")



    def plot_feature_importance(self, model, feature_names, top_n, filename):

        try:
            importances = model.feature_importances_

            df_importances = pd.DataFrame({
                "feature": feature_names,
                "Importance_Level": importances
            }).sort_values(by="Importance_Level", ascending=False).head(top_n)

            plt.figure(figsize=(10,6))
            sns.barplot(x="Importance_Level", y="feature", data=df_importances, palette="viridis")
            plt.title(f"Most Effective {top_n} Attribute", fontweight="bold")
            plt.xlabel("Importance_Level")
            plt.ylabel("Features")

            save_path = self.save_dir / filename
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            plt.close()
            self.logger.info(f"The graph has been saved: {save_path}")
        except Exception as e:
            self.logger.error(f"An error occurred while plotting feature importances: {e}")
        




