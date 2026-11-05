import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


def confucion_matrix(test_predictions):

    class_names = ['AD', 'CN', 'EMCI', 'LMCI', 'MCI', 'SMC']

    for orientation, result in test_predictions.items():

        # Convert labels to integers
        y_true = np.asarray(result['True patient Diagnosis']).astype(int)
        y_pred = np.asarray(result['Predicted diagnosis']).astype(int)

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=np.arange(6)
        )

        cm_normalized = confusion_matrix(
            y_true,
            y_pred,
            labels=np.arange(6),
            normalize='true'
        )

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[0]
        )

        axes[0].set_title(f'{orientation} - Absolute')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')

        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[1]
        )

        axes[1].set_title(f'{orientation} - Normalized')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')

        plt.tight_layout()
        plt.show()

    pass


def UMAP():
    pass