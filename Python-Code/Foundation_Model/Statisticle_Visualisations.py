import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import numpy as np


def confucion_Matrix(test_predictions, level, model):
    name = ''
    model_ = ''
    class_names = [
        'CN', 'SMC', 'EMCI', 'MCI', 'LMCI', 'AD'
    ]

    lables = np.arange(6)

    if level == 'Sclice':
        name = 'Slice Level'
    elif level == 'Patient':
        name = 'Patient Level'
    else:
        name = 'Independend'  
        
    if model =='RF':
        model_ = 'Random Forest'
    elif model == 'SVC':
        model = 'Suport Vector'
    elif model ==' LR':
        model_ = 'Linear Regression'
    elif model == 'LDA':
        model_ ='LDA'
        
    if name == 'Independend':
        y_true = test_predictions['y_true']
        y_pred = test_predictions['y_pred']
        cm = metrics.confusion_matrix(
                y_true, y_pred, labels=lables
            )

        cm_normalize = metrics.confusion_matrix(
                y_true, y_pred, labels=lables, normalize='true'
            )

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax1
            )

        ax1.set_title(f'Confusion Matrix : {name} ({model_})')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')

        sns.heatmap(
                cm_normalize,
                annot=True,
                fmt='.2f',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax2
            )

        ax2.set_title(f'Normalized Confusion Matrix: {name} ({model_})')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')

        plt.tight_layout()
        plt.savefig(f'{name}_{model_}.png',dpi=300,bbox_inches = 'tight')
        plt.show()
        plt.close(f)
        
        pass        
    else: 
        for orientation, data in test_predictions.items():

            y_true = data['y_true']
            y_pred = data['y_pred']

            cm = metrics.confusion_matrix(
                y_true, y_pred, labels=lables
            )

            cm_normalize = metrics.confusion_matrix(
                y_true, y_pred, labels=lables, normalize='true'
            )

            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax1
            )

            ax1.set_title(f'Confusion Matrix: {orientation} : {name} ({model_})')
            ax1.set_xlabel('Predicted')
            ax1.set_ylabel('Actual')

            sns.heatmap(
                cm_normalize,
                annot=True,
                fmt='.2f',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax2
            )

            ax2.set_title(f'Normalized Confusion Matrix: {orientation} : {name} ({model_})')
            ax2.set_xlabel('Predicted')
            ax2.set_ylabel('Actual')

            plt.tight_layout()
            plt.savefig(f'{orientation}_{name}_{model_}.png',dpi=300,bbox_inches = 'tight')
            plt.show()
            plt.close(f)
        
        pass
def hold_out_matrix():
    
    pass