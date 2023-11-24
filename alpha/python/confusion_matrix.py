from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import click
import os

@click.command()
@click.argument('current_datetime', type=str, required=True)

#Create function for confusion matrix generation to simplify code

def confusionmatrix(current_datetime):
    datetime_str = current_datetime

    LOADFROM = f"../training/{datetime_str}/eval/"
    MATRIX_PATH = f"../training/{datetime_str}/confusion_matrices/"

    all_labels = np.load(LOADFROM + "labels.npy")
    all_preds = np.load(LOADFROM + "preds.npy")
    labels = [1, 2, 3, 4, 5, 6]
    
    if (not os.path.exists(MATRIX_PATH)):
        os.makedirs(MATRIX_PATH)

    plt.figure()
    cm1 = confusion_matrix(all_labels, all_preds, labels=labels)
    
    # Normalize by row
    plt.imshow(cm1, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    thresh = cm1.max() / 2
    for i, j in np.ndindex(cm1.shape):
        plt.text(j, i, f'{cm1[i, j]:}',
                 horizontalalignment="center",
                 color="white" if cm1[i, j] > thresh else "black")
    
    filename_1 = "confusion_matrix.png"
    plt.savefig(MATRIX_PATH + filename_1, bbox_inches='tight')
    
    cm2 = confusion_matrix(all_labels, all_preds, labels=labels)
    
    # Normalize by rows
    plt.figure()
    cm2_normalized = cm2.astype('float') / cm2.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(cm2_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Normalized Confusion Matrix")
    plt.colorbar()
    
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    thresh = cm2_normalized.max() / 2
    for i, j in np.ndindex(cm2_normalized.shape):
        plt.text(j, i, f'{cm2_normalized[i, j]:.2f}',
                 horizontalalignment="center",
                 color="white" if cm2_normalized[i, j] > thresh else "black")
    
    filename_2 = "row_normalized_confusion_matrix.png"
    plt.savefig(MATRIX_PATH + filename_2, bbox_inches='tight')

    plt.figure()
    cm3 = confusion_matrix(all_labels, all_preds, labels=labels)
    
    # Normalize by columns
    column_sums = cm3.sum(axis=0)
    # Avoid division by zero
    column_sums[column_sums == 0] = 1
    cm3_normalized = cm3.astype('float') / column_sums[np.newaxis, :]
    
    plt.imshow(cm3_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Column-Normalized Confusion Matrix")
    plt.colorbar()
    
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    thresh = cm3_normalized.max() / 2
    for i, j in np.ndindex(cm3_normalized.shape):
        plt.text(j, i, f'{cm3_normalized[i, j]:.2f}',
                 horizontalalignment="center",
                 color="white" if cm3_normalized[i, j] > thresh else "black")
    
    filename_3 = "col_normalized_confusion_matrix.png"
    plt.savefig(MATRIX_PATH + filename_3, bbox_inches='tight')

if __name__ == '__main__':
    confusionmatrix()