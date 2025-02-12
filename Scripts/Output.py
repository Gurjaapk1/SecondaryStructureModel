#Output Processing for ML model

#Save results as a CSV file
import csv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

def save_results_to_csv(predictions, output_file, metrics):
    with open(output_file, 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(['Sequence ID', 'True Label', 'Predicted', 'Probability'])
        for seq_id, true_label, prediction, prob in predictions:
            prob_str = ', '.join(map(str, prob))
            writer.writerow([seq_id, true_label, prediction, prob_str])
        writer.writerow([''])
        writer.writerow(['Metrics'])
        for metric, value in metrics.items():
            writer.writerow([metric, value])

def plot_training_metrics(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1,2,1)
    plt.plot(history['train loss'], label='Train Loss')
    if 'val loss' in history:  
        plt.plot(history['val loss'], label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training & Validation Loss")
    plt.subplot(1,2,2)
    plt.plot(history['train accuracy'], label='Train Accuracy')
    if 'val accuracy' in history:  
        plt.plot(history['val accuracy'], label='Validation Accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training & Validation Accuracy")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()



               
