import matplotlib.pyplot as plt    
from pathlib import Path
import numpy as np

def plot_loss(history, save_path):
    
    epoch_range = range(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots()
    ax.plot(epoch_range, history["train_loss"], label="Train Loss")
    ax.plot(epoch_range, history["val_loss"], label="Validation Loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train - Validation Loss")

    plt.legend()
    plt.savefig(save_path)
    plt.show()
    plt.close()

def plot_acc(history, save_path):

    epoch_range = range(1, len(history["train_acc"]) + 1)

    fig, ax = plt.subplots()
    ax.plot(epoch_range, history["train_acc"], label="Train Accuracy")
    ax.plot(epoch_range, history["val_acc"], label="Validation Accuracy")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Train - Validation Accuracy")
    ax.legend()

    plt.savefig(save_path)
    plt.show()
    plt.close()

def plot_roc_auc_curve(fpr, tpr, class_names, roc_auc, save_path):
    
    
    for i in range(len(class_names)):
        fig, ax = plt.subplots()
        ax.plot(fpr[i], 
                tpr[i], 
                label=f"{class_names[i]} (AUC = {roc_auc[i]:.4f})")

        ax.set_xlabel("fpr")
        ax.set_ylabel("tpr")
        ax.set_title(f"ROC Curve - {class_names[i]}")
        ax.legend()
        
        plt.savefig(save_path/f"ROC_Curve_{class_names[i]}.png")
        plt.show()
        plt.close(fig)
        
def plot_confusion_matrix(confusion_matrix, class_names, save_path):
    cm_normalized = ( confusion_matrix.astype(float)
                    / confusion_matrix.sum(axis=1, keepdims=True))

    fig, ax = plt.subplots()
    im = ax.imshow(cm_normalized)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))

    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Normalized Confusion Matrix")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm_normalized[i, j]:.2%}", ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close(fig)

    