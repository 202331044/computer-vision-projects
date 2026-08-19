import matplotlib.pyplot as plt    
from pathlib import Path

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