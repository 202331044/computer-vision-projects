
import torch
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (accuracy_score,
                            precision_score,
                            recall_score,
                            f1_score,
                            confusion_matrix,
                            roc_curve,
                            roc_auc_score)

def evaluate(model, test_loader, device, criterion):
    model.eval()

    loss = 0
    total_size = 0

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for text, label in test_loader:
            text = text.to(device)
            label = label.to(device)

            batch_size = label.size(0)

            outputs = model(text)

            loss += criterion(outputs, label).item() * batch_size
            total_size += batch_size

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds)
            all_labels.append(label)
            all_probs.append(probs)
    
    all_preds = torch.cat(all_preds).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()
    all_probs = torch.cat(all_probs).cpu().numpy()

    accuracy = accuracy_score(all_labels,
                              all_preds)

    precision = precision_score(all_labels,
                                all_preds,
                                average="macro",
                                zero_division=0)

    recall = recall_score(all_labels,
                          all_preds,
                          average="macro",
                          zero_division=0)

    f1 = f1_score(all_labels,
                  all_preds,
                  average="macro",
                  zero_division=0)

    cm = confusion_matrix(all_labels, all_preds)

    num_classes = all_probs.shape[1]
    y_true = label_binarize(all_labels, classes=list(range(num_classes)))

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], all_probs[:, i])
        roc_auc[i] = roc_auc_score(y_true[:, i], all_probs[:, i])

    macro_roc_auc = roc_auc_score(y_true, all_probs, multi_class="ovr", average="macro")
   
    return {
            "loss": loss / total_size,
            "accuracy" : accuracy,
            "precision" : precision,
            "recall" : recall,
            "f1" :f1,
            "confusion_matrix": cm,
            "fpr": fpr,
            "tpr": tpr,
            "roc_auc": roc_auc,
            "macro_roc_auc": macro_roc_auc        
            }   
