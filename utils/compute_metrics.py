from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import numpy as np

def compute_metrics(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn = []
    fp = []
    fn = []
    tp = []
    
    for i in range(len(labels)):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = np.sum(cm) - (TP + FN + FP)
        tn.append(TN)
        fp.append(FP)
        fn.append(FN)
        tp.append(TP)

    sensitivity = np.mean([tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0 for i in range(len(labels))])
    specificity = np.mean([tn[i] / (tn[i] + fp[i]) if (tn[i] + fp[i]) > 0 else 0 for i in range(len(labels))])
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    
    return acc, f1, sensitivity, specificity