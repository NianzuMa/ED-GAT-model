from sklearn.metrics import f1_score
from sklearn.metrics import classification_report


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1_multiclass(preds, labels, label_str_list):
    acc = simple_accuracy(preds, labels)
    f1_micro = f1_score(y_true=labels, y_pred=preds, average='micro')
    f1_macro = f1_score(y_true=labels, y_pred=preds, average='macro')
    f1_weighted = f1_score(y_true=labels, y_pred=preds, average='weighted')
    f1_none = f1_score(y_true=labels, y_pred=preds,
                       average=None).tolist()  # If None, the scores for each class are returned
    return {
        "acc": acc,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f_None": f1_none,
        "classification_report": classification_report(labels, preds, digits=4, target_names=label_str_list)
    }
