def overall_accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def average_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    return np.nanmean(class_accuracies)

def kappa_coefficient(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred)

def calculate_f1_precision_recall(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    return f1, precision, recall