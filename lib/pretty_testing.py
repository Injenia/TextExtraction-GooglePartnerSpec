# This Python file uses the following encoding: utf-8
from __future__ import division
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels]+[5]) # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print "    " + empty_cell,
    for label in labels: 
        print "%{0}s".format(columnwidth) % label,
    print
    # Print rows
    for i, label1 in enumerate(labels):
        print "    %{0}s".format(columnwidth) % label1,
        for j in range(len(labels)): 
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print cell,
        print
        
def predict_test(model, X_test, y_test, labels, verbose=0):
    y_pred = model.predict_classes(X_test, verbose=verbose)
    report = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print()
    print('Test classification report')
    print('Accuracy: %f' % accuracy)
    print(report)
    print('Test confusion Matrix')
    cm = confusion_matrix(y_test, y_pred)
    print_cm(cm, labels)
    return accuracy_score(y_test, y_pred)

def items_count(l):
    d = dict()
    for i in l:
        if i in d.keys():
            d[i] += 1
        else:
            d[i] = 1
    return d

def class_weights_max(labels):
    counts = items_count(labels)
    max_count = max(counts.values())
    weights = dict()
    for key,value in counts.items():
        weights[key] = max_count / counts[key]
    return weights