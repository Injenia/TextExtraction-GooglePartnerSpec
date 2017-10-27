# This Python file uses the following encoding: utf-8
from __future__ import division
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from keras.callbacks import EarlyStopping
#import matplotlib.pyplot as plt

def fitValidationSplit(model, X_train, y_train, split=2/7, epochs=1000, patience=10):
    model.fit(X_train, y_train, validation_split=split, verbose=2, epochs=epochs, callbacks=[EarlyStopping(monitor='loss', patience=100)]) #categorical_accuracy

def fitValidate(model, X_train, y_train, X_val, y_val, labels,  model_filename, class_weights, patience=10, resume=False, max_train_accuracy=0.99):
    best_accuracy = -1
    patience_count = 0
    i = 0
    if resume:
        model.load_weights(model_filename)
    while True:
        hist = model.fit(X_train, y_train, verbose=2, epochs=1, class_weight=class_weights)
        loss = hist.history['loss'][0]
        train_accuracy = hist.history['categorical_accuracy'][0]
        y_pred = model.predict_classes(X_val)

        # Compute scores on validation
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted')
        recall = recall_score(y_val, y_pred, average='weighted')
        f1 = f1_score(y_val, y_pred, average='weighted')

        report = classification_report(y_val, y_pred)
        print('\n\nReport at epoch '+str(i))
        print('Loss: ' + str(loss))
        print('Train accuracy: '+ str(train_accuracy))
        print()

        print('Validation accuracy: '+str(accuracy))
        print('Validation precision: '+str(precision))
        print('Validation recall: '+str(recall))
        print('Validation F1 Score: '+str(f1))
        print()
        print(report)

        cm = confusion_matrix(y_val, y_pred)
        print('Confusion matrix')
        print_cm(cm, labels)
        if best_accuracy == -1 or accuracy > best_accuracy:
            best_accuracy = accuracy
            #Save the model with the best accuracy on the validation set
            model.save_weights(model_filename)
            patience_count = 0
        else:
            patience_count += 1
            print('\nPatience count: '+str(patience_count)+'/'+str(patience))
            if patience_count == patience:
                print('Accuracy on validation stopped decreasing') #loss
                break
        i += 1
        if train_accuracy >= max_train_accuracy:
            print('Maximum train accuracy reached...')
            break
    model.load_weights(model_filename)



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
    return y_pred

def roc_curve_plot(fpr, tpr, roc_auc):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    return fig


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