import numpy as np
import argparse
import re
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_absolute_error
from sklearn import svm
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from sklearn.datasets import make_classification


#Opening files
file = open("/home/s3843408/Desktop/train_preprocessed.csv")
content = file.readlines()
file.close()
data = [line.split(",") for line in content] 
#Delete the unnecessary columns
for x in data:
    del x[1]
    del x[4]

class_names= ['comment_text','identity_hate','insult','obscene','severe_toxic','threat','toxic','toxicity']

#Create an list with all the comments and with all the classes (labels)
comments = []
classes = []
for x in data[1:]:
    y = x[0]
    z = x[1:]
    comments.append(y)
    classes.append(z)

#Change the label representation to 'glued' numbers (from list of "0.0 0.0 etc." to "00etc."")
new = ""
lab = []
for labels in classes:
    for label in labels:
        label = int(float(label))
        label = str(label)
        new = new + label
    lab.append(new)
    new = ""

#Vectorize the comments                
vectorizer = CountVectorizer()
vectors = vectorizer.fit_transform(comments)

smt = SMOTETomek(ratio='auto')
X_smt, y_smt = smt.fit_sample(vectors, lab)
print (X_smt.shape)

#Define train and test sets (10% test)
x_test = X_smt.shape[1] * 0.1
x_train = X_smt[0:X_smt.shape[1] - x_test,:]
y_test = y_smt.shape[1] * 0.1
y_train = y_smt.shape[0:y_smt.shape[1]-y_test,:]

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


#Train and test classifier
clf = svm.SVC()
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
score = clf.score(x_test, y_test)
score2 = mean_absolute_error(y_test, y_pred)
conmax = plot_confusion_matrix(y_test, y_pred,classes=class_names,
                      title='Confusion matrix')

results = {'Data': [y_test,y_pred]}
    # Save data to the CSV file.
df = pd.DataFrame(results)
df.T.to_csv(os.path.join('/home/s3843408/Desktop' + '/resultslabels.csv'), mode='a', index=False, header=False, sep='\t', encoding='utf-8')

f= open("output.txt","w+")
f.write("\nAccuracy score:")
f.write(score)
f.write("\nMean absolute error:")
f.write(score2)
f.write("\nConfusion matrix:\n")
f.write(conmax)
f.close()

