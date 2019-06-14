import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

#Opening files
file = open("/home/s3612406/BERT-toxic-classification/train_preprocessed.csv")
content = file.readlines()
file.close()
data = [line.split(",") for line in content] 
#Delete the unnecessary columns
for x in data:
    del x[1]
    del x[4]

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

#Define train and test sets (10% test)
x_train = vectors[:143615]
x_test = vectors[143615:]
y_train = lab[:143615]
y_test = lab[143615:]

#Train and test classifier
clf = svm.SVC()
clf.fit(x_train, y_train)
score = clf.score(x_test, y_test)
print ("Score:\n", score)

y_pred = clf.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))