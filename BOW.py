import matplotlib
matplotlib.use('Agg')
import numpy as np
import argparse
import re
import pandas as pd
import scipy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
import seaborn as sn
import matplotlib.pyplot as plt

#Opening data file and removing unnecessary columns
data = pd.read_csv("/home/s3843408/Desktop/kindabalanced.csv")
data = data.drop(["severe_toxic","threat","identity_hate","label"],axis=1)

#Vectorize the comments                
vectorizer = CountVectorizer()
vectors = vectorizer.fit_transform(data.iloc[:,2])

#Glue the values of the labels together (to create values like this: '000','100' etc.)
labels = []
for i in data.index:
    x = data.iloc[i]
    labels.append(''.join(map(str, map(int, (x[3], x[4], x[5])))))

#Define train and test sets (10% test)
x_train, x_test, y_train, y_test = train_test_split(vectors, labels, test_size = 0.1, random_state = 0)

#Transform the y_test (test labels) values to a dimension of 8 (all possible combinations of the three labels),
#so it matches the dimension of the predicted probability labels  
mlb = MultiLabelBinarizer()
myarray = np.asarray(y_test)
y_test_new = mlb.fit_transform(myarray.reshape(-1,1))

#Train classifier
clf = svm.SVC(probability=True)
clf.fit(x_train,y_train)

#Predict labels from the test set (also probabilities for labels)
y_pred = clf.predict(x_test)
y_pred_prob = clf.predict_proba(x_test)

#Get the evaluation metrics
accuracy = clf.score(x_test, y_test)
mae = mean_absolute_error(y_test_new, y_pred_prob)

#Save data to the CSV file.
results = {'Data': [y_test,y_pred]}
df = pd.DataFrame(results)
df.T.to_csv('/home/s3843408/Desktop/resultslabels.csv', mode='a', index=False, header=False, sep='\t', encoding='utf-8')

#Plot the confusion matrix
conmax = confusion_matrix(y_test,y_pred,labels=['000','001','010','011','100','101','110','111'])
lab_order = ['000','001','010','011','100','101','110','111']
sn.set()
fig, ax = plt.subplots(figsize=(15,10))
ax = sn.heatmap(conmax, annot=True, vmax=np.mean(conmax)+60,vmin=-20,fmt='g')
ax.set_title('Bag-of-Words confusion matrix', fontsize=20, fontdict={})
plt.yticks(np.arange(8)+0.5,lab_order,va="center",fontsize="20")
plt.xticks(np.arange(8)+0.5,lab_order,va="center",fontsize="20")
plt.ylabel("Gold Standard",fontsize="20")
plt.xlabel("Predictions",fontsize="20")
plt.savefig('/home/s3843408/Desktop/plot.png')

#Print the final metrics
print(classification_report(y_test, y_pred))
print("Mean Absolute Error:",mae)
print("Accuracy Score:",accuracy)
print("Confusion Matrix:\n",conmax)
