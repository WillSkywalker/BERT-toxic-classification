import scipy
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.datasets import make_classification

# Reading the dataset
data = pd.read_csv("./data/toxic_comments/train.csv")
print (np.array(data.values).shape)
x_train = data.iloc[:,0]
y_train = data.iloc[:,2:7]
vectorizer = CountVectorizer()
vectors = vectorizer.fit_transform(x_train)



# Under sampling
rus = RandomUnderSampler(return_indices=True)
X_rus, y_rus, id_rus = rus.fit_sample(vectors, y_train.values)
#
print('Removed indexes:', id_rus)
print (X_rus.shape)


# # Over sampling
# smote = SMOTE(ratio='minority')
# X_sm, y_sm = smote.fit_sample(vectors, y_train.values)
# print (X_sm.shape)


# over sampling followed by under sampling
smt = SMOTETomek(ratio='auto')
X_smt, y_smt = smt.fit_sample(vectors, y_train.values)
print (X_smt.shape)