from sklearn.metrics import mean_absolute_error,accuracy_score,confusion_matrix
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

data = pd.read_csv("./results/results.csv")
y_true = data.iloc[:,1:4]
y_pred = data.iloc[:,5:8]
print (data.iloc[0,5:8])
# print (y_true)
# print (mean_absolute_error(y_true.values.astype(float),y_pred.values.astype(float)))
# print (np.array(data.values).shape[0])
mae = []

def merger(y):
    print(y,''.join(map(str, map(np.round, y))) )
    return ''.join(map(str, map(int, map(np.round, y))))


# print (merger(np.round(y_pred.values.astype(np.double))))
c_mat = confusion_matrix(list(map(merger, y_true.values)), list(map(merger, y_pred.values)),labels=['000','001','010','011','100','101','110','111'])
# print(np.round(y_true.values.astype(np.double)))
print (c_mat)
# df_cm = pd.DataFrame(c_mat, index = ['Toxic','Obscene','Insult'],
#                   columns = ['Toxic_p','Obscene_p','Insult_p'])
sn.set()
# c_mat = c_mat.pivot('000','001','010','011','100','101','110','111')
c_mat = pd.DataFrame(c_mat,columns=['000','001','010','011','100','101','110','111'])
sn.heatmap(c_mat, annot=True, vmax=np.mean(c_mat.values),vmin=-5,xticklabels=['000','001','010','011','100','101','110','111'])
plt.show()
accuracy = np.array([])
for i in range(np.array(data.values).shape[0]):
    y_t = data.iloc[i,1:4]
    y_p = data.iloc[i,5:8]
    accuracy =np.append(accuracy,accuracy_score(np.round(y_t.values.astype(np.double)),np.round(y_p.values.astype(np.double))))
    mae = np.append(mae,mean_absolute_error(y_t.values.astype(float),y_p.values.astype(float)))


mean_mae = np.mean(mae)
print (accuracy)
mean_accuracy = np.mean(accuracy)
print ("mean MAE:::",mean_mae)
print ("mean accuracy::",mean_accuracy)