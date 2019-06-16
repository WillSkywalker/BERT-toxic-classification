from sklearn.metrics import mean_absolute_error,accuracy_score,confusion_matrix
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

data = pd.read_csv("./results/results.csv")
y_true = data.iloc[:,1:4]
y_pred = data.iloc[:,5:8]
print (data.iloc[0,5:8])
mae = []


# Making the confusion matrix
def merger(y):
    print(y,''.join(map(str, map(np.round, y))) )
    return ''.join(map(str, map(int, map(np.round, y))))

lab_order = ['000','001','010','011','100','101','110','111']
c_mat = confusion_matrix(list(map(merger, y_true.values)), list(map(merger, y_pred.values)),labels=lab_order)
print (c_mat)
sn.set()
fig, ax = plt.subplots(figsize=(15,10))
ax = sn.heatmap(c_mat, annot=True, vmax=np.mean(c_mat)+60,vmin=-20,fmt='g')
ax.set_title('BERT confusion matrix', fontsize=20, fontdict={})
plt.yticks(np.arange(8)+0.5,lab_order,va="center",fontsize="20")
plt.xticks(np.arange(8)+0.5,lab_order,va="center",fontsize="20")
plt.ylabel("Gold Standard",fontsize="20")
plt.xlabel("Predictions",fontsize="20")
plt.show()


# Calculating the accuracy and MAE
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