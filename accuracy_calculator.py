from sklearn.metrics import mean_absolute_error,accuracy_score
import pandas as pd
import numpy as np

accuracy = []
data = pd.read_csv("./results/results.csv")
y_true = data.iloc[0,1:6]
y_pred = data.iloc[0,7:12]
print (data.iloc[0,7:12])
print (y_true)
print (mean_absolute_error(y_true.values.astype(float),y_pred.values.astype(float)))
print (np.array(data.values).shape[0])
mae = []
for i in range(np.array(data.values).shape[0]):
    y_true = data.iloc[i,1:6]
    y_pred = data.iloc[i,7:12]
    mae = np.append(mae,mean_absolute_error(y_true.values.astype(float),y_pred.values.astype(float)))

mean_mae = np.mean(mae)
print ("mean MAE:::",mean_mae)