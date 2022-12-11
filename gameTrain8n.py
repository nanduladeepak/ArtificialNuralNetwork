from ann import ANN 
import numpy as np
import pandas as pd
import progressbar
from save_model import saveModelNpy , LoadModelNpy
from time import sleep
import matplotlib.pyplot as plt


def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

df = pd.read_csv("ce889_dataCollection.csv", header=None)
# print(df)
df = remove_outlier(df,0)
df = remove_outlier(df,1)
X_raw = df.iloc[:, 0:2].copy()
X_normalized=(X_raw-X_raw.mean())/X_raw.std()
y = df.iloc[:, 2:4].copy()
y_normalized=(y-y.min())/(y.max()-y.min())

# print(X_normalized)
# print(y)

ann = ANN(2,8, 1,2)


X_np = X_normalized.to_numpy()
y_np = y_normalized.to_numpy()

savedModel = LoadModelNpy('savedModels/landerBot8n','1')
print(savedModel)
ann.setupAnn(savedModel.tolist())

# ann.setupAnn()

barSize = len(y_np)

bar = progressbar.ProgressBar(maxval=barSize, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()
j = 0
rsmeList = []
for X,Y in zip(X_np,y_np):
    bar.update(j+1)
    # sleep(0.1)
    for i in range(0,10):
        ann.trainAnn(X, Y)
    j+=1
    
bar.finish()

for X,Y in zip(X_np,y_np):
    rsmeList.append(ann.getRsme(X,Y))

print(ann.getPredOutput(X_np[1]))
model = ann.getModel()
print(model)
saveModelNpy(model,'savedModels/landerBot8n','1')

plt.plot(rsmeList)
plt.show()