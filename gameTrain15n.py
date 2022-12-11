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
# print(df.info())
# # df = df.drop(df[df[0]<0 and df[3]<0].index, inplace = True)
# # df = df.loc((df[0]<0) & (df[3]<0))
# df.drop(df[(df[0] <0 & df[3]>0 )].index, inplace = True)
# print(df)
# print(df.info())
# df_new = df
df_new = df
df_new = df.drop(df[(df[0]<0) & (df[3]>0)].index)
df_new = df.drop(df[(df[0]>0) & (df[3]<0)].index)
df_new = df_new.drop(df_new[(df_new[2]==0) & (df_new[3]==0)].index)
df.drop_duplicates()
df_new = remove_outlier(df_new,2)
df = remove_outlier(df_new,0)
df_new = remove_outlier(df_new,1)
df_new = remove_outlier(df_new,3)
# df_new = df.drop(df[(df[0]>0)].index)
print(df_new)
df_new.to_csv('clearedData.csv',sep=',', index=False)
df_new[3],df_new[2]=df_new[2],df_new[3]
df_new = df_new.reindex(np.random.permutation(df_new.index))
# df_new.dropna(inplace= True)
X_raw = df_new.iloc[:, 0:2].copy()
X_normalized=(X_raw-X_raw.min())/(X_raw.max()-X_raw.min())
# X_normalized=np.log(X_raw.to_numpy())
print(X_normalized)
y = df_new.iloc[:, 2:4].copy()
y_normalized=(y-y.min())/(y.max()-y.min())
print(y_normalized)

# print(X_normalized)
# print(y)

ann = ANN(2,15, 1,2)


X_np = X_normalized.to_numpy()
y_np = y_normalized.to_numpy()
# y_np[0],y_np[1]=y_np[1],y_np[0]

savedModel = LoadModelNpy('savedModels/landerBot15n','3')
# print(savedModel)

ann.setupAnn(savedModel.tolist())

# ann.setupAnn()

barSize = len(y_np)

bar = progressbar.ProgressBar(maxval=50, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
j = 0
rsmeList = []
    # sleep(0.1)
bar.start()
for i in range(0,50):
    bar.update(j+1)
    for X,Y in zip(X_np,y_np):
        ann.trainAnn(X, Y)
    j+=1
bar.finish()
    

for X,Y in zip(X_np,y_np):
    rsmeList.append(ann.getRsme(X,Y))

# print(ann.getPredOutput(X_np[1],True))
model = ann.getModel()
# print(model)
saveModelNpy(model,'savedModels/landerBot15n','3')

plt.plot(rsmeList)
plt.show()