from ann import ANN 
import numpy as np
import pandas as pd
import progressbar
from save_model import saveModelNpy , LoadModelNpy
import matplotlib.pyplot as plt

# loading preprocessed data
df_new = pd.read_csv("clearedData.csv",sep=',')

# swapping output columns
col_list = list(df_new)
df_new['3'],df_new['2']=df_new['2'],df_new['3']
df_new.columns = col_list

# shuffaling data
df_new = df_new.reindex(np.random.permutation(df_new.index))

# splitting input and output columns
X_raw = df_new.iloc[:, 0:2].copy()
y = df_new.iloc[:, 2:4].copy()

# normalizing data
X_normalized=(X_raw-X_raw.min())/(X_raw.max()-X_raw.min())
print(X_normalized)
y_normalized=(y-y.min())/(y.max()-y.min())
print(y_normalized)


# setting up ann 2 input 1 hidden with 15 nurons and 2 outputs
ann = ANN(2,15, 1,2)

# changing numpy array to list
X_np = X_normalized.to_numpy()
y_np = y_normalized.to_numpy()

# splitting data to training and testing
X_test,X_training = X_np[:80,:], X_np[80:,:]
y_test, y_training = y_np[:80,:], y_np[80:,:]

# loding saved weights
savedModel = LoadModelNpy('savedModels/landerBot15n','3')

# settingup saved weights to the ann
ann.setupAnn(savedModel.tolist())
# ann.setupAnn()

# setting up progressbar
barSize = len(y_np)

bar = progressbar.ProgressBar(maxval=50, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
j = 0

# training ann
bar.start()
for i in range(0,50):
    bar.update(j+1)
    for X,Y in zip(X_training,y_training):
        ann.trainAnn(X, Y)
    j+=1
bar.finish()
    
# calcularing root mean square error for testing data(rmse)
rsmeList = []
for X,Y in zip(X_test,y_test):
    rsmeList.append(ann.getRsme(X,Y))

# saving weights of the model
model = ann.getModel()
saveModelNpy(model,'savedModels/landerBot15n','3')

# plotting rmsr
plt.plot(rsmeList)
plt.show()