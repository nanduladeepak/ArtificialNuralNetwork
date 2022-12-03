from ann import ANN 
import numpy as np
import pandas as pd
import progressbar
from save_model import saveModelNpy , LoadModelNpy
from time import sleep

df = pd.read_csv("ce889_dataCollection.csv", header=None)
# print(df)
X_raw = df.iloc[:, 0:2].copy()
X_normalized=(X_raw-X_raw.min())/(X_raw.max()-X_raw.min())
y = df.iloc[:, 2:4].copy()

# print(X_normalized)
# print(y)

ann = ANN(2,4, 1,2)


X_np = X_normalized.to_numpy()
y_np = y.to_numpy()

savedModel = LoadModelNpy('savedModels/landerBot','1')

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
    rsmeList.append([])
    # sleep(0.1)
    for i in range(0,1000):
        rsmeList[j].append(ann.trainAnn(X, Y))
    j+=1
    

bar.finish()

model = ann.getModel()
print(model)
saveModelNpy(model,'savedModels/landerBot','1')