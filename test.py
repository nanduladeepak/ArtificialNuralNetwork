import pandas as pd
from save_model import saveModelNpy , LoadModelNpy
from pandas_profiling import ProfileReport


def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

df = pd.read_csv("ce889_dataCollection.csv", header=None)
df.rename({0:'input1',1:'input2',2:'output1',3:'output2'}, axis=1, inplace=True)
print(df)
# Generate a report
df = remove_outlier(df,'output1')
print(df)
profile = ProfileReport(df)
profile.to_file(output_file="test.pdf")



# df = pd.read_csv("ce889_dataCollection.csv", header=None)
# # print(df)
# X = df.iloc[:, 0:2].copy()
# X_normalized=(X-X.min())/(X.max()-X.min())
# y = df.iloc[:, 2:4].copy()
# y_normalized=(y-y.min())/(y.max()-y.min())

# # print(X_normalized)
# # print(y_normalized)

# X_ZNorm = (X-X.mean())/X.std()
# y_ZNorm = (y-y.mean())/y.std()

# print(X_ZNorm)
# print(y_ZNorm)