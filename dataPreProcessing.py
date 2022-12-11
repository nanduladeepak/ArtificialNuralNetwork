import pandas as pd

# function to remove outliers
def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

df = pd.read_csv("ce889_dataCollection.csv", header=None)
df_new = df

# removing values where x velocity is is in thw wrong direction
df_new = df.drop(df[(df[0]<0) & (df[3]>0)].index)
df_new = df.drop(df[(df[0]>0) & (df[3]<0)].index)

# dropping zero values
df_new = df_new.drop(df_new[(df_new[2]==0) & (df_new[3]==0)].index)
df.drop_duplicates()

# removing outliers from the data
df_new = remove_outlier(df_new,2)
df = remove_outlier(df_new,0)
df_new = remove_outlier(df_new,1)
df_new = remove_outlier(df_new,3)
print(df_new)
df_new.to_csv('clearedData.csv',sep=',', index=False)