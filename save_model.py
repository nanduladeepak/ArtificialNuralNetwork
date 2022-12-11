import numpy as np

# to save weights of the ann
def saveModelNpy(model:list,fileName:str,FileVersion:str):
    model_reshaped = np.asarray(model,dtype=object)
    np.save(f'{fileName}_{FileVersion}.npy', model_reshaped)

# to retreave the saved weights of the ann
def LoadModelNpy(fileName:str,FileVersion:str):
    loaded_model = np.load(f'{fileName}_{FileVersion}.npy', allow_pickle=True)
    return loaded_model