import numpy as np

def saveModelNpy(model:list,fileName:str,FileVersion:str):
    model_reshaped = np.asarray(model)
    np.save(f'{fileName}_{FileVersion}.npy', model_reshaped)

def LoadModelNpy(fileName:str,FileVersion:str):
    loaded_model = np.load(f'{fileName}_{FileVersion}.npy', allow_pickle=True)
    return loaded_model