import numpy as np
import pandas as pd

class MNISTBinary():
    def __init__(self, path):
        '''
            path: path to the .csv file 
        '''
        self.df = pd.read_csv(path)
        self.data = np.array(
            self.df.iloc[:, 2:]
        )
        self.labels = np.array(
            self.df.iloc[:, 1].to_numpy()
        )
    
    def getDFData(self):
        return self.df

    def getData(self):
        return self.data
    
    def getLabels(self):
        return self.labels

