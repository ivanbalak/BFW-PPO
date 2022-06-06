import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
from scipy.signal import savgol_filter

def load_data(filename, factor = None,fname = 'LvFactor', nrows=None, y = 'Reward',smooth = None):
    df = pd.read_csv(filename,nrows=nrows)
    #df['Reward'] = df['Reward'].rolling(64).mean()
    if smooth != None :
        df[y]=df[y].ewm(alpha = smooth).mean()
        #df[y]=savgol_filter(df[y], smooth, 3)
    if factor != None :
        df[fname] = np.full(len(df),factor)
    return df