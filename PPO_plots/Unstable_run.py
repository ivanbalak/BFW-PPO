import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
from scipy.signal import savgol_filter
from load import load_data

if __name__ == '__main__':
    nrows = 615
    afig = plt.figure(figsize = (6,3))
    ax1 = afig.add_subplot(1,2,1)
    ax2 = afig.add_subplot(1,2,2)
    y ='Reward'
    smooth = .1
    df1 = load_data('data/runs_May19_23-37-21_score.csv',smooth = smooth)
    df2 = load_data('data/runs_May20_11-41-40_score.csv',smooth = 0.02)
    sns.lineplot(data = df1, x = 'Episodes',y =y, ax = ax1,color='steelblue')
    sns.lineplot(data = df2, x = 'Episodes',y =y, ax = ax1,color='orangered')
    smooth = .1
    y ='Invalid moves'
    df1 = load_data('data/runs_May19_23-37-21_moves.csv',y =y, smooth = smooth)
    df2 = load_data('data/runs_May20_11-41-40_moves.csv',y =y,smooth = smooth)
    sns.lineplot(data = df1, x = 'Episodes',y = y, ax = ax2,color='steelblue')
    sns.lineplot(data = df2,  x = 'Episodes',y = y, ax = ax2,color='orangered')

    afig.tight_layout()
    plt.show(block = True )
    afig.savefig('data/Unstable.svg',format='svg')
   