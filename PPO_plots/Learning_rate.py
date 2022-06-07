import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
from scipy.signal import savgol_filter
from load import load_data

#def load_data(filename, factor,nrows=None, y = 'Reward'):
 #   df = pd.read_csv(filename,nrows=nrows)
 #   #df['Reward'] = df['Reward'].rolling(64).mean()
  #  df[y]=savgol_filter(df[y], 101, 3)
   # df['LvFactor'] = np.full(len(df),factor)
    #return df

def make_plot( data, ax, x = 'Episodes', y = 'Reward', hue = 'LvFactor' ):
        formatter1 = EngFormatter(places=0, sep="\N{THIN SPACE}")
        ax.xaxis.set_major_formatter(formatter1)
        #palette=sns.cubehelix_palette( start=1, rot= 0, light=.8,reverse=True, as_cmap=True)
        palette=sns.color_palette("vlag", as_cmap=True)
        #palette=sns.cubehelix_palette(start=.5, rot=0, dark=0.2, light=.7, reverse=True, as_cmap=True)
        #sns.lineplot(data = data, x = x, y = y, ax = ax, hue = hue)
        sns.lineplot(data = data, x = x, y = y, ax = ax, hue = hue, palette=palette)

if __name__ == '__main__':
    fname = 'Learning rate'
    nrows = 615
    smooth = 0.05
    df3 = load_data('data/May28_08-22-19_score.csv',factor=0.0003,fname=fname,smooth =smooth)
    df2 = load_data('data/May29_18-34-43_score.csv',factor=0.0001,fname=fname,smooth =smooth)
    df1 = load_data('data/May27_10-23-46_score.csv',factor=0.00003,fname=fname,smooth =smooth)
    dfa = pd.concat([df3,df2,df1],ignore_index=True)
    #df2 = pd.read_csv('data/runs_May27_10-23-46_gold.csv')
    #df2['Avg_gold']=df2['Gold'].rolling(32).mean()
    afig = plt.figure(figsize = (6,3))
    ax1 = afig.add_subplot(1,2,1) 
    ax2 = afig.add_subplot(1,2,2)
    make_plot(dfa, ax1, y = 'Reward', hue = fname)
    y ='Villages taken'
    df3 = load_data('data/May28_08-22-19_villages.csv',factor=0.0003,y=y,fname=fname,smooth =smooth)
    df2 = load_data('data/May29_18-34-43_villages.csv',factor=0.0001,y=y,fname=fname,smooth =smooth)
    df1 = load_data('data/May27_10-23-46_villages.csv',factor=0.00003,y=y,fname=fname,smooth =smooth)
    dfa = pd.concat([df3,df2,df1],ignore_index=True)
    make_plot(dfa, ax2, y = y,hue = fname)
        #sns.lineplot(x = df2.Episodes, y = df2.Gold, ax = ax2 )
    #palette = sns.color_palette("seagreen", 2)
    #sns.lineplot(data = pd.melt(df2, id_vars=['Episodes'], value_vars=['Gold','Avg_gold'],value_name = 'Gold'),
    #                                x='Episodes', y='Gold', hue='variable', palette=palette,legend = False)   
    afig.tight_layout()
    #sns.pairplot(df, hue="species")
    plt.show(block = True )
    afig.savefig('data/Learning_rate.svg',format='svg')
   