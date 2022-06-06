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
        palette=sns.cubehelix_palette( start=.5, rot=-.75, light=.8,reverse=True, as_cmap=True)
        #palette=sns.cubehelix_palette(start=.5, rot=0, dark=0.2, light=.7, reverse=True, as_cmap=True)
        #sns.lineplot(data = data, x = x, y = y, ax = ax, hue = hue)
        sns.lineplot(data = data, x = x, y = y, ax = ax, hue = hue, palette=palette)

if __name__ == '__main__':
    nrows = 615
    smooth = 0.05
    df1 = load_data('data/May26_16-20-42_score.csv',factor=0.75,smooth =smooth, nrows=nrows)
    print(df1)
    df2 = load_data('data/May26_15-59-51_score.csv',factor=0.2,smooth =smooth)
    df3 = load_data('data/May26_14-58-19_score.csv',factor=0.5,smooth =smooth)
    dfa = pd.concat([df3,df2,df1],ignore_index=True)
    print(dfa)
    #df2 = pd.read_csv('data/runs_May27_10-23-46_gold.csv')
    #df2['Avg_gold']=df2['Gold'].rolling(32).mean()
    afig = plt.figure(figsize = (6,3))
    ax1 = afig.add_subplot(1,2,1)
    ax2 = afig.add_subplot(1,2,2)
    make_plot(dfa, ax1, y = 'Reward')
    y ='Unit moves'
    df1 = load_data('data/May26_16-20-42_range.csv',factor=0.75,smooth =smooth,nrows=nrows, y = y)
    print(df1)
    df2 = load_data('data/May26_15-59-51_range.csv',factor=0.25,y = y,smooth =smooth)
    df3 = load_data('data/May26_14-58-19_range.csv',factor=0.5,y = y,smooth =smooth)
    dfa = pd.concat([df3,df2,df1],ignore_index=True)
    make_plot(dfa, ax2, y = y)
        #sns.lineplot(x = df2.Episodes, y = df2.Gold, ax = ax2 )
    #palette = sns.color_palette("seagreen", 2)
    #sns.lineplot(data = pd.melt(df2, id_vars=['Episodes'], value_vars=['Gold','Avg_gold'],value_name = 'Gold'),
    #                                x='Episodes', y='Gold', hue='variable', palette=palette,legend = False)   
    afig.tight_layout()
    #sns.pairplot(df, hue="species")
    plt.show(block = True )
    afig.savefig('data/Lv_compare.svg',format='svg')
   