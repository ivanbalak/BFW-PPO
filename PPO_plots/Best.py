"""

Example Plot

"""


import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
from load import load_data

if __name__ == '__main__':
    
    df1 = pd.read_csv('data/runs_May27_10-23-46_score.csv')
    df2 = pd.read_csv('data/runs_May27_10-23-46_gold.csv')
    df2['Avg_gold']=df2['Gold'].ewm(alpha = 0.04).mean()
    df2['Gold']=df2['Gold'].ewm(alpha = 0.65).mean()
    afig = plt.figure(figsize = (6,3))
    ax1 = afig.add_subplot(1,2,1)
    ax2 = afig.add_subplot(1,2,2)
    formatter1 = EngFormatter(places=0, sep="\N{THIN SPACE}")  # U+2009
    ax1.xaxis.set_major_formatter(formatter1)
    ax2.xaxis.set_major_formatter(formatter1)
    palette = sns.light_palette("seagreen")
    sns.lineplot(x = df1.Episodes, y = df1.Reward, ax = ax1,color='seagreen')
    #sns.lineplot(x = df2.Episodes, y = df2.Gold, ax = ax2 )
    palette = sns.color_palette("YlOrBr",2)
    #palette = sns.color_palette("seagreen", 2)
    sns.lineplot(data = pd.melt(df2, id_vars=['Episodes'], value_vars=['Gold','Avg_gold'],value_name = 'Gold'),
                                    x='Episodes', y='Gold', hue='variable', palette=palette,legend = False)   
    afig.tight_layout()
    #sns.pairplot(df, hue="species")
    plt.show(block = True )
    afig.savefig('data/Best_run.svg',format='svg')
   