import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter

if __name__ == '__main__':
    
    df1 = pd.read_csv('data/runs_May27_10-23-46_score.csv')
    df2 = pd.read_csv('data/runs_May23_07-46-35_DQN_reward.csv')
    df1.time -= df1.time[0]
    df2.time -= df2.time[0]
    print(df1.time)
    formatter1 = EngFormatter(places=0, sep="\N{THIN SPACE}")  # U+2009
    #ax1.xaxis.set_major_formatter(formatter1)
    #ax2.xaxis.set_major_formatter(formatter1)
    palette = sns.light_palette("seagreen")
    sns.lineplot(data=df1,x = 'time', y = 'Reward', color='seagreen')
    sns.lineplot(data=df2,x = 'time', y = 'Reward', color='red')
    #sns.lineplot(x = df2.Episodes, y = df2.Gold, ax = ax2 )
    palette = sns.color_palette("YlOrBr",2)
    #palette = sns.color_palette("seagreen", 2)
    #sns.lineplot(data = pd.melt(df2, id_vars=['Episodes'], value_vars=['Gold','Avg_gold'],value_name = 'Gold'),
    #                                x='Episodes', y='Gold', hue='variable', palette=palette,legend = False)   
    #afig.tight_layout()
    #sns.pairplot(df, hue="species")
    #plt.xaxis.set_major_formatter(formatter1)
    plt.legend(labels = ['PPO', 'DQN'])
    plt.show(block = True )
    #afig=plt.gcf()
    #afig.savefig('data/Compare_DQN.svg',format='svg')