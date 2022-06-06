import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
from load import load_data

if __name__ == '__main__':
    
    df1 = pd.read_csv('data/runs_May27_10-23-46_score.csv')
    df2 = pd.read_csv('data/runs_May23_07-46-35_DQN_reward.csv')
    df3 = load_data('data/runs_Bfw-v0__1654022076_score.csv',smooth = 0.05)
    df4 = load_data('data/runs_Bfw-v0__1654346159_score.csv',smooth = 0.05)
    df5 = load_data('data/runs_Bfw-v0__LSTM_1654392102_score.csv',smooth = 0.05)
    df1.time -= df1.time[0]
    df2.time -= df2.time[0]
    df1.Step = df1.Episodes*100
    df2['Episodes'] = df2.Step // 100
    df3['Episodes'] = df3.Step // 100
    df4['Episodes'] = df4.Step // 100
    df5['Episodes'] = df5.Step // 100

    formatter1 = EngFormatter(places=0, sep="\N{THIN SPACE}")  # U+2009
    #ax1.xaxis.set_major_formatter(formatter1)
    #ax2.xaxis.set_major_formatter(formatter1)
    palette = sns.light_palette("seagreen")
    sns.lineplot(data=df1,x = 'Episodes', y = 'Reward', color='magenta')
    sns.lineplot(data=df2,x = 'Episodes', y = 'Reward', color='red')
    #sns.lineplot(data=df3,x = 'Episodes', y = 'Reward', color='blue')
    #sns.lineplot(data=df4,x = 'Episodes', y = 'Reward', color='orange')
    sns.lineplot(data=df5,x = 'Episodes', y = 'Reward', color='seagreen')
    #sns.lineplot(x = df2.Episodes, y = df2.Gold, ax = ax2 )
    palette = sns.color_palette("YlOrBr",2)
    #palette = sns.color_palette("seagreen", 2)
    #sns.lineplot(data = pd.melt(df2, id_vars=['Episodes'], value_vars=['Gold','Avg_gold'],value_name = 'Gold'),
    #                                x='Episodes', y='Gold', hue='variable', palette=palette,legend = False)   
    #afig.tight_layout()
    #sns.pairplot(df, hue="species")
    #plt.xaxis.set_major_formatter(formatter1)
    plt.legend(labels = ['PPO', 'Double DQN','PPO LSTM'])
    plt.show(block = True )
    #afig=plt.gcf()
    #afig.savefig('data/Compare_DQN.svg',format='svg')