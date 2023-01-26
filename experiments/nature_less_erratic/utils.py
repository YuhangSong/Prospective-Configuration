import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from analysis_utils import reduce


def plot(df, lr_name='pc_learning_rate'):
    groups = [lr_name, 'seed']

    groups = ['Rule', lr_name]
    df = reduce(df, groups, lambda df: {'final_length mean': df['final_length'].mean(), 'final_length sem': df['final_length'].sem(
    ) * 0.5, 'traj_length mean': df['traj_length'].mean(), 'traj_length sem': df['traj_length'].sem() * 0.5})
    def convert(x): return np.squeeze(x.to_numpy())
    plt.errorbar(convert(df[['final_length mean']]), convert(df[['traj_length mean']]), convert(df[['final_length sem']]), convert(
        df[['traj_length sem']]), 'none', ecolor='gray', elinewidth=1.5, capsize=1.7, capthick=1.5, zorder=1)
    df = df.sort_values(['Rule'], ascending=False)
    ax = sns.scatterplot(data=df, x='final_length mean', y='traj_length mean',
                         hue='Rule', size=lr_name, style='Rule', alpha=0.75)
