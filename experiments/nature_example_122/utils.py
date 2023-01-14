import analysis_utils as au
import torch
import seaborn as sns
import matplotlib
import torch.nn as nn


class Layer1(nn.Module):

    def __init__(
        self,
        w_1, w_2,
    ):

        super().__init__()

        self.w_1 = nn.Parameter(torch.tensor([[w_1]]))
        self.w_2 = nn.Parameter(torch.tensor([[w_2]]))

    def forward(
        self,
        x
    ):
        return torch.cat([torch.matmul(x, self.w_1), torch.matmul(x, self.w_2)], dim=1)


class Layer2(nn.Module):

    def __init__(
        self,
        w_1, w_2,
        is_trainable=False,
    ):

        super().__init__()

        self.w_1 = nn.Parameter(torch.tensor(
            [[w_1]])) if is_trainable else torch.tensor([[w_1]])
        self.w_2 = nn.Parameter(torch.tensor(
            [[w_2]])) if is_trainable else torch.tensor([[w_2]])

    def forward(
        self,
        x
    ):
        return torch.cat([torch.matmul(x[:, 0:1], self.w_1), torch.matmul(x[:, 1:2], self.w_2)], dim=1)


plot_kwargs = {
    'facet_kws': {
        'legend_out': True,
    },
}


def lim(g, id):
    if id == 'w':
        [ax.set_ylim(-0.1, 1.1) for ax in g.axes.flat]
        [ax.set_xlim(-0.1, 2.1) for ax in g.axes.flat]
        [ax.set_aspect(1.0) for ax in g.axes.flat]
    else:
        [ax.set_ylim(-0.1, 2.1) for ax in g.axes.flat]
        [ax.set_xlim(-0.1, 2.1) for ax in g.axes.flat]
        [ax.set_aspect(1.0) for ax in g.axes.flat]


def plot_levelmap(df, id='w'):

    g = sns.relplot(
        data=df,
        x=f'{id}_1',
        y=f'{id}_2',
        kind='line',
        palette=sns.color_palette("crest", as_cmap=True),
        hue='train__loss',
        style='sign',
        # hue_norm=matplotlib.colors.LogNorm(),
        **plot_kwargs,
    )

    lim(g, id)


def plot_traj(df, id='w', is_style_rule=True):

    # input(df[['num_iterations',f'{id}_1', f'{id}_2']])
    df = au.nature_pre(df)

    g = sns.relplot(
        data=df,
        x=f'{id}_1',
        y=f'{id}_2',
        kind='line',
        hue='Rule',
        style='Rule' if is_style_rule else None,
        markers=True,
        # size='pc_learning_rate',
        # size_norm=matplotlib.colors.LogNorm(),
        **plot_kwargs,
    )

    lim(g, id)
