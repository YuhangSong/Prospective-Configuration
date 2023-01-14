import numpy as np
import seaborn as sns
import torch
import torch.nn as nn

import utils as u
import analysis_utils as au
import fit_data as fd


class Perception(nn.Module):

    def __init__(
        self,
        w_L=0.0, w_N=0.0, hidden_size=2, is_fc=False,
    ):

        super().__init__()

        self.is_fc = is_fc

        self.w_L = nn.Parameter(
            torch.zeros(
                1, int(hidden_size/2) if not self.is_fc else hidden_size
            ),
            True,
        )
        self.w_L.data.fill_(w_L)

        self.w_N = nn.Parameter(
            torch.zeros(
                1, int(hidden_size/2) if not self.is_fc else hidden_size
            ),
            True,
        )
        self.w_N.data.fill_(w_N)

    def forward(
        self,
        x
    ):

        if not self.is_fc:
            return torch.cat(
                [
                    torch.matmul(x[:, 0:1], self.w_L),
                    torch.matmul(x[:, 1:2], self.w_N),
                ],
                dim=1,
            )
        else:
            return torch.matmul(x[:, 0:1], self.w_L) + torch.matmul(x[:, 1:2], self.w_N)


class Mix(nn.Module):

    def __init__(
        self,
        w_L=0.0, w_N=0.0, hidden_size=2,
    ):

        super().__init__()

        self.w_L = nn.Parameter(
            torch.zeros(int(hidden_size/2), 1),
            True,
        )
        self.w_L.data.fill_(w_L)

        self.w_N = nn.Parameter(
            torch.zeros(int(hidden_size/2), 1),
            True,
        )
        self.w_N.data.fill_(w_N)

        self.hidden_size = hidden_size

    def forward(
        self,
        x
    ):

        return torch.matmul(x[:, 0:int(self.hidden_size/2)], self.w_L) + torch.matmul(x[:, int(self.hidden_size/2):int(self.hidden_size)], self.w_N)


basic_config_columns = [
    'L',
    'N',
    'L_lr',
    'N_lr',
    'perception_lr',
]

mean_columns = ['seed']

metric_column = 'fear_to_N'

method_column = 'PC'

plot_column = 'group'


def fit_data(df, config_columns):

    df, _ = fd.fit_data(
        df=df,
        config_columns=config_columns,
        mean_columns=mean_columns,
        metric_column=metric_column,
        method_column=method_column,
        plot_column=plot_column,
        raw_data=[
            ['LN+', 47.24872978275533],
            ['LN+', 23.47713468720331],
            ['LN+', 19.880670399836205],
            ['LN+', 13.862975007958582],
            ['LN+', 24.757350992262996],
            ['LN+,L-', 49.84528803043753],
            ['LN+,L-', 51.442164621276824],
            ['LN+,L-', 40.05602559086313],
            ['LN+,L-', 46.52921815201096],
            ['LN+,L-', 27.845251568225706],
            ['N+', 51.77500851953603],
            ['N+', 50.038818231666866],
            ['N+', 35.143724222268034],
            ['N+', 42.84476774272263],
            ['N+', 42.02041322905141],
        ]
    )

    return df


def fit_data_and_plot(df, kind='bar'):

    df = df.groupby(
        ['init_std'],
    ).apply(
        lambda df: fit_data(df, basic_config_columns),
    )

    facet_grid_kwargs = {
        'col': plot_column,
        'sharey': True,
        'aspect': 0.7,
    }
    plot_kwargs = {
        'data': df,
        'x': method_column,
        'y': f'{metric_column}: fitted',
        # 'hue': method_column,
        # 'palette': 'Blues',
        'order': [
            'Data',
            True,
            False,
        ],
    }

    if kind == 'strip':

        g = au.nature_catplot(
            kind='strip',
            linewidth=1,
            alpha=0.5,
            **plot_kwargs,
            **facet_grid_kwargs,
        )

    elif kind == 'bar':

        g = au.nature_catplot(
            kind='bar',
            capsize=0.2,
            errwidth=3,
            **plot_kwargs,
            **facet_grid_kwargs,
        )

    elif kind in ['violin', 'box', 'boxen', 'swarm']:

        g = au.nature_catplot(
            kind=kind,
            **plot_kwargs,
            **facet_grid_kwargs,
        )

    elif kind == 'bar_strip':

        g = sns.FacetGrid(df, **facet_grid_kwargs, margin_titles=True)
        g.map(sns.stripplot, **plot_kwargs)
        g.map(sns.barplot, **plot_kwargs)

    else:

        raise ValueError(f'Unknown kind: {kind}')


def fit_data_and_plot_init_std(df, kind='bar'):

    df = df.groupby(
        ['init_std'],
    ).apply(
        lambda df: fit_data(df, basic_config_columns),
    )

    facet_grid_kwargs = {
        'col': method_column,
        'sharey': True,
        'aspect': 0.7,
    }
    plot_kwargs = {
        'data': df,
        'x': plot_column,
        'y': f'{metric_column}: fitted',
        'hue': 'init_std',
        'palette': 'Blues',
        'order': [
            'N+',
            'LN+',
            'LN+,L-',
        ],
    }

    if kind == 'strip':

        g = au.nature_catplot(
            kind='strip',
            linewidth=1,
            alpha=0.5,
            **plot_kwargs,
            **facet_grid_kwargs,
        )

    elif kind == 'bar':

        g = au.nature_catplot(
            kind='bar',
            capsize=0.2,
            errwidth=3,
            **plot_kwargs,
            **facet_grid_kwargs,
        )

    elif kind in ['violin', 'box', 'boxen', 'swarm']:

        g = au.nature_catplot(
            kind=kind,
            **plot_kwargs,
            **facet_grid_kwargs,
        )

    elif kind == 'bar_strip':

        g = sns.FacetGrid(df, **facet_grid_kwargs, margin_titles=True)
        g.map(sns.stripplot, **plot_kwargs)
        g.map(sns.barplot, **plot_kwargs)

    else:

        raise ValueError(f'Unknown kind: {kind}')

    # [ax.set_ylim(-0.1, 1.0) for ax in g.axes.flat]


def fit_data_and_plot_hidden_size(df, kind='bar', is_has_is_fc=True):

    df = df.groupby(
        ['hidden_size', 'is_fc'] if is_has_is_fc else ['hidden_size'],
    ).apply(
        lambda df: fit_data(df, basic_config_columns),
    )

    facet_grid_kwargs = {
        'row': 'is_fc' if is_has_is_fc else None,
        'col': method_column,
        'sharey': True,
        'aspect': 0.7,
    }
    plot_kwargs = {
        'data': df,
        'x': plot_column,
        'y': f'{metric_column}: fitted',
        'hue': 'hidden_size',
        'palette': 'Greens',
        'order': [
            'N+',
            'LN+',
            'LN+,L-',
        ],
    }

    if kind == 'strip':

        g = au.nature_catplot(
            kind='strip',
            linewidth=1,
            alpha=0.5,
            **plot_kwargs,
            **facet_grid_kwargs,
        )

    elif kind == 'bar':

        g = au.nature_catplot(
            kind='bar',
            capsize=0.2,
            errwidth=3,
            **plot_kwargs,
            **facet_grid_kwargs,
        )

    elif kind == 'bar_strip':

        g = sns.FacetGrid(df, **facet_grid_kwargs, margin_titles=True)
        g.map(sns.stripplot, **plot_kwargs)
        g.map(sns.barplot, **plot_kwargs)

    else:

        raise ValueError(f'Unknown kind: {kind}')

    # [ax.set_ylim(-0.1, 1.0) for ax in g.axes.flat]
