import analysis_utils as au
import torch
import seaborn as sns
import matplotlib
import torch.nn as nn

acf2gain = {
    'nn.Sigmoid()': 1,
    'nn.Tanh()': 1.6666666666666667,
    'nn.ReLU()': 1.4142135623730951,
    'nn.LeakyReLU()': 0.9999000099990001,
    'nn.Identity()': 1.0,
    'pc.NoneModule()': 1.0,
}


class ScaleSigmoid(nn.Sigmoid):

    def __init__(self, scale1=1.0, scale2=1.0, offset=0.0):
        super(ScaleSigmoid, self).__init__()

        self.scale1 = scale1
        self.scale2 = scale2
        self.offset = offset

    def forward(self, x):
        return super(ScaleSigmoid, self).forward(x * self.scale1) * self.scale2 + self.offset


def plot(df):

    df = au.nature_pre(df)

    g = au.nature_relplot(
        data=df,
        x='pc_learning_rate',
        y='target_alignment',
        hue='Rule', style='Rule',
    ).set(xscale='log')

    au.nature_post(g, is_grid=True)


def plot_iteration(df, id='target_alignment'):

    df = au.extract_plot(df, f'train:{id}', 'training_iteration')

    df = au.nature_pre(df)

    g = au.nature_relplot_curve(
        data=df,
        x='training_iteration',
        y=f'train:{id}',
        hue='Rule', style='Rule',
        row='pc_learning_rate',
        sharey=False,
        aspect=3,
    )

    au.nature_post(g, is_grid=True)


def plot_acf(df):

    df = au.nature_pre(df)

    g = au.nature_relplot(
        data=df,
        x='num_layers',
        y='prediction_std',
        hue='Gain',
        col='acf',
        sharey=False,
    )

    au.nature_post(g, is_grid=True)


def plot_depth_width(df, id='target_alignment'):

    df = au.nature_pre(df)

    g = au.nature_relplot(
        data=df,
        x='num_layers',
        y=f'{id}',
        hue='Rule', style='Rule',
        col='acf',
        row='gain_lg',
        aspect=0.8,
        sharey=False,
    )

    [ax.set_ylim(0, 1) for ax in g.axes.flat]

    au.nature_post(g, is_grid=True)


def plot_width_linear(df, id='target_alignment'):

    df = au.nature_pre(df)

    g = au.nature_relplot(
        data=df,
        x='hidden_size',
        y=f'{id}',
        hue='Rule', style='Rule',
        col='acf',
        row='gain_lg',
        aspect=0.8,
        sharey=False,
    )

    [ax.set_xscale('log', basex=2) for ax in g.axes.flat]

    # [ax.set_ylim(0, 1.02) for ax in g.axes.flat]

    au.nature_post(g, is_grid=True)


def plot_depth_width_linear(df, id='target_alignment'):

    df = au.nature_pre(df)

    g = au.nature_relplot(
        data=df,
        x='num_layers',
        y=f'{id}',
        hue='Rule', style='Rule',
        col='acf',
        row='gain_lg',
        aspect=0.8,
        sharey=False,
    )

    [ax.set_ylim(0, 1.02) for ax in g.axes.flat]

    au.nature_post(g, is_grid=True)


def plot_acf_init(df):

    df = au.nature_pre(df)

    g = au.nature_relplot(
        data=df,
        x='pc_learning_rate',
        y='target_alignment',
        hue='Rule', style='Rule',
        col='Gain',
        row='acf',
    ).set(xscale='log')

    au.nature_post(g, is_grid=True)


def plot_112(df):

    df = au.nature_pre(df)

    df = au.extract_plot(df, 'prediction', 'training_iteration')

    df = df.loc[df['training_iteration'] < 24]

    df = au.new_col(df, 'x_0', lambda row: eval(row['prediction'])[0])
    df = au.new_col(df, 'x_1', lambda row: eval(row['prediction'])[1])

    # version
    g = sns.relplot(
        data=df,
        x='x_0',
        y='x_1',
        style='Rule',
        hue='training_iteration',
        palette='rocket',
        aspect=0.94,
        facet_kws={
            'legend_out': True,
        },
    )
    [ax.set_xlim(0) for ax in g.axes.flat]

    # # version
    # g = sns.relplot(
    #     data=df,
    #     x='x_0',
    #     y='x_1',
    #     style='Rule',
    #     hue='training_iteration',
    #     palette='rocket',
    #     aspect=1.0,
    #     facet_kws={
    #         # 'legend_out': True,
    #         'legend_out': False,
    #     },
    # )
    # [ax.set_ylim(0, 1) for ax in g.axes.flat]
    # [ax.set_xlim(0) for ax in g.axes.flat]
    # g._legend.remove()

    au.nature_post(g, is_grid=True)


def plot_112_lr(df):

    df = au.nature_pre(df)

    df = au.extract_plot(df, 'prediction', 'training_iteration')

    df = df.loc[df['pc_learning_rate'] < 0.5]
    # df = df.loc[df['training_iteration'] < 48]

    df = au.new_col(df, 'x_0', lambda row: eval(row['prediction'])[0])
    df = au.new_col(df, 'x_1', lambda row: eval(row['prediction'])[1])

    g = sns.relplot(
        data=df,
        # kind='line',
        x='x_0',
        y='x_1',
        style='Rule',
        # hue='training_iteration',
        # palette=sns.dark_palette("#69d", reverse=True, as_cmap=True),
        aspect=0.94,
        facet_kws={
            'legend_out': True,
        },
        hue='pc_learning_rate',
        hue_norm=matplotlib.colors.LogNorm(),
        edgecolor="none",
        # linewidth=0.1,
        s=20,
        # alpha=0.75,
    )

    au.nature_post(g, is_grid=True)


def plot_112_lr_first(df):

    df = au.nature_pre(df)

    df = au.extract_plot(df, 'prediction', 'training_iteration')

    df = df.loc[df['training_iteration'] < 3]

    df = au.new_col(df, 'x_0', lambda row: eval(row['prediction'])[0])
    df = au.new_col(df, 'x_1', lambda row: eval(row['prediction'])[1])

    g = sns.relplot(
        data=df,
        x='x_0',
        y='x_1',
        style='Rule',
        hue='training_iteration',
        # palette='rocket',
        palette='vlag',
        aspect=0.94,
        facet_kws={
            'legend_out': True,
        },
        size='pc_learning_rate',
        size_norm=matplotlib.colors.LogNorm(),
    )

    au.nature_post(g, is_grid=True)


def plot_112_heatmap_traj(df):

    df = au.nature_pre(df)

    df = au.extract_plot(df, 'prediction', 'training_iteration')

    df = df.loc[df['training_iteration'] < 48]

    df = au.new_col(df, 'x_0', lambda row: eval(row['prediction'])[0])
    df = au.new_col(df, 'x_1', lambda row: eval(row['prediction'])[1])

    g = sns.relplot(
        data=df,
        x='x_0',
        y='x_1',
        style='Rule',
        hue='training_iteration',
        palette='rocket',
        aspect=0.94,
        facet_kws={
            'legend_out': True,
        },
        # size='pc_learning_rate',
        # size_norm=matplotlib.colors.LogNorm(),
    )

    au.nature_post(g, is_grid=True)


def plot_112_heatmap(df):

    df = df.drop(df[(df.x_0 == 0) & (df.x_1 == 1)].index)

    g = sns.relplot(
        data=df,
        x='x_0',
        y='x_1',
        style='seed',
        hue='loss',
        hue_norm=matplotlib.colors.LogNorm(),
        palette='Blues',
        aspect=0.94,
        facet_kws={
            'legend_out': True,
        },
        markers={1482555873: "s"},
        edgecolor="none",
    )

    # [ax.set_ylim(0.75, 1.0) for ax in g.axes.flat]
    # [ax.set_xlim(0.0, 1.0) for ax in g.axes.flat]


def plot_112_levelmap(df):

    g = sns.relplot(
        data=df,
        kind='line',
        x='x_0_level',
        y='x_1_level',
        palette='Blues',
        hue='loss_level',
        hue_norm=matplotlib.colors.LogNorm(),
        aspect=0.94,
        facet_kws={
            'legend_out': True,
        },
    )

    [ax.set_ylim(0.75, 1.0) for ax in g.axes.flat]
    [ax.set_xlim(0.0, 1.0) for ax in g.axes.flat]


base_depth_kwargs = {
    'x': 'num_layers',
    'hue': 'Rule', 'style': 'Rule',
    'aspect': 0.8,
    'sharey': True,
    'legend_out': False,
}


def base_depth(df, id='target_alignment'):

    df = au.nature_pre(df)

    g = au.nature_relplot(
        data=df,
        y=f'{id}',
        **base_depth_kwargs,
    )

    [ax.set_ylim(0, 1.02) for ax in g.axes.flat]

    au.nature_post(g, is_grid=True)


def base_depth_acf(df, id='target_alignment'):

    df = au.nature_pre(df)

    g = au.nature_relplot(
        data=df,
        y=f'{id}',
        col='acf',
        **base_depth_kwargs,
    )

    [ax.set_ylim(0, 1.02) for ax in g.axes.flat]

    au.nature_post(g, is_grid=True)


def base_prospective_index(df):

    g = au.nature_relplot(
        data=df,
        y='prospective_index',
        x='prospective_index_l',
        row='pc_learning_rate',
        col='acf',
    )

    # [ax.set_ylim(0, 1.02) for ax in g.axes.flat]

    au.nature_post(g, is_grid=True)


def base_target_alignment(df):

    g = au.nature_relplot(
        data=df,
        y='target_alignment',
        x='prospective_index_l',
        row='pc_learning_rate',
        col='acf',
    )

    # [ax.set_ylim(0, 1.02) for ax in g.axes.flat]

    au.nature_post(g, is_grid=True)


def base_depth_init(df, id='target_alignment'):

    df = au.nature_pre(df)

    g = au.nature_relplot(
        data=df,
        y=f'{id}',
        col='init_fn',
        **base_depth_kwargs,
    )

    [ax.set_ylim(0, 1.02) for ax in g.axes.flat]

    au.nature_post(g, is_grid=True)


def base_depth_orth_init(df, id='target_alignment'):

    df = au.nature_pre(df)

    g = au.nature_relplot(
        data=df,
        y=f'{id}',
        **base_depth_kwargs,
    )

    [ax.set_ylim(0, 1.02) for ax in g.axes.flat]

    au.nature_post(g, is_grid=True)
