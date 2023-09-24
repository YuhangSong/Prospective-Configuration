import analysis_utils as au
import torch
import matplotlib

acf2gain = {
    'nn.Sigmoid()': 1,
    'nn.Tanh()': 1.6666666666666667,
    'nn.ReLU()': 1.4142135623730951,
    'nn.LeakyReLU()': 0.9999000099990001,
    'pc.NoneModule()': 1.0,
}


def plot_mean(df, id='test'):

    df = au.nature_pre(df)

    # input(df.loc[df['pc_learning_rate'] == 0.005][f'Mean of {id}__classification_error'])

    g = au.nature_relplot(
        data=df,
        x='pc_learning_rate',
        y=f'Mean of {id}__classification_error',
        hue='Rule', style='Rule',
        col='num_layers',
        row='acf',
    ).set(xscale='log')

    au.nature_post(g, is_grid=True)

    return df


def plot_mean_select_lr(df, config_columns, id='test'):

    df = au.nature_pre(df)

    df = au.summarize_select(
        df, config_columns,
        ['seed'], lambda df: df[f'Mean of {id}__classification_error'].mean(),
        ['pc_learning_rate'], lambda df: lambda df: df['summarize'] == df['summarize'].min(),
    )
    df = df.sort_values(['Rule'], ascending=False)

    g = au.nature_relplot(
        data=df,
        x='num_layers',
        y=f'Mean of {id}__classification_error',
        hue='Rule', style='Rule',
        row='acf',
        aspect=0.8,
    )

    au.nature_post(g, is_grid=True)


def plot_curve(df, config_columns, id='test', is_select_lr=True):

    df = au.nature_pre(df)

    if is_select_lr:
        df = au.summarize_select(
            df, config_columns,
            ['seed'], lambda df: df[f'Mean of {id}__classification_error'].mean(
            ),
            ['pc_learning_rate'], lambda df: lambda df: df['summarize'] == df['summarize'].min(),
        )
    df = df.sort_values(['Rule'], ascending=False)

    y_name = {
        'test': 'test__classification_error',
        'train': 'train:error',
    }[id]

    df = au.extract_plot(df, y_name, 'training_iteration')

    if is_select_lr:
        kwargs = {
            'hue': 'Rule',
            'style': 'Rule',
        }
    else:
        kwargs = {
            'hue': 'Rule',
            'style': 'Rule',
            'size': 'pc_learning_rate',
            'size_norm': matplotlib.colors.LogNorm(),
            'legend_out': True,
        }

    g = au.nature_relplot_curve(
        data=df,
        x='training_iteration',
        y=y_name,
        row='acf',
        col='num_layers',
        **kwargs,
    )

    au.nature_post(g, is_grid=True)


def regression_plot_curve(df, id='test'):

    df = au.nature_pre(df)

    y_name = {
        'test': 'test__classification_error',
        'train': 'train:error',
    }[id]

    df = au.extract_plot(df, y_name, 'training_iteration')

    g = au.nature_relplot_curve(
        data=df,
        x='training_iteration',
        y=y_name,
        hue='Rule', style='Rule',
        size='num_layers',
    ).set(xscale='log')

    au.nature_post(g, is_grid=True)
