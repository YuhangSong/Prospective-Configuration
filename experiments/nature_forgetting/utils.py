import analysis_utils as au


def plot_mean(df):

    df = au.nature_pre(df)
    g = au.nature_relplot(
        data=df,
        x='pc_learning_rate',
        y='Mean of test__classification_error',
        hue='Rule', style='Rule',
        col='Dataset',
        row='energy_fn_str',
    ).set(xscale='log')

    au.nature_post(g, is_grid=True)


def plot(df):

    df = au.nature_pre(df)

    groups = ['Dataset', 'Rule', 'energy_fn_str', 'pc_learning_rate']

    df = au.add_metric_per_group(
        df, groups,
        lambda df: (
            'mean per group', df['Mean of test__classification_error'].mean()
        ),
    )

    groups.pop(-1)

    df = au.select_rows_per_group(
        df, groups,
        lambda df: df['mean per group'] == df['mean per group'].min()
    )

    df = au.drop_cols(df, ['mean per group'])

    df = au.extract_plot(df, 'test__classification_error',
                         'training_iteration')

    g = au.nature_relplot_curve(
        data=df,
        x='training_iteration',
        y='test__classification_error',
        hue='log_task_i',
        style='Rule',
        col='Dataset',
        row='energy_fn_str',
    )

    au.nature_post(g, is_grid=True)


def plot_mean_shuffle_task_3(df):

    df = au.nature_pre(df)
    g = au.nature_relplot(
        data=df,
        x='pc_learning_rate',
        y='Mean of test__classification_error',
        hue='Rule', style='Rule',
        col='Dataset',
        row='energy_fn_str',
        sharey=False,
        sharex=False,
    ).set(xscale='log')

    au.nature_post(g, is_grid=True)


def plot_shuffle_task_3(df):

    df = au.nature_pre(df)

    groups = ['Dataset', 'Rule', 'energy_fn_str', 'pc_learning_rate']

    df = au.add_metric_per_group(
        df, groups,
        lambda df: (
            'mean per group', df['Mean of test__classification_error'].mean()
        ),
    )

    groups.pop(-1)

    df = au.select_rows_per_group(
        df, groups,
        lambda df: df['mean per group'] == df['mean per group'].min()
    )

    df = au.drop_cols(df, ['mean per group'])

    df = au.extract_plot(df, 'test__classification_error',
                         'training_iteration')

    g = au.nature_relplot_curve(
        data=df,
        x='training_iteration',
        y='test__classification_error',
        style='log_task_i',
        hue='Rule',
        col='Dataset',
        row='energy_fn_str',
        sharey=False,
        aspect=1.5,
    )

    au.nature_post(g, is_grid=True)


def plot_mean_shuffle_task_5(df):

    df = au.nature_pre(df)
    g = au.nature_relplot(
        data=df,
        x='pc_learning_rate',
        y='Mean of test__classification_error',
        hue='Rule', style='Rule',
        col='Dataset',
        row='energy_fn_str',
        sharey=False,
        sharex=False,
    ).set(xscale='log')

    au.nature_post(g, is_grid=True)

    return df


def plot_shuffle_task_5(df):

    df = au.nature_pre(df)

    groups = ['Dataset', 'Rule', 'energy_fn_str', 'pc_learning_rate']

    df = au.add_metric_per_group(
        df, groups,
        lambda df: (
            'mean per group', df['Mean of test__classification_error'].mean()
        ),
    )

    groups.pop(-1)

    df = au.select_rows_per_group(
        df, groups,
        lambda df: df['mean per group'] == df['mean per group'].min()
    )

    df = au.drop_cols(df, ['mean per group'])

    df = au.extract_plot(df, 'test__classification_error',
                         'training_iteration')

    df = df.loc[df['training_iteration'] < 84]

    g = au.nature_relplot_curve(
        data=df,
        x='training_iteration',
        y='test__classification_error',
        style='log_task_i',
        hue='Rule',
        hue_order=['PC', 'BP'],
        col='Dataset',
        row='energy_fn_str',
        sharey=False,
        aspect=1.5,
    )

    au.nature_post(g, is_grid=True)

    return df


def plot_mean_shuffle_task_5_fr(df):

    df = au.nature_pre(df)
    g = au.nature_relplot(
        data=df,
        x='pc_learning_rate',
        y='Mean of test__classification_error',
        hue='Rule', style='log_id',
        col='Dataset',
        row='energy_fn_str',
        sharey=False,
        sharex=False,
    ).set(xscale='log')

    au.nature_post(g, is_grid=True)
