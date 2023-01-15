import analysis_utils as au
import numpy as np
import seaborn as sns
import matplotlib

# define data_ from experiment
# cam extract with https://apps.automeris.io/wpd/
data_ = [
    {
        'at': 'at_punish',
        'is_switch': False,
        'value': 0.07656332326255083,
        'PC': 'Data',
    },
    {
        'at': 'next_trial',
        'is_switch': True,
        'value': 0.030083598201063405,
        'PC': 'Data',
    },
    {
        'at': 'at_punish',
        'is_switch': True,
        'value': -0.05977541312190382,
        'PC': 'Data',
    },
    {
        'at': 'next_trial',
        'is_switch': False,
        'value': -0.1794935997796775,
        'PC': 'Data',
    },
]

plot_groups_ = [
    'at',
    'is_switch',
]


def prepare_df(df):
    """Base prepare dataframe.
    """

    # extract log
    df = au.extract_plot(df, 'log', 'training_iteration')
    # create is_switch from log
    df = au.new_col(df, 'is_switch', lambda row: eval(row['log'])[0])
    # value is_switch from log
    df = au.new_col(df, 'value', lambda row: eval(row['log'])[1])
    # remove is_switch==None, which is the trials when the subject is not punished
    df = df[df['is_switch'].notna()]
    # keep only the later trials as they are more stable
    df = df.loc[df['training_iteration'] > 64]
    return df


def plot(df, **kwargs):
    """Base plot function.
    """

    return au.nature_relplot(
        data=df,
        x='at',
        y='value',
        style='is_switch',
        hue='is_switch',
        col='PC',
        palette=np.concatenate(
            (sns.color_palette()[2:], sns.color_palette()[:2])
        ),
        **kwargs,
    )


def reduce_mean(df, plot_groups=plot_groups_):
    """Average value over configs other than plot_groups
        at this stage, only config columns lefted are those related to plots (no config columns related to hparams should be lefted)
    """

    df = au.reduce(
        df, plot_groups,
        lambda df: {
            'value_mean': df['value'].mean(),
        },
    )

    return df


def _get_record(df, dict_):
    """Get individual record, specified by dict_.
    """

    return au.filter_dataframe_by_dict(df, dict_)['value_mean_fit_data'].item()


def _get_records(df, data, plot_groups):
    """Create records, which is a np array summarizing the essential numbers.
    """

    records = np.array([
        _get_record(
            df,
            dict_={
                plot_group: data_item[plot_group] for plot_group in plot_groups
            }
        ) for data_item in data
    ])

    return records


def fit_data(
    df, kw_id,
    data=data_,
    plot_groups=plot_groups_,
    # the initial search range of fit_data_w and fit_data_b can be resolved by
    # maping the maximum and minimum values from the best fit model to data
    # the mapping can be resolved with
    # https://www.wolframalpha.com/input/?i=systems+of+equations+calculator&assumption=%7B%22F%22%2C+%22SolveSystemOf2EquationsCalculator%22%2C+%22equation1%22%7D+-%3E%220.014532383858935233+w+%2B+b+%3D+0.07656332326255083%22&assumption=%22FSelect%22+-%3E+%7B%7B%22SolveSystemOf2EquationsCalculator%22%7D%7D&assumption=%7B%22F%22%2C+%22SolveSystemOf2EquationsCalculator%22%2C+%22equation2%22%7D+-%3E%220.00939979654120041+w+%2B+b+%3D+-0.1794935997796775%22
    fit_data_w_true=np.arange(5.0, 5.2, 0.01).tolist(),
    fit_data_w_false=np.arange(30, 34, 0.1).tolist(),
    fit_data_b_true=np.arange(-0.4, -0.2, 0.01).tolist(),
    fit_data_b_false=np.arange(-0.5, -0.3, 0.01).tolist(),
):
    """Fit the model to data.
    """

    # add fit_data_w
    df = au.explode_with(
        df, 'fit_data_w',
        {
            True: fit_data_w_true,
            False: fit_data_w_false,
        }[kw_id['PC']]
    )

    # add fit_data_b
    df = au.explode_with(
        df, 'fit_data_b',
        {
            True: fit_data_b_true,
            False: fit_data_b_false,
        }[kw_id['PC']]
    )

    # map value_mean to value_mean_fit_data
    df = au.new_col(
        df, 'value_mean_fit_data',
        lambda row: row['value_mean'] * row['fit_data_w'] + row['fit_data_b'],
    )

    # merge some columns so that get a column with the same format of data from the original publication
    df = au.reduce(
        df, ['fit_data_w', 'fit_data_b'],
        lambda df: {
            'records': _get_records(df, data, plot_groups),
        },
    )

    # data (obtained from the original publication)
    records_data = np.array([data_item['value'] for data_item in data])
    # calculate error when fitting the data
    df = au.new_col(
        df, 'fit_error',
        lambda row: np.sum(0.5 * (row['records'] - records_data)**2)
    )

    # sorting according to error when fitting the data
    df = df.sort_values(['fit_error'])

    return df


def apply_fit_data(
    df,
    fit_data_w_true=5.06,
    fit_data_w_false=33.3,
    fit_data_b_true=-0.3,
    fit_data_b_false=-0.41,
):

    # rename the old value column
    df = df.rename(columns={'value': 'value (not fit)'})

    # add the new value column with fitting parameters
    df = au.new_col(
        df, 'value',
        lambda row: row['value (not fit)'] * {
            True: fit_data_w_true,
            False: fit_data_w_false,
        }[row['PC']] + {
            True: fit_data_b_true,
            False: fit_data_b_false,
        }[row['PC']],
    )

    return df


def add_data(df, data=data_):
    """Add each data item to dataframe as a row
    """

    for data_item in data:
        df = df.append(data_item, ignore_index=True)

    return df


"""
    Specified functions to this experiment.
"""

# for investigation of reason


def prepare_df_reason(df, y='weight'):
    """Prepare dataframe.
    """

    # extract log
    df = au.extract_plot(df, y, 'training_iteration')

    return df


def plot_reason(df, **kwargs):
    """Plot function.
    """

    return au.nature_relplot_curve(
        data=df,
        x='training_iteration',
        y='weight',
        size='seed',
        size_norm=(1, 1),
        hue='PC',
        hue_order=[True, False],
        col='which_weight',
        **kwargs,
    )


def plot_reason_seed(df, y):
    """Plot function.
    """

    return au.nature_relplot_curve(
        data=df,
        x='training_iteration',
        y=y,
        row='seed',
        hue='PC',
        hue_order=[True, False],
        col='which_weight',
        sharey=False,
    )


def plot_reason_simple(df, **kwargs):
    """Plot function.
    """

    return au.nature_relplot_curve(
        data=df,
        x='training_iteration',
        y='weight',
        size='pc_learning_rate',
        size_norm=matplotlib.colors.LogNorm(),
        hue='PC',
        hue_order=[True, False],
        col='which_weight',
        **kwargs,
    )


def prepare_df_reason_inference(df):
    """Prepare dataframe.
    """

    # extract log
    df = au.extract_plot(df, 'value', 't')

    return df


def plot_reason_simple_inference(df, **kwargs):
    """Plot function.
    """

    return sns.relplot(
        data=df,
        x='t',
        y='value',
        hue='value',
        hue_norm=(-0.62, 0.62),
        palette='vlag',
        style='at_iteration',
        # size='at_iteration',
        # size_norm=(1, 1),
        kind='scatter',
        facet_kws={
            # legend inside
            'legend_out': False,
            # titles of row and col on margins
            'margin_titles': True,
            # share axis
            'sharey': True,
            'sharex': True,
        },
        **kwargs,
    )
