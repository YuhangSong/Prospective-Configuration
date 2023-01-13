import copy
import numpy as np

import analysis_utils as au


def fit_data(df,
             config_columns,
             mean_columns,
             metric_column,
             method_column,
             plot_column,
             raw_data,
             process_plot_column_fn_in_raw_data=lambda plot: plot,
             fit_with='kb',
             is_return_best_fit_per_method=True,
             is_reduce_multiple_best_fit=True,
             is_print_best_fit_per_method=True,
             ):
    """See https://github.com/YuhangSong/general-energy-nets/tree/master/experiments#fit-to-biological-experiment-data

    Args:
        df (pd.DataFrame): The dataframe.
        config_columns (list): Columns corresponds to all configs.
        metric_column (str): A column corresponds to the metric.
        mean_column (str): A column corresponds to the searching space you want to average to get a mean of the metric.
        method_column (str): A column corresponds to the method.
        plot_column (str): A column corresponds to the axis generating the plot (first element in raw_data).
        raw_data (list): Data (cam be extracted with https://apps.automeris.io/wpd/).
            The first element corresponds plot_column; the second element corresponds metric_column.
            Need to be ordered according to the default sorting of plot_column by calling (df.sort_values(plot_column)).
            There can be duplications on the first axis, which will be considered as from different seeds. 
                But the first a few ones need to be sorted for me to infer the order of data.
        process_plot_column_fn_in_raw_data (callable): plot element in raw_data need to be processed.
            Example: lambda plot: np.round(plot)
        fit_with (str): Fit with
            'kb', model * k + b = data
            'k', model * k = data
        is_return_best_fit_per_method (bool): Whether to return only the best fit for each method.
        is_reduce_multiple_best_fit (bool): There could be multiple best fit, whether to reduce them.
        is_print_best_fit_per_method (bool): Whether to print the above results.

    Return:
        A dataframe with new columns:
            f'{metric_column}: fitted': The metric that is being fitted to data
            'k': The fitting parameter, coefficient.
            'b': The fitting parameter, bias.
            f'{metric_column}: fitting_error': The fitting error.
    """

    assert isinstance(config_columns, list)
    assert isinstance(mean_columns, list)
    assert isinstance(metric_column, str)
    assert isinstance(method_column, str)
    assert isinstance(plot_column, str)

    s = ''

    all_columns = config_columns + mean_columns + \
        [metric_column]+[method_column]+[plot_column]

    # checking that there are no duplicate columns in the list of all_columns.
    assert len(all_columns) == len(set(all_columns)), (
        "There cannot be overlap among config_columns, mean_columns, metric_column, method_column, plot_column. You have: \n"
        f"config_columns = {config_columns} \n"
        f"mean_columns = {mean_columns} \n"
        f"metric_column = {metric_column} \n"
        f"method_column = {method_column} \n"
        f"plot_column = {plot_column} \n"
    )

    # process the plot_column of raw_data with process_plot_column_fn_in_raw_data
    assert isinstance(raw_data, list), (
        f"raw_data = {raw_data} is not a list."
    )
    raw_data = copy.deepcopy(raw_data)

    assert callable(process_plot_column_fn_in_raw_data), (
        f"process_plot_column_fn_in_raw_data = {process_plot_column_fn_in_raw_data} is not callable."
    )

    for i in range(len(raw_data)):

        assert isinstance(raw_data[i], list), (
            f"raw_data[{i}] = {raw_data[i]} is not a list."
        )

        assert len(raw_data[i]) == 2, (
            f"raw_data[{i}] = {raw_data[i]} does not have length 2."
        )

        assert isinstance(raw_data[i][0], (str, int, float)), (
            f"raw_data[{i}][0] = {raw_data[i][0]} is not a string, int, or float. Instead, it is of type {type(raw_data[i][0])}."
        )
        assert isinstance(raw_data[i][1], (int, float)), (
            f"raw_data[{i}][1] = {raw_data[i][1]} is not an int or float. Instead, it is of type {type(raw_data[i][1])}."
        )
        raw_data[i][0] = process_plot_column_fn_in_raw_data(raw_data[i][0])

    # one row per config
    df = au.one_row_per_config(
        df,
        metric_columns=[metric_column],
        config_columns=config_columns + mean_columns +
        [method_column]+[plot_column],
    )

    # get {metric_column}: mean over mean_columns and {metric_column}: sem over mean_columns over mean_columns

    if len(mean_columns) > 0:

        df = au.add_metric_per_group(
            df, config_columns+[method_column]+[plot_column],
            lambda df: (
                f'{metric_column}: mean over mean_columns',
                df[metric_column].mean()
            )
        )

        df = au.add_metric_per_group(
            df, config_columns+[method_column]+[plot_column],
            lambda df: (
                f'{metric_column}: sem over mean_columns',
                df[metric_column].sem() if len(df[metric_column]) > 1 else -1.0
            )
        )
        df = au.add_metric_per_group(
            df, config_columns+[method_column],
            lambda df: (
                f'{metric_column}: sem over mean_columns: mean over plot_column',
                df[f'{metric_column}: sem over mean_columns'].mean()
            )
        )

    else:

        df = au.new_col(
            df,
            f'{metric_column}: mean over mean_columns',
            lambda row: row[metric_column]
        )

        df = au.new_col(
            df,
            f'{metric_column}: sem over mean_columns: mean over plot_column',
            lambda row: -1.0
        )

    # compute k and b

    data_dict_ = {}
    for raw_data_item_ in raw_data:
        if raw_data_item_[0] not in data_dict_.keys():
            data_dict_[raw_data_item_[0]] = [raw_data_item_[1]]
        else:
            data_dict_[raw_data_item_[0]].append(raw_data_item_[1])
    for k in data_dict_.keys():
        data_dict_[k] = np.mean(data_dict_[k])

    data = np.array([data_dict_[k] for k in data_dict_.keys()])

    def get_model(df):
        return df.sort_values(plot_column)[f'{metric_column}: mean over mean_columns'].to_numpy()

    def get_abmcdn(model, data):
        assert model.shape == data.shape
        assert len(model.shape) == 1
        # prepare for https://baike.baidu.com/item/%E9%80%9A%E8%A7%A3%E4%BA%8C%E5%85%83%E4%B8%80%E6%AC%A1%E6%96%B9%E7%A8%8B/3071492
        a = np.sum(model**2)
        b = np.sum(model)
        m = np.sum(np.multiply(model, data))
        c = np.sum(model)
        d = np.shape(model)[0]
        n = np.sum(data)
        return a, b, m, c, d, n

    def solve_k(model, data, fit_with):
        # solve x (k) with https://baike.baidu.com/item/%E9%80%9A%E8%A7%A3%E4%BA%8C%E5%85%83%E4%B8%80%E6%AC%A1%E6%96%B9%E7%A8%8B/3071492
        a, b, m, c, d, n = get_abmcdn(model, data)
        if fit_with == 'kb':
            return (b * n - d * m) / (b * c - a * d)
        elif fit_with == 'k':
            return m / a
        else:
            raise NotImplementedError

    def solve_b(model, data, fit_with):
        # solve y (b) with https://baike.baidu.com/item/%E9%80%9A%E8%A7%A3%E4%BA%8C%E5%85%83%E4%B8%80%E6%AC%A1%E6%96%B9%E7%A8%8B/3071492
        a, b, m, c, d, n = get_abmcdn(model, data)
        if fit_with == 'kb':
            return (a * n - c * m) / (a * d - b * c)
        else:
            raise NotImplementedError

    if 'k' in fit_with:
        df = au.add_metric_per_group(
            df, config_columns+mean_columns+[method_column],
            lambda df: (
                'k', solve_k(get_model(df), data, fit_with)
            )
        )

    if 'b' in fit_with:
        df = au.add_metric_per_group(
            df, config_columns+mean_columns+[method_column],
            lambda df: (
                'b', solve_b(get_model(df), data, fit_with)
            )
        )

    # produce {metric_column}: fitted

    df = au.new_col(
        df,
        f'{metric_column}: fitted',
        lambda row: (
            row[metric_column] *
            row['k'] if 'k' in fit_with else row[metric_column]
        ) + (
            row['b'] if 'b' in fit_with else 0.0
        )
    )

    # produce {metric_column}: mean over mean_columns: fitted

    df = au.new_col(
        df,
        f'{metric_column}: mean over mean_columns: fitted',
        lambda row: (
            row[f'{metric_column}: mean over mean_columns'] *
            row['k'] if 'k' in fit_with else row[metric_column]
        ) + (
            row['b'] if 'b' in fit_with else 0.0
        )
    )

    df = au.new_col(
        df,
        f'{metric_column}: sem over mean_columns: mean over plot_column: fitted',
        lambda row: (
            row[f'{metric_column}: sem over mean_columns: mean over plot_column'] *
            row['k'] if 'k' in fit_with else row[f'{metric_column}: sem over mean_columns: mean over plot_column']
        )
    )

    # produce {metric_column}: fitting_error

    df = au.add_metric_per_group(
        df, config_columns+mean_columns+[method_column],
        lambda df: (
            f'{metric_column}: mean over mean_columns: fitting_error', np.sum(
                (
                    df.sort_values(plot_column)[
                        f'{metric_column}: mean over mean_columns: fitted'
                    ].to_numpy() - data
                )**2
            )
        )
    )

    # add data

    def add_data_row(df):
        for i in range(len(raw_data)):
            data_row = df.iloc[-1].copy()
            data_row[method_column] = 'Data'
            data_row[plot_column] = raw_data[i][0]
            data_row[f'{metric_column}: fitted'] = raw_data[i][1]
            df = df.append(data_row)
        return df

    if len(config_columns) > 0:
        df = df.groupby(
            config_columns,
        )
        df = df.apply(
            add_data_row,
        ).reset_index(drop=True)

    else:
        df = add_data_row(df)

    if is_return_best_fit_per_method:

        # first criteria is to keep only the ones with minimum {metric_column}: mean over mean_columns: fitting_error
        df = au.select_rows_per_group(
            df, [method_column],
            lambda df: df[f'{metric_column}: mean over mean_columns: fitting_error'] == df[
                f'{metric_column}: mean over mean_columns: fitting_error'].min()
        )

        # second criteria is to keep only the ones with minimum {metric_column}: sem over mean_columns: mean over plot_column: fitted
        df = au.select_rows_per_group(
            df, [method_column],
            lambda df: df[
                f'{metric_column}: sem over mean_columns: mean over plot_column: fitted'
            ] == df[f'{metric_column}: sem over mean_columns: mean over plot_column: fitted'].min()
        )

        # if there are still multiple best fits, these multiple bet fits cannot be distinguished just on the fitting results
        if is_reduce_multiple_best_fit:
            # reduce by just keeping one config then
            for config_column in config_columns:
                df = au.select_rows_per_group(
                    df, [method_column],
                    lambda df: df[config_column] == df[config_column].iloc[0]
                )

        if is_print_best_fit_per_method:
            print_df = df[
                [method_column] +
                [f'{metric_column}: mean over mean_columns: fitting_error'] +
                config_columns
            ].drop_duplicates()
            print_df = print_df[
                print_df[method_column] != 'Data'
            ]
            print_df = au.df2tb(print_df)
            s += print_df
            print()
            print()
            print(print_df)
            print()
            print()

    return df, s
