import pytest
import os
import json
import pandas as pd
import numpy as np
import seaborn as sns

import utils as u
import analysis_utils as au

from fit_data import fit_data


@pytest.fixture
def get_fig_save_dir():
    return os.path.join(
        'tests',
        'test_fit_data',
    )


@pytest.fixture
def get_relvent_columns():
    def get_relvent_columns(df):
        return df[['method', 'learning_rate', 'plot', 'seed', 'metric: fitted']]
    return get_relvent_columns


@pytest.fixture
def default_kwargs():

    df = pd.DataFrame({
        'method':        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        'learning_rate': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        'plot':          [1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3],
        'seed':          [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
    })

    def metric_fn(row):
        if row['method'] == 1:
            return (row['learning_rate'] * row['plot'] * row['seed']) if row['plot'] in [1, 2] else (row['learning_rate'] * row['plot'] * row['seed']-1)
        elif row['method'] == 2:
            return (row['learning_rate'] * row['plot'] * row['seed']) if row['plot'] in [1, 2] else 1

    df = au.new_col(
        df, 'metric', metric_fn
    )

    return {
        'df': df,
        'config_columns': [
            'learning_rate',
        ],
        'mean_columns': [
            'seed',
        ],
        'metric_column': 'metric',
        'method_column': 'method',
        'plot_column': 'plot',
    }


def test_basic(get_fig_save_dir, get_relvent_columns, default_kwargs):

    expected_df = pd.DataFrame.from_dict({
        "learning_rate": [
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0
        ],
        "method": [
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            "Data",
            "Data",
            "Data"
        ],
        "metric: fitted": [
            0.05526315789473669,
            0.1342105263157893,
            0.17368421052631564,
            0.1342105263157893,
            0.2921052631578946,
            0.4105263157894735,
            0.2210526315789474,
            0.18947368421052632,
            0.23684210526315794,
            0.18947368421052632,
            0.12631578947368416,
            0.23684210526315794,
            0.1,
            0.2,
            0.3
        ],
        "plot": [
            1,
            2,
            3,
            1,
            2,
            3,
            1,
            2,
            3,
            1,
            2,
            3,
            1,
            2,
            3
        ],
        "seed": [
            1.0,
            1.0,
            1.0,
            2.0,
            2.0,
            2.0,
            1.0,
            1.0,
            1.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0
        ]
    })

    fig_save_dir = os.path.join(
        get_fig_save_dir, 'test_basic',
    )

    sns.relplot(
        data=default_kwargs['df'],
        row='method',
        col='learning_rate',
        x='plot',
        y='metric',
        style='seed',
    )
    au.save_fig(fig_save_dir, 'metric')

    # Test case: basic

    fit_data_df, _ = fit_data(
        **default_kwargs,
        raw_data=[
            [1, 0.1],
            [2, 0.2],
            [3, 0.3],
        ],
    )

    # u.print_df_as_dict(get_relvent_columns(fit_data_df))
    u.assert_frame_equal(
        get_relvent_columns(fit_data_df), expected_df
    )

    sns.relplot(
        kind='line',
        data=fit_data_df,
        hue='method',
        x='plot',
        y='metric: fitted',
        style='seed',
    )
    au.save_fig(fig_save_dir, 'metric_fitted_per_seed')

    sns.relplot(
        kind='line',
        data=fit_data_df,
        hue='method',
        x='plot',
        y='metric: fitted',
    )
    au.save_fig(fig_save_dir, 'metric_fitted')

    # Test case: process_plot_column_fn_in_raw_data

    fit_data_df, _ = fit_data(
        **default_kwargs,
        raw_data=[
            [1.1, 0.1],
            [1.9, 0.2],
            [3.1, 0.3],
        ],
        process_plot_column_fn_in_raw_data=lambda plot: np.round(plot),
    )

    u.assert_frame_equal(
        get_relvent_columns(fit_data_df), expected_df
    )

    # Test case: is_print_best_fit_per_method

    fit_data_df, _ = fit_data(
        **default_kwargs,
        raw_data=[
            [1, 0.1],
            [2, 0.2],
            [3, 0.3],
        ],
        is_print_best_fit_per_method=False,
    )

    u.assert_frame_equal(
        get_relvent_columns(
            fit_data_df[fit_data_df.method != 'Data']
        ), expected_df[
            expected_df.method != 'Data'
        ]
    )


def test_fit_with_k(get_fig_save_dir, get_relvent_columns, default_kwargs):

    expected_df = pd.DataFrame.from_dict({
        "learning_rate": [
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0
        ],
        "method": [
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            "Data",
            "Data",
            "Data"
        ],
        "metric: fitted": [
            0.07155963302752294,
            0.14311926605504588,
            0.17889908256880735,
            0.14311926605504588,
            0.28623853211009176,
            0.39357798165137614,
            0.08571428571428572,
            0.17142857142857143,
            0.08571428571428572,
            0.17142857142857143,
            0.34285714285714286,
            0.08571428571428572,
            0.1,
            0.2,
            0.3
        ],
        "plot": [
            1,
            2,
            3,
            1,
            2,
            3,
            1,
            2,
            3,
            1,
            2,
            3,
            1,
            2,
            3
        ],
        "seed": [
            1.0,
            1.0,
            1.0,
            2.0,
            2.0,
            2.0,
            1.0,
            1.0,
            1.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0
        ]
    })

    fig_save_dir = os.path.join(
        get_fig_save_dir, 'test_fit_with_k',
    )

    fit_data_df, _ = fit_data(
        **default_kwargs,
        raw_data=[
            [1, 0.1],
            [2, 0.2],
            [3, 0.3],
        ],
        fit_with='k',
    )

    # u.print_df_as_dict(get_relvent_columns(fit_data_df))
    u.assert_frame_equal(
        get_relvent_columns(fit_data_df), expected_df
    )

    sns.relplot(
        kind='line',
        data=fit_data_df,
        hue='method',
        x='plot',
        y='metric: fitted',
        style='seed',
    )
    au.save_fig(fig_save_dir, 'metric_fitted_per_seed')

    sns.relplot(
        kind='line',
        data=fit_data_df,
        hue='method',
        x='plot',
        y='metric: fitted',
    )
    au.save_fig(fig_save_dir, 'metric_fitted')


def test_not_return_best_fit_per_method(get_fig_save_dir, get_relvent_columns, default_kwargs):

    expected_df = pd.DataFrame.from_dict({
        "learning_rate": [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0
        ],
        "method": [
            1,
            1,
            1,
            2,
            2,
            2,
            1,
            1,
            1,
            2,
            2,
            2,
            "Data",
            "Data",
            "Data",
            1,
            1,
            1,
            2,
            2,
            2,
            1,
            1,
            1,
            2,
            2,
            2,
            "Data",
            "Data",
            "Data"
        ],
        "metric: fitted": [
            0.046153846153846,
            0.13846153846153825,
            0.13846153846153825,
            0.21923076923076934,
            0.1961538461538462,
            0.21923076923076934,
            0.13846153846153825,
            0.32307692307692276,
            0.415384615384615,
            0.1961538461538462,
            0.14999999999999997,
            0.21923076923076934,
            0.1,
            0.2,
            0.3,
            0.05526315789473669,
            0.1342105263157893,
            0.17368421052631564,
            0.2210526315789474,
            0.18947368421052632,
            0.23684210526315794,
            0.1342105263157893,
            0.2921052631578946,
            0.4105263157894735,
            0.18947368421052632,
            0.12631578947368416,
            0.23684210526315794,
            0.1,
            0.2,
            0.3
        ],
        "plot": [
            1,
            2,
            3,
            1,
            2,
            3,
            1,
            2,
            3,
            1,
            2,
            3,
            1,
            2,
            3,
            1,
            2,
            3,
            1,
            2,
            3,
            1,
            2,
            3,
            1,
            2,
            3,
            1,
            2,
            3
        ],
        "seed": [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0
        ]
    })

    fig_save_dir = os.path.join(
        get_fig_save_dir, 'test_not_return_best_fit_per_method',
    )

    fit_data_df, _ = fit_data(
        **default_kwargs,
        raw_data=[
            [1, 0.1],
            [2, 0.2],
            [3, 0.3],
        ],
        is_return_best_fit_per_method=False,
    )

    # u.print_df_as_dict(get_relvent_columns(fit_data_df))
    u.assert_frame_equal(
        get_relvent_columns(fit_data_df), expected_df
    )

    sns.relplot(
        kind='line',
        data=fit_data_df,
        hue='method',
        x='plot',
        y='metric: fitted',
        style='seed',
        col='learning_rate',
    )
    au.save_fig(fig_save_dir, 'metric_fitted_per_seed')

    sns.relplot(
        kind='line',
        data=fit_data_df,
        hue='method',
        x='plot',
        y='metric: fitted',
        col='learning_rate',
    )
    au.save_fig(fig_save_dir, 'metric_fitted')
