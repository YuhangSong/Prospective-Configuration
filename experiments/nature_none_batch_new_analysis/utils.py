import collections
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import predictive_coding as pc

import analysis_utils as au


def plot_mean(df):

    df = au.nature_pre(df)
    g = au.nature_relplot(
        data=df,
        x='pc_learning_rate',
        y='Mean of test__classification_error',
        hue='Rule', style='Rule',
        col='batch_size',
        row='num_batch_per_iteration',
        sharey=False,
    ).set(xscale='log')
    au.nature_post(g, is_grid=True)


def plot_min(df):

    df = au.nature_pre(df)
    g = au.nature_relplot(
        data=df,
        x='pc_learning_rate',
        y='Min of test__classification_error',
        hue='Rule', style='Rule',
        col='batch_size',
        row='num_batch_per_iteration',
        sharey=False,
    ).set(xscale='log')
    au.nature_post(g, is_grid=True)


def select_best_lr(df):

    groups = ['num_batch_per_iteration',
              'batch_size', 'Rule', 'pc_learning_rate']

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

    return df


def plot_mean_best(df):

    df = au.nature_pre(df)

    df = select_best_lr(df)

    df = df.sort_values(['batch_size'], ascending=False)

    df = au.new_col(df, 'batch_size_str', lambda row: str(row['batch_size']))

    g = au.nature_relplot(
        data=df,
        x='batch_size_str',
        y='Mean of test__classification_error',
        hue='Rule', style='Rule',
        hue_order=['PC', 'BP'],
        style_order=['PC', 'BP'],
        row='num_batch_per_iteration',
        sharey=False,
    )

    au.nature_post(g, is_grid=True)

    return df


def plot_curve(df):

    df = au.nature_pre(df)

    df = select_best_lr(df)

    df = au.extract_plot(df, f'test__classification_error',
                         'training_iteration')

    g = au.nature_relplot_curve(
        data=df,
        x='training_iteration',
        y=f'test__classification_error',
        hue='Rule', style='Rule',
        hue_order=['PC', 'BP'],
        style_order=['PC', 'BP'],
        col='batch_size',
        row='num_batch_per_iteration',
        aspect=0.8,
        sharey=False
    )

    au.nature_post(g, is_grid=True)


def plot_curve_best(df):

    df = au.nature_pre(df)

    df = select_best_lr(df)

    df = df.sort_values(['batch_size'], ascending=False)

    df = au.new_col(df, 'batch_size_str', lambda row: str(row['batch_size']))

    df = au.extract_plot(df, f'test__classification_error',
                         'training_iteration')

    g = au.nature_relplot_curve(
        data=df,
        x='training_iteration',
        y='test__classification_error',
        hue='Rule', style='Rule',
        hue_order=['PC', 'BP'],
        style_order=['PC', 'BP'],
        col='batch_size_str',
        row='num_batch_per_iteration',
        sharey=False,
    ).set(yscale='log')

    au.nature_post(g, is_grid=True)
