import collections
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import predictive_coding as pc

import analysis_utils as au


def plot(df):

    df = au.nature_pre(df)

    g = au.nature_relplot(
        data=df,
        x='pc_learning_rate',
        y='Mean of test__classification_error',
        hue='Rule', style='Rule',
    ).set(xscale='log', yscale='log')

    au.nature_post(g, is_grid=True)


def plot_curve(df):

    df = au.nature_pre(df)

    df = au.extract_plot(df, f'test__classification_error',
                         'training_iteration')

    g = au.nature_relplot_curve(
        data=df,
        x='training_iteration',
        y=f'test__classification_error',
        hue='Rule', style='Rule',
        hue_order=['PC', 'BP'],
        style_order=['PC', 'BP'],
        aspect=1.0,
    )

    au.nature_post(g, is_grid=True)


def plot_curve_best(df, config_columns):

    df = au.nature_pre(df)

    df = au.select_best_lr(
        df, config_columns, 'Mean of test__classification_error')

    df = au.extract_plot(df, f'test__classification_error',
                         'training_iteration')

    g = au.nature_relplot_curve(
        data=df,
        x='training_iteration',
        y=f'test__classification_error',
        hue='Rule', style='Rule',
        hue_order=['PC', 'BP'],
        style_order=['PC', 'BP'],
        aspect=1.0,
    )

    au.nature_post(g, is_grid=True)
