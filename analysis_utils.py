import copy
import json
import numbers
import os
import pickle
import re
import shutil
import yaml
import pprint
import traceback

import torch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
import utils as u
import plotly.express as px
import plotly.graph_objs as go

from tabulate import tabulate
from typing import Any, Dict, List, Optional, Tuple

from ray.tune.experiment.trial import Trial
from ray.tune.result import DEFAULT_METRIC, EXPR_PARAM_FILE, EXPR_PROGRESS_FILE, \
    CONFIG_PREFIX, TRAINING_ITERATION
from ray.tune.search.variant_generator import generate_variants


logger = u.getLogger(__name__)

pp = pprint.PrettyPrinter(indent=4)

"""
    resolve_nested_dict has been made private in later release of ray, so keep a copy here
"""

# from ray.tune.suggest.variant_generator import resolve_nested_dict


def resolve_nested_dict(nested_dict: Dict) -> Dict[Tuple, Any]:
    """Flattens a nested dict by joining keys into tuple of paths.
    Can then be passed into `format_vars`.
    """
    res = {}
    for k, v in nested_dict.items():
        if isinstance(v, dict):
            for k_, v_ in resolve_nested_dict(v).items():
                res[(k,) + k_] = v_
        else:
            res[(k,)] = v
    return res


"""
    Analysis has been depreciated by ray in favor of ExperimentAnalysis, which only loads recent experiment using the large json file.
    This Analysis is what needed by this lib so keep a copy of it from ray==1.4.0
"""


class Analysis:
    """Analyze all results from a directory of experiments.
    To use this class, the experiment must be executed with the JsonLogger.
    Args:
        experiment_dir (str): Directory of the experiment to load.
        default_metric (str): Default metric for comparing results. Can be
            overwritten with the ``metric`` parameter in the respective
            functions. If None but a mode was passed, the anonymous metric
            `ray.tune.result.DEFAULT_METRIC` will be used per default.
        default_mode (str): Default mode for comparing results. Has to be one
            of [min, max]. Can be overwritten with the ``mode`` parameter
            in the respective functions.
    """

    def __init__(self,
                 experiment_dir: str,
                 default_metric: Optional[str] = None,
                 default_mode: Optional[str] = None):
        experiment_dir = os.path.expanduser(experiment_dir)
        if not os.path.isdir(experiment_dir):
            raise ValueError(
                "{} is not a valid directory.".format(experiment_dir))
        self._experiment_dir = experiment_dir
        self._configs = {}
        self._trial_dataframes = {}

        self.default_metric = default_metric
        if default_mode and default_mode not in ["min", "max"]:
            raise ValueError(
                "`default_mode` has to be None or one of [min, max]")
        self.default_mode = default_mode

        if self.default_metric is None and self.default_mode:
            # If only a mode was passed, use anonymous metric
            self.default_metric = DEFAULT_METRIC

        if not pd:
            logger.warning(
                "pandas not installed. Run `pip install pandas` for "
                "Analysis utilities.")
        else:
            self.fetch_trial_dataframes()

    def _validate_metric(self, metric: str) -> str:
        if not metric and not self.default_metric:
            raise ValueError(
                "No `metric` has been passed and  `default_metric` has "
                "not been set. Please specify the `metric` parameter.")
        return metric or self.default_metric

    def _validate_mode(self, mode: str) -> str:
        if not mode and not self.default_mode:
            raise ValueError(
                "No `mode` has been passed and  `default_mode` has "
                "not been set. Please specify the `mode` parameter.")
        if mode and mode not in ["min", "max"]:
            raise ValueError("If set, `mode` has to be one of [min, max]")
        return mode or self.default_mode

    def dataframe(self,
                  metric: Optional[str] = None,
                  mode: Optional[str] = None) -> pd.DataFrame:
        """Returns a pd.DataFrame object constructed from the trials.
        Args:
            metric (str): Key for trial info to order on.
                If None, uses last result.
            mode (str): One of [min, max].
        Returns:
            pd.DataFrame: Constructed from a result dict of each trial.
        """
        # Allow None values here.
        if metric or self.default_metric:
            metric = self._validate_metric(metric)
        if mode or self.default_mode:
            mode = self._validate_mode(mode)

        rows = self._retrieve_rows(metric=metric, mode=mode)
        all_configs = self.get_all_configs(prefix=True)
        for path, config in all_configs.items():
            if path in rows:
                rows[path].update(config)
                rows[path].update(logdir=path)
        return pd.DataFrame(list(rows.values()))

    def get_best_config(self,
                        metric: Optional[str] = None,
                        mode: Optional[str] = None) -> Optional[Dict]:
        """Retrieve the best config corresponding to the trial.
        Args:
            metric (str): Key for trial info to order on. Defaults to
                ``self.default_metric``.
            mode (str): One of [min, max]. Defaults to
                ``self.default_mode``.
        """
        metric = self._validate_metric(metric)
        mode = self._validate_mode(mode)

        rows = self._retrieve_rows(metric=metric, mode=mode)
        if not rows:
            # only nans encountered when retrieving rows
            logger.warning("Not able to retrieve the best config for {} "
                           "according to the specified metric "
                           "(only nans encountered).".format(
                               self._experiment_dir))
            return None
        all_configs = self.get_all_configs()
        compare_op = max if mode == "max" else min
        best_path = compare_op(rows, key=lambda k: rows[k][metric])
        return all_configs[best_path]

    def get_best_logdir(self,
                        metric: Optional[str] = None,
                        mode: Optional[str] = None) -> Optional[str]:
        """Retrieve the logdir corresponding to the best trial.
        Args:
            metric (str): Key for trial info to order on. Defaults to
                ``self.default_metric``.
            mode (str): One of [min, max]. Defaults to ``self.default_mode``.
        """
        metric = self._validate_metric(metric)
        mode = self._validate_mode(mode)

        assert mode in ["max", "min"]
        df = self.dataframe(metric=metric, mode=mode)
        mode_idx = pd.Series.idxmax if mode == "max" else pd.Series.idxmin
        try:
            return df.iloc[mode_idx(df[metric])].logdir
        except KeyError:
            # all dirs contains only nan values
            # for the specified metric
            # -> df is an empty dataframe
            logger.warning("Not able to retrieve the best logdir for {} "
                           "according to the specified metric "
                           "(only nans encountered).".format(
                               self._experiment_dir))
            return None

    def fetch_trial_dataframes(self) -> Dict[str, pd.DataFrame]:
        fail_count = 0
        for path in self._get_trial_paths():
            try:
                self.trial_dataframes[path] = pd.read_csv(
                    os.path.join(path, EXPR_PROGRESS_FILE))
            except Exception:
                fail_count += 1

        if fail_count:
            logger.debug(
                "Couldn't read results from {} paths".format(fail_count))
        return self.trial_dataframes

    def get_all_configs(self, prefix: bool = False) -> Dict[str, Dict]:
        """Returns a list of all configurations.
        Args:
            prefix (bool): If True, flattens the config dict
                and prepends `config/`.
        Returns:
            Dict[str, Dict]: Dict of all configurations of trials, indexed by
                their trial dir.
        """
        fail_count = 0
        for path in self._get_trial_paths():
            try:
                with open(os.path.join(path, EXPR_PARAM_FILE)) as f:
                    config = json.load(f)
                    if prefix:
                        for k in list(config):
                            config[CONFIG_PREFIX + k] = config.pop(k)
                    self._configs[path] = config
            except Exception:
                fail_count += 1

        if fail_count:
            logger.warning(
                "Couldn't read config from {} paths".format(fail_count))
        return self._configs

    def get_trial_checkpoints_paths(self,
                                    trial: Trial,
                                    metric: Optional[str] = None
                                    ) -> List[Tuple[str, numbers.Number]]:
        """Gets paths and metrics of all persistent checkpoints of a trial.
        Args:
            trial (Trial): The log directory of a trial, or a trial instance.
            metric (str): key for trial info to return, e.g. "mean_accuracy".
                "training_iteration" is used by default if no value was
                passed to ``self.default_metric``.
        Returns:
            List of [path, metric] for all persistent checkpoints of the trial.
        """
        metric = metric or self.default_metric or TRAINING_ITERATION

        if isinstance(trial, str):
            trial_dir = os.path.expanduser(trial)
            # Get checkpoints from logdir.
            chkpt_df = TrainableUtil.get_checkpoints_paths(trial_dir)

            # Join with trial dataframe to get metrics.
            trial_df = self.trial_dataframes[trial_dir]
            path_metric_df = chkpt_df.merge(
                trial_df, on="training_iteration", how="inner")
            return path_metric_df[["chkpt_path", metric]].values.tolist()
        elif isinstance(trial, Trial):
            checkpoints = trial.checkpoint_manager.best_checkpoints()
            # Support metrics given as paths, e.g.
            # "info/learner/default_policy/policy_loss".
            return [(c.value, unflattened_lookup(metric, c.result))
                    for c in checkpoints]
        else:
            raise ValueError("trial should be a string or a Trial instance.")

    def get_best_checkpoint(self,
                            trial: Trial,
                            metric: Optional[str] = None,
                            mode: Optional[str] = None) -> Optional[str]:
        """Gets best persistent checkpoint path of provided trial.
        Args:
            trial (Trial): The log directory of a trial, or a trial instance.
            metric (str): key of trial info to return, e.g. "mean_accuracy".
                "training_iteration" is used by default if no value was
                passed to ``self.default_metric``.
            mode (str): One of [min, max]. Defaults to ``self.default_mode``.
        Returns:
            Path for best checkpoint of trial determined by metric
        """
        metric = metric or self.default_metric or TRAINING_ITERATION
        mode = self._validate_mode(mode)

        checkpoint_paths = self.get_trial_checkpoints_paths(trial, metric)
        if not checkpoint_paths:
            logger.error(f"No checkpoints have been found for trial {trial}.")
            return None
        if mode == "max":
            return max(checkpoint_paths, key=lambda x: x[1])[0]
        else:
            return min(checkpoint_paths, key=lambda x: x[1])[0]

    def get_last_checkpoint(self,
                            trial=None,
                            metric="training_iteration",
                            mode="max"):
        """Helper function that wraps Analysis.get_best_checkpoint().
        Gets the last persistent checkpoint path of the provided trial,
        i.e., with the highest "training_iteration".
        If no trial is specified, it loads the best trial according to the
        provided metric and mode (defaults to max. training iteration).
        Args:
            trial (Trial): The log directory or an instance of a trial.
            If None, load the latest trial automatically.
            metric (str): If no trial is specified, use this metric to identify
            the best trial and load the last checkpoint from this trial.
            mode (str): If no trial is specified, use the metric and this mode
            to identify the best trial and load the last checkpoint from it.
        Returns:
            Path for last checkpoint of trial
        """
        if trial is None:
            trial = self.get_best_logdir(metric, mode)

        return self.get_best_checkpoint(trial, "training_iteration", "max")

    def _retrieve_rows(self,
                       metric: Optional[str] = None,
                       mode: Optional[str] = None) -> Dict[str, Any]:
        assert mode is None or mode in ["max", "min"]
        rows = {}
        for path, df in self.trial_dataframes.items():
            if mode == "max":
                idx = df[metric].idxmax()
            elif mode == "min":
                idx = df[metric].idxmin()
            else:
                idx = -1
            try:
                rows[path] = df.iloc[idx].to_dict()
            except TypeError:
                # idx is nan
                logger.warning(
                    "Warning: Non-numerical value(s) encountered for {}".
                    format(path))

        return rows

    def _get_trial_paths(self) -> List[str]:
        _trial_paths = []
        for trial_path, _, files in os.walk(self._experiment_dir):
            if EXPR_PROGRESS_FILE in files:
                _trial_paths += [trial_path]

        if not _trial_paths:
            raise TuneError("No trials found in {}.".format(
                self._experiment_dir))
        return _trial_paths

    @property
    def trial_dataframes(self) -> Dict[str, pd.DataFrame]:
        """List of all dataframes of the trials."""
        return self._trial_dataframes


"""
    Produce Nature-level plots.
"""


def nature_pre(df, our_name='PC', base_name='BP'):

    # rename
    if 'PC' in df.columns:
        df.insert(
            1, 'Rule',
            df.apply(
                lambda row: {
                    True: our_name,
                    False: base_name,
                    'TP': 'TP',
                }[row['PC']], axis=1
            )
        )
    elif 'Rule' in df.columns:
        df.insert(
            1, 'Rule-t',
            df.apply(
                lambda row: {
                    'Predictive-Coding': our_name,
                    'Back-Propagation': base_name
                }[row['Rule']], axis=1
            )
        )
        del df['Rule']
        df = df.rename(columns={'Rule-t': 'Rule'})
    else:
        raise NotImplementedError

    if 'ns' in df.columns:
        df.insert(
            1, 'Hidden size',
            df.apply(
                lambda row: eval(row['ns'])[1], axis=1
            )
        )

    # sort
    df = df.sort_values(['Rule'], ascending=False)
    df1 = df.set_index('Rule')
    sort_order = ['PC', 'BP']
    if 'TP' in df['Rule'].unique().tolist():
        sort_order.append('TP')
    df = df1.loc[sort_order].reset_index()

    return df


def nature_relplot(kind='line', sharey=True, sharex=True, legend_out=False, **kwargs):
    """
    """
    return sns.relplot(
        errorbar=('ci', 68),
        # standard error
        # historically, I was using errorbar=('ci', 68),
        # this is because 68%CI = Score ±SEM, see https://www.statisticshowto.com/standard-error-of-measurement/
        # but latter Rafal noticed that the error bars could be asymmetric,
        # the reason is that
        # Seaborn's errorbar function produces asymmetric error bars by default because it uses the bootstrapped confidence intervals for the error bars.
        # This means that it samples the data with replacement to calculate the confidence intervals.
        # Bootstrapping is a way to estimate the sampling distribution of a statistic (such as the mean or standard deviation) using the data itself,
        # without making assumptions about the underlying distribution.
        # This is a robust method that can work well even when the underlying distribution is not normal.
        # However, it can result in asymmetric error bars.
        # seaborn's documentation: https://seaborn.pydata.org/tutorial/error_bars.html#confidence-interval-error-bars
        # so I switched to 'se'
        # errorbar=("se"),
        # line
        kind=kind,
        # with markers
        markers=True,
        facet_kws={
            # legend inside
            'legend_out': legend_out,
            # titles of row and col on margins
            'margin_titles': True,
            # share axis
            'sharey': sharey,
            'sharex': sharex,
        },
        # bars for error
        err_style='bars',
        err_kws={
            # settings for error bars
            'capsize': 6,
            'capthick': 2,
        },
        **kwargs,
    )


def nature_relplot_curve(sharey=True, sharex=True, legend_out=False, **kwargs):
    """
    """
    return sns.relplot(
        # standard error
        errorbar=('ci', 68),
        # line
        kind='line',
        # without markers
        markers=False,
        facet_kws={
            # legend inside
            'legend_out': legend_out,
            # share axis
            'sharey': sharey,
            'sharex': sharex,
            'margin_titles': True,
        },
        # band for error
        err_style='band',
        **kwargs,
    )


def nature_catplot(**kwargs):
    """
    """
    return sns.catplot(
        # standard error
        errorbar=('ci', 68),
        # legend inside
        legend_out=False,
        # titles of row and col on margins
        margin_titles=True,
        **kwargs,
    )


def nature_catplot_sharey(*args, **kwargs):
    """
    """
    return sns.catplot(
        *args, **kwargs,
        # standard error
        errorbar=('ci', 68),
        # bar
        kind='bar',
        # legend inside
        legend_out=False,
        # titles of row and col on margins
        margin_titles=True,
        # sharing y
        sharey=True,
        # # bars for error
        # err_style='bars',
        # err_kws={
        #     # settings for error bars
        #     'capsize': 6,
        #     'capthick': 2,
        # },
    )


def nature_post(g, xticks=None, yticks=None, is_grid=True):
    # set xticks
    if xticks is not None:
        if isinstance(xticks, list):
            [ax.set_xticks(xticks) for ax in g.axes.flat]
            [ax.set_xticklabels(xticks) for ax in g.axes.flat]
        else:
            raise NotImplementedError
    if yticks is not None:
        if isinstance(yticks, list):
            [ax.set_yticks(yticks) for ax in g.axes.flat]
            [ax.set_yticklabels(yticks) for ax in g.axes.flat]
        else:
            raise NotImplementedError
    # create grid
    if is_grid:
        [ax.grid(True, which='both') for ax in g.axes.flat]


"""
    End of Produce Nature-level plots.
"""


def format_friendly_string(string, is_abbr=False):
    """See https://github.com/YuhangSong/general-energy-nets/tree/master/experiments#you-may-want-to-format-a-string-to-be-friendly-for-being-a-dictionary

        Format a string to be friendly. Specifically,
            1, replace all special characters, punctuation and spaces from string with '_'
            2, remove consecutive duplicates of '_'
            3, remove '_' at the start and end

        Args:
            is_abbr (bool): if use abbreviations.
    """

    # replace all special characters, punctuation and spaces from string with '_'
    friendly_string = []
    for e in string:
        if e.isalnum():
            friendly_string.append(e)
        else:
            friendly_string.append('_')

    # remove consecutive duplicates of '_'
    friendly_string = re.sub(r'(_)\1+', r'\1', ''.join(friendly_string))

    # remove '_' at the start and end
    if len(friendly_string) > 0:
        if friendly_string[0] == '_':
            friendly_string = friendly_string[1:]
    if len(friendly_string) > 0:
        if friendly_string[-1] == '_':
            friendly_string = friendly_string[:-1]

    if is_abbr:
        logger.warning(
            "Replacing some long names with abbreviations, which should not cause any confusion, but be careful with this. "
            "If you saw some images got overwritten (flashes while running <analysis_v1.py>), it is because of this. "
            "You then need to turn this off or check if the following replacing is causing some confusion: different configs got the same abbreviation. "
        )
        friendly_string = friendly_string.replace('__', '_')
        friendly_string = friendly_string.replace('True', 'T')
        friendly_string = friendly_string.replace('False', 'F')
        friendly_string = friendly_string.replace(
            'test_classification_error', 'te_ce'
        )
        friendly_string = friendly_string.replace(
            'train_classification_error', 'tr_ce'
        )
        friendly_string = friendly_string.replace('M', 'M')
        friendly_string = friendly_string.replace('FashionMNIST', 'FM')
        friendly_string = friendly_string.replace('CIFAR10', 'C10')
        friendly_string = friendly_string.replace('CIFAR100', 'C100')
        friendly_string = friendly_string.replace('optim_', '')
        friendly_string = friendly_string.replace('torch_nn_init_', '')
        friendly_string = friendly_string.replace('lambda_', '')
        friendly_string = friendly_string.replace('F_', '')
        friendly_string = friendly_string.replace('init_', '')
        friendly_string = friendly_string.replace('xavier', 'xa')
        friendly_string = friendly_string.replace('kaiming', 'ka')
        friendly_string = friendly_string.replace('normal', 'n')
        friendly_string = friendly_string.replace('uniform', 'u')
        friendly_string = friendly_string.replace('reciprocal', 'r')
        friendly_string = friendly_string.replace('friendly_', '')
    return friendly_string


def save_fig(logdir, title, formats=['png'], savefig_kwargs={}):
    """Save current figure for plt.

    Args:
        logdir (str): Log directory of the figure.
        title (str): Title of the figure.
        formats (list): A list of formats.
            Example: formats=['pdf', 'png']
        savefig_kwargs (dict): Keyword arguments passed to <plt.savefig()>.
            Example: savefig_kwargs={'pad_inches': 0.1,
                'dpi': 500, 'bbox_inches': 'tight'}

    Returns:
        paths (dict): e.g.,
            {
                'png': 'path_to_png_figure.png',
                'pdf': 'path_to_pdf_figure.pdf',
            }
    """

    assert isinstance(logdir, str)
    assert isinstance(title, str)
    assert isinstance(formats, list)
    for format in formats:
        assert isinstance(format, str)
    assert isinstance(savefig_kwargs, dict)

    logger.info(f"Save fig {title} to {logdir} in formats {formats}.")

    u.prepare_dir(logdir)

    paths = {}

    for format in formats:
        paths[format] = '{}/{}.{}'.format(logdir, title, format)
        plt.savefig(
            paths[format],
            **savefig_kwargs,
        )

    return paths


class _SafeFallbackEncoder(json.JSONEncoder):
    def __init__(self, nan_str="null", **kwargs):
        super(_SafeFallbackEncoder, self).__init__(**kwargs)
        self.nan_str = nan_str

    def default(self, value):
        try:
            if np.isnan(value):
                return self.nan_str

            if (type(value).__module__ == np.__name__
                    and isinstance(value, np.ndarray)):
                return value.tolist()

            if issubclass(type(value), numbers.Integral):
                return int(value)
            if issubclass(type(value), numbers.Number):
                return float(value)

            return super(_SafeFallbackEncoder, self).default(value)

        except Exception:
            return str(value)  # give up, just stringify it (ok for logs)


def save_dict(d, logdir, title, formats=['pickle']):
    """Save a dictionary

    Args:
        d (dict): The dictionary to save.
        logdir (str): Directory to save.
        title (str): Title to save.
        formats (list): A list of formats to save, as the following: json, pickle, yaml.
    """

    assert isinstance(d, dict)
    assert isinstance(logdir, str)
    assert isinstance(title, str)
    assert isinstance(formats, list)
    for format in formats:
        assert isinstance(format, str)

    logger.info(f"Saving dict {title} to {logdir} in formats {formats}.")

    u.prepare_dir(logdir)

    for format in formats:

        path = os.path.join(logdir, '{}.{}'.format(title, format))

        if format == 'json':
            with open(path, "w") as f:
                json.dump(
                    d, f,
                    indent=2,
                    sort_keys=True,
                    cls=_SafeFallbackEncoder)

        elif format == 'pickle':
            with open(path, 'wb') as handle:
                pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

        elif format == 'yaml':
            with open(path, 'w') as f:
                yaml.dump(d, f)

        else:
            raise NotADirectoryError


def load_dict(logdir=None, title=None, format='pickle', path=None):
    """Load a dictionary

    Args:
        logdir (None | str): The log directory.
        title (None | str): The title.
        format (None | str): The format in the following: json, pickle, yaml.
        path (None | str): The path, which will override logdir, title and format.
    """

    if (path is not None):
        assert isinstance(path, str)
        assert (logdir is None) and (title is None)
        logger.info(f"Loading dict from path {path}")
    elif (logdir is not None) and (title is not None) and (format is not None):
        assert isinstance(logdir, str)
        assert isinstance(title, str)
        assert isinstance(format, str)
        logger.info(f"Loading dict {title} from {logdir} in format {format}")
        path = os.path.join(logdir, '{}.{}'.format(title, format))
    else:
        raise NotImplementedError

    format = path.split('.')[-1]

    if format == 'json':
        with open(path, "r") as f:
            d = json.load(f)
    elif format == 'pickle':
        with open(path, 'rb') as handle:
            d = pickle.load(handle)
    elif format == 'yaml':
        with open(path, "r") as f:
            d = yaml.safe_load(f)
    else:
        raise NotImplementedError

    return d


def load_config_from_params(logdir, silent_broken_trials=True):
    """Load config from params.json in a logdir.

    Args:
        logdir (str): The logdir, in which we load the config from params.json.
        silent_broken_trials (bool): Whether to be silent when seeing broken trials.

    Returns:
        (Dict) The config.

    """
    path = os.path.join(logdir, "params.json")
    try:
        with open(path, "rt") as f:
            config = json.load(f)
        return config
    except Exception as e:
        if not silent_broken_trials:
            logger.warning(
                f"Cannot load config from paramas.json in {logdir}: \n{e}")
            is_remove = u.inquire_confirm(
                "would you like to remove the folder?")
            if is_remove:
                shutil.rmtree(logdir)
        return None


def summarize_select(df, config_columns, group_summarize, summarize_fn, group_select, select_fn):
    """See https://github.com/YuhangSong/general-energy-nets/tree/master/experiments#summarize-and-select

    Args:
        df (pd.DataFrame): The dataframe.
        config_columns (list): A list of column names specifying config values. It is obtained from analysis_v1.py.
        group_summarize (list): A list of column names specifying groups to summarize over. E.g.: ['seed'].
        summarize_fn (callable): The function for summarize over the specified groups.
            E.g.:
                lambda df: df['test error'].mean()
        group_select (list): A list of column names specifying groups to select over. E.g.: ['learning rate'].
        select_fn (callable): The function for select over the specified groups.
            E.g.:
                lambda df: lambda df: df['summarize'] == df['summarize'].min()

    Returns:
        (pd.DataFrame): The dataframe.
    """

    groups = copy.deepcopy(config_columns)

    [groups.remove(key) for key in group_summarize]
    df = add_metric_per_group(
        df, groups,
        lambda df: (
            'summarize',
            summarize_fn(df)
        )
    )

    [groups.remove(key) for key in group_select]
    df = select_rows_per_group(
        df, groups,
        lambda df: select_fn(df)
    )
    df = drop_cols(df, ['summarize'])

    return df


def select_best_lr(df, config_columns, metric, group_summarize=['seed'], group_select=['learning_rate'], is_min=True):
    """See https://github.com/YuhangSong/general-energy-nets/tree/master/experiments#summarize-and-select

    Args:
        df: See summarize_select.
        config_columns: See summarize_select.
        metric (str): The name of the metric column.
        group_summarize: See summarize_select.
        group_select: See summarize_select.
        is_min (bool): Select the minimum (True) or maximum (False).

    Returns:
        (pd.DataFrame): The dataframe.
    """

    if is_min:
        def select_fn(df): return df['summarize'] == df['summarize'].min()
    else:
        def select_fn(df): return df['summarize'] == df['summarize'].max()

    return summarize_select(
        df, config_columns,
        group_summarize, lambda df: df[metric].mean(),
        group_select, select_fn
    )


def one_row_per_config(df, metric_columns, config_columns, keep_small=True):
    """Drop rows of the same config (specified by config_columns), only leave one row per config in df.

    Note that those metrics (specified by metric_columns) with NaN values will be dropped with priority.
    """

    for metric_column in metric_columns:
        if "-along-" not in metric_column:
            df = df.sort_values(
                by=[metric_column], ascending=keep_small, na_position='last'
            )

    return df.drop_duplicates(
        subset=config_columns
    )


class AnalysisDataFrame():
    """Dataframe with specfications of config_columns and metric_columns.
    """

    def __init__(self, dataframe, config_columns, metric_columns):
        self.dataframe = dataframe
        self.config_columns = config_columns
        self.metric_columns = metric_columns

    def one_row_per_config(self):
        """Drop duplicates according to config_columns.
        """
        self.dataframe = one_row_per_config(
            df=self.dataframe,
            metric_columns=self.metric_columns,
            config_columns=self.config_columns,
        )
        return self

    def rename(self, mapper, is_include_dataframe=True):
        """Rename columns.
        """
        if is_include_dataframe:
            self.dataframe = self.dataframe.rename(
                columns=mapper,
            )
        self.config_columns = u.map_list_by_dict(
            self.config_columns, mapper
        )
        self.metric_columns = u.map_list_by_dict(
            self.metric_columns, mapper
        )
        return self

    def get_dataframe_with_config_metric_columns_only(self):
        """Return a dataframe with only config_columns and metric_columns.
        """
        return self.dataframe[self.config_columns + self.metric_columns]

    def get_dataframe_with_config_columns_only(self):
        """Return a dataframe with only config_columns.
        """
        return self.dataframe[self.config_columns]

    def get_dataframe_with_metric_columns_only(self):
        """Return a dataframe with only metric_columns.
        """
        return self.dataframe[self.metric_columns]


def get_analysis_df(analysis, metrics, is_log_progress=True, silent_broken_trials=True):
    """Get a dataframe for an analysis.

    Note that this function will not drop any trials.

    Args:
        analysis (ray.tune.Analysis): Analysis object to get dataframes from.
        metrics (list of str): Each of which is a evalable str that will produce a metric.
        is_log_progress (bool): Whether to show progress.
        silent_broken_trials (bool): Whether to be silent when seeing broken trials.

    Returns:
        (pd.DataFrame) with each config or metric as a column, each trial as a row.

    """

    # sanitize args
    assert isinstance(analysis, Analysis)
    assert isinstance(metrics, list)
    for m in metrics:
        assert isinstance(m, str)
    assert isinstance(is_log_progress, bool)
    assert isinstance(silent_broken_trials, bool)

    # load config
    # flatten config to flatten_configs
    flatten_configs = {}
    for logdir in analysis.trial_dataframes.keys():
        # load config
        config = load_config_from_params(
            logdir, silent_broken_trials
        )
        if config is not None:
            # flatten config to flatten_configs
            flatten_configs[logdir] = resolve_nested_dict(config)
        else:
            continue

    # init config_columns and metric_columns
    config_columns = []
    metric_columns = []

    # init rows
    rows = []

    # iterate over each trial
    iterator = flatten_configs.items()
    if is_log_progress:
        iterator = tqdm.tqdm(iterator, leave=False)
    for logdir, config in iterator:
        if is_log_progress:
            iterator.set_description('get_analysis_df')

        # get df of the trial
        df = analysis.trial_dataframes[logdir]

        if 'done' in df.columns:

            # Why some done values has Nan (even at iloc[-1])? Anyway, replace these NaN values with False should be safe
            df['done'].fillna(False, inplace=True)

            if not df['done'].iloc[-1].item():

                # the training is not completed, skip incompleted trails
                # it is not a warning message as these incompleted may have been re-runned
                logger.warning(f"Incompleted trail found.")
                continue

        else:

            logger.warning((
                f"The 'done' signal is not in the logged columns, might be an issue with ray version. Please install the correct version required. "
            ))

        # init row
        row = {}
        row['logdir'] = logdir

        # if no metrics specified
        if len(metrics) == 0:

            logger.error(
                f"No metrics specified. You have to generate metrics from \n{df}"
            )

            raise NotImplementedError

        def compress_plot(col, along=None):
            """
            Args:
                col: col that is compressed and later extracted as a plot
                along ('training_iteration'|'time_since_restore'|'test_time'): along which dimension the plot is compressed and extracted

            Example with analysis_v1.py:
                python analysis_v1.py \
                ...
                -m "compress_plot('test__test_split__error','training_iteration')" \
                ...
                -v \
                "df=extract_plot(df,'test__test_split__error','training_iteration')" \
                ...
                "sns.lineplot(data=df,x='training_iteration',y='test__test_split__error',hue='algorithm',errorbar=('ci', 68))"

                WARNING: when using with
                        -m "compress_plot('test__test_split__error','training_iteration')" \
                    you should pass arguments as non-keyword arguments, as when extracting the plot with extract_plot(), it needs to know the name string of the metric as such default format.
            """
            assert isinstance(col, str)
            if along is None:
                raise Exception('Argument <along> need to be specified')
            assert isinstance(along, str)
            compressed_plot = df[[col, along]].values.tolist()
            return '{}-along-{}'.format(col, along), compressed_plot

        # eval metrics
        # add to row as a cell
        for m in metrics:
            assert isinstance(m, str)
            # eval metrics
            try:
                nan = float('nan')
                e_m = eval(m)
            except Exception:
                # it is not a warning message as different structure of code may have been used in the same logdir
                logger.warning((
                    f"Generate metrics \n\t`{m}` \nfailed. This trial will be skipped. "
                ))
                traceback.print_exc()
                continue
            if 'compress_plot' in m:
                # if compress_plot, it returns a tuple
                # where first element is the key and the second element is the compressed plot
                m, e_m = e_m
            # add to row as a cell
            row[m] = e_m
            if m not in metric_columns:
                metric_columns.append(m)

        # eval config
        # add to row and config_columns as a cell
        for k, v in config.items():
            # eval config
            k = str(k)
            if not (isinstance(v, numbers.Number) or isinstance(v, str)):
                raise TypeError(
                    'Config value has to be a number or a string. '
                    'But get {}={} ({})'.format(
                        k, v, type(v)
                    )
                )
            # add to row and config_columns as a cell
            row[k] = v
            if k not in config_columns:
                config_columns.append(k)

        # add row to rows
        rows.append(row)

    if len(rows) == 0:
        logger.warning(
            "You have an empty analysis_df. Maybe because you are looking at an empty logdir or none of the trials has completed yet."
        )

    dataframe = pd.DataFrame.from_dict(rows)

    # construct AnalysisDateFrame
    analysis_df = AnalysisDataFrame(
        dataframe,
        config_columns=copy.deepcopy(config_columns),
        metric_columns=copy.deepcopy(metric_columns),
    )

    return analysis_df


def auto_phrase_config_for_analysis_df(analysis_df):
    """Automatically phrase the configs for an analysis_df to plot-friendly strings."""

    assert isinstance(analysis_df, AnalysisDataFrame)

    rename_dict = {}
    for column in analysis_df.dataframe.columns:
        try:
            new_column = eval(column)
        except:
            continue
        if isinstance(new_column, tuple):
            rename_dict[column] = ': '.join(list(new_column))

    analysis_df = analysis_df.rename(rename_dict)
    return analysis_df


def process_dkeys(d, process):
    """Apply <process()> to each key of a dictionary.

    Args:
        d (dict): The dictionary, the keys of which will be processed by <process()>.
        process (callable): The function to process each key of the dictionary.
    """
    assert isinstance(d, dict)
    assert callable(process)
    processed_d = {}
    for k, v in d.items():
        processed_d[process(k)] = v
    return processed_d


def filter_dataframe_by_dict(dataframe, d):
    """Filter dataframe by a dictionary of keys (columns) and values.

    See https://github.com/YuhangSong/general-energy-nets/tree/master/experiments#select

    Args:
        dataframe (pd.DataFrame): The dataframe to be filtered.
        d (dict): The dictionary to filter the dataframe.
    """
    assert isinstance(dataframe, pd.DataFrame)
    assert isinstance(d, dict)

    # if the keys in d are not in dataframe columns, return None
    try:
        indexes = (dataframe[list(d)] == pd.Series(d))
    except KeyError as e:
        return None

    return dataframe.loc[
        indexes.all(axis=1)
    ]


def drop_same_columns(df):
    """Drop columns with the same values in a dataframe.

    Args:
        df (pd.DataFrame): The dataframe.

    Returns:
        (pd.DataFrame): The dataframe.
    """
    assert isinstance(df, pd.DataFrame)
    df = df.applymap(u.hashify)
    nunique = df.apply(pd.Series.nunique)
    return df.drop(
        nunique[nunique == 1].index,
        axis=1,
    )


# def get_null_rows_columns(df):
#     """Get rows and columns with NaN values.

#     Args:
#         df (pd.DataFrame): The dataframe.

#     Returns:
#         (pd.DataFrame): The dataframe.
#     """
#     assert isinstance(df, pd.DataFrame)
#     return df.loc[:, df.isnull().any()]


"""Deal with analysis_df"""


def purge_analysis_df(analysis_df, config, is_log_progress=True):
    """Purge the analysis_df according to the config.

    Note this function will:
        - [1] index only variants specified in <config>.

    Args:
        analysis_df (AnalysisDateFrame): The analysis_df generated from get_analysis_df().
        config (dict): Purge according to this config.
        is_log_progress (bool): Whether to show progress.

    Returns:
        (AnalysisDateFrame) Purged analysis_df.
    """

    # sanitize args
    assert isinstance(analysis_df, AnalysisDataFrame)
    assert isinstance(config, dict)
    assert isinstance(is_log_progress, bool)

    # init rows
    rows = []
    # missing_trials will be a table with each row being a missing_trial
    missing_trials = None

    # refer to [1]
    if is_log_progress:
        pbar = tqdm.tqdm(
            total=len(list(generate_variants(config))),
            leave=False,
        )
    for config_variant in generate_variants(config):
        if is_log_progress:
            pbar.set_description('purge_analysis_df')

        # get variant
        filter = process_dkeys(
            resolve_nested_dict(
                config_variant[1]
            ),
            process=lambda k: str(k),
        )

        # index row
        row = filter_dataframe_by_dict(analysis_df.dataframe, filter)

        # check and warning
        if (row is not None) and (len(row.index) > 0):

            if len(row.index) > 1:
                logger.warning((
                    f"You have more than one trials on a config. "
                    f"If you have called analysis_df.one_row_per_config() before, it means the metric columns of these two trials are different. "
                    f"It further means your experiment cannot be reproduced with the same config yaml. "
                    f"Anyway, I will include all of them and proceed. "
                ))

            rows.append(row)
            analysis_df.dataframe.drop(row.index, inplace=True)

        else:

            logger.warning((f"You have zero trial on a config."))

            # a missing_trial will be a row with columns being the config keys and entries being the config values
            missing_trial = pd.DataFrame(
                [filter.values()],
                columns=filter.keys(),
            )
            if missing_trials is None:
                missing_trials = missing_trial
            else:
                # missing_trial is appended to missing_trials as a new row
                missing_trials = pd.concat(
                    [missing_trials, missing_trial],
                    ignore_index=True,
                )

        pbar.update(1)

    # check and warning if no rows were found
    if len(rows) == 0:
        logger.warning("You have indexed in total zero trials.")

    # check and warning missing trials
    if missing_trials is not None:
        # unique_missing_trials is dict with keys being the config keys and values being the missing values of this config
        unique_missing_trials = {}
        for col in missing_trials:
            unique_missing_trials[col] = missing_trials[col].unique().tolist()
        logger.warning(
            f"You have missing trials on \n{pp.pformat(unique_missing_trials)}")

    analysis_df.dataframe = pd.concat(rows)

    # drop the config_columns where all elements are NaN
    # this is necessary because some configs are not in some yamls, but they all got logged together
    for config_column in copy.deepcopy(analysis_df.config_columns):
        if analysis_df.dataframe[config_column].isnull().all():
            analysis_df.config_columns.remove(config_column)
            analysis_df.dataframe = analysis_df.dataframe.drop(
                [config_column], axis=1
            )

    return analysis_df


def purge_analysis_df_with_config_dicts(analysis_df, config_dicts, rename_dict={}):

    purged_analysis_dfs = []

    for config_dict in config_dicts:

        purged_analysis_df = purge_analysis_df(
            copy.deepcopy(analysis_df),
            config=config_dict,
        )

        # rename for each config_dict, as they could use the same config after renaming
        purged_analysis_df = purged_analysis_df.rename(rename_dict)

        purged_analysis_dfs.append(
            purged_analysis_df
        )

    # dataframe is obtained from concating purged_analysis_dfs, other properties are retained.
    analysis_df.dataframe = pd.concat(
        [purged_analysis_df.dataframe for purged_analysis_df in purged_analysis_dfs]
    )

    analysis_df = analysis_df.rename(
        rename_dict,
        # because dataframe has just been updated the line above
        is_include_dataframe=False,
    )

    # some columns are full of NaN values, drop them
    analysis_df.dataframe = analysis_df.dataframe.dropna(
        axis="columns", how="all"
    )
    columns = analysis_df.dataframe.columns.values.tolist()
    analysis_df.config_columns = u.list_intersection(
        analysis_df.config_columns, columns
    )
    analysis_df.metric_columns = u.list_intersection(
        analysis_df.metric_columns, columns
    )

    return analysis_df


def explode_with(df, new_col_name, x):
    """Explode with x.

    Args:
        new_col_name: name of the new column.
        x:
            if (list): element of which to explode.
    """

    assert isinstance(df, pd.DataFrame)
    assert isinstance(new_col_name, str)

    if isinstance(x, list):

        df = new_col(
            df, '{}-along-indexes'.format(new_col_name),
            lambda row: torch.stack([
                torch.Tensor(x),
                torch.Tensor(list(range(len(x)))),
            ]).t().tolist(),
        )
        df = extract_plot(df, new_col_name, 'indexes')
        df = df.drop(['indexes'], axis=1)

    else:

        raise NotImplementedError

    return df


def drop_cols(df, cols):
    """See https://github.com/YuhangSong/general-energy-nets/tree/master/experiments#drop-cols
    """
    assert isinstance(df, pd.DataFrame)
    return df.drop(cols, axis=1)


def reduce(df, by, reduce_fn):
    """See https://github.com/YuhangSong/general-energy-nets/tree/master/experiments#reduce-by-group
    """

    # sanitize args
    assert isinstance(df, pd.DataFrame)
    assert isinstance(by, list)
    assert len(by) > 0
    for col in by:
        assert isinstance(col, str)
    assert callable(reduce_fn)

    # sort by by
    df = df.sort_values(by)
    # group by by
    df = df.groupby(by)
    # apply reduce_fn
    df = df.apply(lambda df: pd.Series(reduce_fn(df)))
    # reset MultiIndex to (normal) Index
    df = df.reset_index()

    return df


def add_metric_per_group(df, by, metric_fn):
    """See https://github.com/YuhangSong/general-energy-nets/tree/master/experiments#add-metric-per-group
    """

    # sanitize args
    assert isinstance(df, pd.DataFrame)
    assert isinstance(by, list)
    for col in by:
        assert isinstance(col, str)
    assert callable(metric_fn)

    return df.groupby(
        by, group_keys=False
    ).apply(
        lambda df: df.assign(
            **{
                metric_fn(df)[0]: df.apply(
                    lambda row: metric_fn(df)[1],
                    axis=1,
                )
            }
        )
    )


def select_rows_per_group(df, by, select_fn):
    """See https://github.com/YuhangSong/general-energy-nets/tree/master/experiments#reduce-by-group
    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(by, list)
    for col in by:
        assert isinstance(col, str)
    assert callable(select_fn)

    return df.groupby(
        by, group_keys=False
    ).apply(
        lambda df: df.loc[select_fn(df)]
    )


def new_col(df, col_key, apply_fn):
    """See https://github.com/YuhangSong/general-energy-nets/tree/master/experiments#generate-new-column-from-each-row
    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(col_key, str)
    assert callable(apply_fn)

    df.insert(len(df.columns), col_key, df.apply(
        lambda row: apply_fn(row), axis=1))

    return df


def combine_cols(df, col_key, cols, combine_opt='dict'):
    """See https://github.com/YuhangSong/general-energy-nets/tree/master/experiments#combine-cols
    """

    def combine_fn(row):

        if combine_opt == 'dict':
            return dict2str(
                row[cols].to_dict()
            )

        elif combine_opt == 'values':
            return list2str(
                list(
                    row[cols].to_dict().values()
                )
            )

        else:
            raise NotImplementedError

    return new_col(df, col_key, combine_fn)


def extract_plot(df, col, along):
    """See https://github.com/YuhangSong/general-energy-nets/tree/master/experiments#plot-along-training-iteration
    """

    assert isinstance(df, pd.DataFrame)
    assert isinstance(col, str)
    assert isinstance(along, str)

    compressed_col_key = '{}-along-{}'.format(col, along)

    df = df.explode(compressed_col_key)
    df[
        [
            col, along
        ]
    ] = pd.DataFrame(
        df[compressed_col_key].tolist(),
        index=df.index,
    )

    df = df.drop([compressed_col_key], axis=1)

    df = df.reset_index(drop=True)

    return df


# def codify_col(df, col):
#     """See https://github.com/YuhangSong/general-energy-nets/tree/master/experiments#codify-a-column
#     """
#     df.insert(
#         1, '{}-code'.format(col),
#         df.apply(
#             lambda row: '```{}```'.format(row[col]), axis=1
#         )
#     )
#     return df


"""Draw and output"""


def reduce_metric(df, config_cols, metric_col, reduce_with=['mean']):
    """Reduce metric_col by config_cols according to reduce_with.

    The relationship of df.columns and other cols should be:
        df.columns:
            config_cols
            metric_col
            ...

    Args:
        df (pd.DataFrame): The DataFrame to reduce.
        config_cols (list): The columns to group by.
        metric_col (str): The column to reduce.
        reduce_with (list): The reduce methods to use.
    """

    assert isinstance(config_cols, list)
    assert isinstance(metric_col, str)
    assert isinstance(reduce_with, list)

    for config_col in config_cols:
        assert isinstance(config_col, str), f'{config_col} is not str'
        assert config_col in df.columns, f'{config_col} not in df.columns {df.columns}'

    assert metric_col in df.columns, f'{metric_col} not in df.columns {df.columns}'
    assert metric_col not in config_cols, f'{metric_col} in config_cols {config_cols}'

    def reduce_fn(df):
        reduced = {}
        for reduce_method in reduce_with:
            reduced[
                f'{metric_col} ({reduce_method})'
            ] = eval(
                f'df[metric_col].{reduce_method}()'
            )
        return reduced

    return reduce(
        df, config_cols,
        lambda df: reduce_fn(df),
    )


def px_lines(df, config_cols, metric_col, progress_col='training_iteration', reduce_col='seed', facet_col=None, facet_row=None, is_with_sem=True):
    """Draw lines with plotly.express

    The relationship of df.columns and other cols should be:

        df.columns:
            config_cols:
                facet_col
                facet_row
                ...
            metric_col
            progress_col
            reduce_col
            ...

    Args:
        df (pd.DataFrame): input dataframe
        config_cols (list): list of config columns
        metric_col (str): metric column
        progress_col (str, optional): progress column. Defaults to 'training_iteration'.
        reduce_col (str, optional): reduce column. Defaults to 'seed'.
        facet_col (str, optional): facet column. Defaults to None.
        facet_row (str, optional): facet row. Defaults to None.
        is_with_sem (bool, optional): whether to draw error bar with sem. Defaults to True.
    """

    assert isinstance(df, pd.DataFrame)
    assert isinstance(config_cols, list)
    assert isinstance(progress_col, str)
    assert isinstance(reduce_col, str)
    assert isinstance(facet_col, str) or facet_col is None
    assert isinstance(facet_row, str) or facet_row is None

    for config_col in config_cols:
        assert config_col in df.columns, 'config_col {} not in df.columns'.format(
            config_col
        )

    assert progress_col in df.columns, 'progress_col {} not in df.columns'.format(
        progress_col
    )
    assert reduce_col in df.columns, 'reduce_col {} not in df.columns'.format(
        reduce_col
    )
    if facet_col is not None:
        assert facet_col in df.columns, 'facet_col {} not in df.columns'.format(
            facet_col
        )
    if facet_row is not None:
        assert facet_row in df.columns, 'facet_row {} not in df.columns'.format(
            facet_row
        )

    assert progress_col not in config_cols, 'progress_col {} in config_cols'.format(
        progress_col
    )
    assert reduce_col not in config_cols, 'reduce_col {} in config_cols'.format(
        reduce_col
    )

    if facet_col is not None:
        assert facet_col in config_cols, 'facet_col {} not in config_cols'.format(
            facet_col
        )
    if facet_row is not None:
        assert facet_row in config_cols, 'facet_row {} not in config_cols'.format(
            facet_row
        )

    reduce_with = ['mean']
    if is_with_sem:
        reduce_with.append('sem')
    df = reduce_metric(
        df,
        config_cols=config_cols+[progress_col],
        metric_col=metric_col,
        reduce_with=reduce_with,
    )

    id_cols = copy.deepcopy(config_cols)
    if facet_col is not None:
        id_cols.remove(facet_col)
    if facet_row is not None:
        id_cols.remove(facet_row)

    df = combine_cols(df, 'id', id_cols)

    fig = px.line(
        df,
        x=progress_col,
        y=f'{metric_col} (mean)',
        error_y=f'{metric_col} (sem)' if is_with_sem else None,
        color='id',
        facet_col=facet_col, facet_row=facet_row,
    )

    fig.show()

    return fig


def get_non_numeric_cols(df):
    """Get non-numeric columns of a dataframe
    """
    assert isinstance(df, pd.DataFrame)

    non_numeric_cols = []
    for col in df.columns:

        # Check if the column contains only numeric data
        if not all((isinstance(i, (int, float)) and (not isinstance(i, bool))) for i in df[col]):
            non_numeric_cols.append(col)

    return non_numeric_cols


def parallel_coordinates(
    df,
    cols=None,
    categorical_cols=[],
    color_col=None,
    colorscale='Viridis',
    unselected_line_kwargs={
        'color': 'red', 'opacity': 0.0,
    },
):
    """
    Draw parallel coordinates plot.

    The relationship of df.columns and other cols should be:

        df.columns:
            cols:
                categorical_cols
                ...
            ...

    :param df: the dataframe to plot
    :param cols: cols to plot
    :param categorical_cols: a list of cols that are categorical (categorical columns are automatically detected, this adds to the list of automatically detected categorical columns)
    :param color_col: the column to color by
    :param colorscale: the colorscale to use
    :param unselected_line_kwargs: kwargs for unselected lines
    """

    # sanitize args
    assert isinstance(df, pd.DataFrame)
    assert isinstance(cols, list) or cols is None
    assert isinstance(categorical_cols, list)
    assert isinstance(color_col, str) or color_col is None
    assert isinstance(unselected_line_kwargs, dict)

    # get cols
    cols = list(df.keys()) if cols is None else cols

    for col in categorical_cols:
        assert col in cols, 'categorical column {} not in cols {}'.format(
            col, cols
        )

    audo_detected_categorical_cols = get_non_numeric_cols(df)

    dimensions = []

    for col in cols:

        # NEED REVIEW >>
        # # copied code, I don't understand it

        if col in list(set(audo_detected_categorical_cols+categorical_cols)):  # categorical columns

            values = df[col].unique()
            # works if values are strings, otherwise we probably need to convert them
            value2dummy = dict(zip(values, range(len(values))))
            df[col] = [value2dummy[v] for v in df[col]]
            dimension = dict(
                label=col,
                tickvals=list(value2dummy.values()),
                ticktext=list(value2dummy.keys()),
                values=df[col],
            )

        else:  # continuous columns

            dimension = dict(
                range=(df[col].min(), df[col].max()),
                label=col,
                values=df[col],
            )

        # >> NEED REVIEW

        dimensions.append(dimension)

    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=df[color_col] if color_col is not None else 'blue',
            colorscale=colorscale,
        ),
        dimensions=dimensions,
        unselected=dict(line=unselected_line_kwargs)
    ))

    fig.show()

    return fig


def my_catplot(*args, **kwargs):
    """See https://github.com/YuhangSong/general-energy-nets/tree/master/experiments#my_catplot
    """
    return sns.catplot(*args, **kwargs, errorbar=('ci', 68), kind='point', capsize=0.4, legend_out=False)


def my_relplot(*args, **kwargs):
    """See https://github.com/YuhangSong/general-energy-nets/tree/master/experiments#my_relplot
    """
    return sns.relplot(*args, **kwargs, errorbar=('ci', 68), kind='line')


def df2tb(df):
    """See https://github.com/YuhangSong/general-energy-nets/tree/master/experiments#save-the-dataframe-to-markdown-file
    """
    return tabulate(df.values, df.columns, tablefmt='pipe')


"""Other"""


def dict2str(d):
    """"""
    assert isinstance(d, dict)
    return str(d)[1:-1].replace('\'', '\"').replace('\"', '').replace(': ', '=')


def list2str(l):
    """"""
    assert isinstance(l, list)
    return str(l)[1:-1].replace('\'', '\"').replace('\"', '')


# def count_parameters(model):
#     """See https://github.com/YuhangSong/general-energy-nets/tree/master/experiments#deal-with-saved-model
#     """
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)


# def get_checkpoint_path(log_dir, i, model_name='model.pth'):
#     """See https://github.com/YuhangSong/general-energy-nets/tree/master/experiments#get-checkpoint-path
#     """
#     assert isinstance(log_dir, str)
#     assert isinstance(i, int)
#     assert i >= 0
#     assert isinstance(model_name, str)

#     return os.path.join(log_dir, 'checkpoint_{}'.format(i), model_name)


# def get_model_delta(model_i, model_j, delta_fn):
#     """See https://github.com/YuhangSong/general-energy-nets/tree/master/experiments#compare-two-checkpoints
#     """
#     delta_all = []
#     if isinstance(model_i, dict):
#         for key, _ in model_i.items():
#             delta_all.append(
#                 delta_fn(
#                     key,
#                     model_i[key],
#                     model_j[key]
#                 )
#             )
#     elif isinstance(model_i, torch.nn.module):
#         raise NotImplementedError
#     else:
#         raise NotImplementedError
#     return sum(delta_all)


"""The following is legacy code."""


# def setup_plt(goal='produce_paper_figs', figsize_ratio=2.0, is_produce_example=False):
#     """Set up plt.

#     Args:
#         goal (str): One of the following goals:
#             Example:
#                 goal='default': default setup of plt.
#                 goal='produce_paper_figs': produce figures for a paper.
#         figsize_ratio (float): The ratio of the figure comparing to the default figsize.
#         is_produce_example (bool): Whether to produce a example figure to the current dictionary.

#     Returns:
#         fig_configs (dict): A dict of configs can be used for figure.
#     """

#     assert isinstance(goal, str)
#     assert isinstance(is_produce_example, bool)
#     assert isinstance(figsize_ratio, float)

#     logger.info('Seting up plt for {}'.format(goal))

#     sns.set()

#     # setup plt
#     fig_configs = {}
#     if goal in ['default']:
#         pass

#     elif goal in ['produce_paper_figs']:
#         matplotlib.rcParams.update({
#             'font.family': 'serif',
#             'text.usetex': True,
#         })
#         fig_configs['fill_between: alpha'] = 0.3

#     else:
#         raise NotImplementedError

#     # setup figure size
#     matplotlib.rcParams.update({
#         "figure.figsize": (np.array([6.4, 4.8]) * figsize_ratio).tolist(),
#     })

#     # produce a example figure to the current dictionary
#     if is_produce_example:

#         fig = plt.figure('fig-example')
#         ax = plt.gca()
#         len = 64
#         x = np.linspace(1., 8., len)
#         y1 = np.cos(x)
#         y2 = np.sin(x)
#         std = np.abs(np.random.normal(0.0, 1.0, len))
#         sns.lineplot(x, y1, color='blue', label=r'$\gamma$')
#         sns.lineplot(x, y2, color='red',
#                      label='$l_{{\\textrm{{max}}}}={}$'.format(2))
#         ax.fill_between(x, y1 + std, y1 - std,
#                         facecolor='blue', alpha=fig_configs['fill_between: alpha'])
#         ax.fill_between(x, y2 + std, y2 - std,
#                         facecolor='red', alpha=fig_configs['fill_between: alpha'])
#         save_fig('./', 'example-fig')

#     return fig_configs


# def down_sample_df(df, ts):
#     """Downsample a dataframe by ts and returns a numpy array.
#     """
#     x = df.to_numpy()
#     x = np.take(x, ts)
#     return x


# def get_energy(x):
#     """Get energy
#     """
#     return 0.5 * (x**2)


# def get_energy_results(df, ts, num_final=1, use_checkpoint=False, is_compute_efficiency=True):
#     """Get energy results. Deprecated.
#     """
#     results = {}

#     results['curr_t'] = ts
#     results['t'] = np.diff(results['curr_t'])

#     if 'test/error' in df.columns:
#         results['test_error'] = down_sample_df(df['test/error'], ts)
#         results['d_test_error'] = np.diff(results['test_error'])
#     else:
#         if is_compute_efficiency:
#             raise Exception(
#                 'ERROR: to compute efficiency, you need test_error')

#     # load weights
#     results['weights'] = []
#     checkpoint_paths = down_sample_df(df['checkpoint_path'], ts)
#     for checkpoint_path in checkpoint_paths:
#         results['weights'].append(
#             torch.load(
#                 checkpoint_path,
#                 map_location=torch.device('cpu'),
#             )
#         )

#     results['energy'] = []
#     for i in range(len(results['weights']) - 1):
#         results['energy'].append(
#             0.0
#         )
#         for weight_key in results['weights'][i].keys():
#             if ('weight' in weight_key) or ('bias' in weight_key):
#                 results['energy'][-1] += (
#                     get_energy(results['weights'][i + 1][weight_key]) -
#                     get_energy(results['weights'][i][weight_key])
#                 ).abs().sum()
#     results['energy'] = np.array(results['energy'])
#     results['curr_energy'] = np.cumsum(results['energy'])

#     results['traj'] = []
#     for i in range(len(results['weights']) - 1):
#         results['traj'].append(
#             0.0
#         )
#         for weight_key in results['weights'][i].keys():
#             if ('weight' in weight_key) or ('bias' in weight_key):
#                 results['traj'][-1] += (
#                     results['weights'][i + 1][weight_key] -
#                     results['weights'][i][weight_key]
#                 ).pow(2).sum()
#         results['traj'][-1] = results['traj'][-1]**0.5
#     results['traj'] = np.array(results['traj'])
#     results['curr_traj'] = np.cumsum(results['traj'])

#     def compute_efficiency(key):
#         if 0 in results[key]:
#             logger.warning('{} is zero somewhere \n{}, \n{}_efficiency is not computed'.format(
#                 key,
#                 results[key],
#                 key,
#             ))
#         else:
#             results['{}_efficiency'.format(
#                 key
#             )] = np.divide(
#                 results['d_test_error'],
#                 results[key],
#             )

#     if is_compute_efficiency:
#         for key in ['t', 'energy', 'traj']:
#             compute_efficiency(key)

#     return results


# class Formater():
#     """Formater.
#     """

#     def __init__(self, target='tex', is_scientific=True, precision=2):
#         """Initializer.

#         Args:
#             target (str): One of the following target of formatting:
#                 tex:
#             is_scientific (bool): If using scientific notation.
#             precision (int): Precision of float formatting.
#         """
#         self.target = target
#         self.is_scientific = is_scientific
#         self.precision = precision
#         self.scalar_formatter = mticker.ScalarFormatter(
#             useOffset=False, useMathText=True)

#     def format(self, value):
#         """Format the value.
#         """
#         if isinstance(value, numbers.Number):
#             if self.is_scientific:
#                 value = self.scalar_formatter._formatSciNotation(
#                     '{:.{precision}e}'.format(
#                         value,
#                         precision=self.precision,
#                     )
#                 )
#             if self.target == 'tex':
#                 value = r"${}$".format(
#                     value
#                 )
#         return value


# def evalute_field(record, field_spec):
#     """Evalute a field of a record using the type of the field_spec as a guide.
#     """
#     if type(field_spec) is int:
#         return str(record[field_spec])
#     elif type(field_spec) is str:
#         return str(getattr(record, field_spec))
#     else:
#         return str(field_spec(record))


# def save_table(
#     headers,
#     table,
#     logdir,
#     title,
#     formats=['md', 'tex'],
#     formater=Formater()
# ):
#     """Save tables.

#     Args:
#         formats (list): List of the following formats: md, tex.
#         logdir (str): Log directory of the table.
#         title (str): Title of the table.
#     """
#     logger.info('save table {} to {} in formats {}'.format(
#         title, logdir, formats))
#     u.prepare_dir(logdir)

#     for format in formats:
#         file = open(
#             os.path.join(
#                 logdir,
#                 '{}.{}'.format(
#                     title,
#                     format,
#                 )
#             ),
#             "w"
#         )
#         generate_table(
#             headers,
#             table,
#             file=file,
#             formats=[format],
#             formater=formater,
#         )
#         file.close()


# def generate_table(
#     headers,
#     table,
#     file=sys.stdout,
#     formats=['md', 'tex'],
#     formater=Formater()
# ):
#     """Generate table and write to file.

#     Args:
#         headers (list): List of column headers.
#         table (neasted 2-D list): Rows will be generated from this.
#         file (object): Any object with a 'write' method that takes a single string
#             parameter. By default, it is sys.stdout.
#         formats (list): List of the following formats: md, tex.
#         formater (Formater): Formater used for format the cell values.
#     """
#     # if 'md' in formats:
#     #     generate_table_md(
#     #         headers, table,
#     #         file=file,
#     #         formater=formater,
#     #     )
#     # if 'tex' in formats:
#     #     generate_table_tex(
#     #         headers, table,
#     #         file=file,
#     #         formater=formater,
#     #     )
#     for format in formats:
#         for row in range(len(table)):
#             for column in range(len(table[row])):
#                 table[row][column] = formater.format(
#                     table[row][column]
#                 )
#         for column in range(len(headers)):
#             headers[column] = formater.format(
#                 headers[column]
#             )
#         file.write('\n')
#         file.write(
#             tabulate(
#                 table,
#                 headers,
#                 tablefmt={
#                     "tex": "latex_raw",
#                     "md": "github",
#                 }[format],
#                 numalign='center',
#                 stralign='center',
#             )
#         )
#         file.write('\n')


# def get_divergence_test_error(df_bp, df_pcn, ord=1):
#     """Get divergence of test error. Outdated.
#     """
#     def process(x):
#         x = x.to_numpy()
#         return x
#     return LA.norm(
#         process(df_bp['test/error']) - process(df_pcn['test/error']),
#         ord=ord,
#         axis=0,
#     )


# def get_divergence_final_weights(df_bp, df_pcn, ord=2):
#     """Get divergence of final weights. Outdated.
#     """
#     divergence_final_weights = []
#     for col in df_bp.columns:
#         if 'train/log_p/base:zero/value/' in col:
#             def process(x):
#                 x = np.squeeze(
#                     np.expand_dims(
#                         x.dropna()[-1:].to_numpy(),
#                         1,
#                     ),
#                     0,
#                 )
#                 return x
#             tmp = LA.norm(
#                 process(df_bp[col]) - process(df_pcn[col]),
#                 ord=ord,
#                 axis=0,
#             )
#             divergence_final_weights.append(tmp)
#     return np.sum(
#         np.array(divergence_final_weights)**ord
#     )**(1.0 / ord)


# def show_values_on_bars(axs, fontsize=12):
#     """Show values on bars.
#     """
#     def _show_on_single_plot(ax):
#         for p in ax.patches:
#             _x = p.get_x() + p.get_width() / 2
#             _y = p.get_y() + p.get_height() + 0.02
#             value = '{:.2f}'.format(p.get_height())
#             ax.text(_x, _y, value, ha="center", fontsize=fontsize)

#     if isinstance(axs, np.ndarray):
#         for idx, ax in np.ndenumerate(axs):
#             _show_on_single_plot(ax)
#     else:
#         _show_on_single_plot(axs)


# def get_dataframes(to_match_configs, analysis, is_reducing_to_one_dataframe=True, silent_broken_trials=True, is_rm_matched_trials=False, is_log_progress=False):
#     """Get dataframes from analysis with specific configs.

#     Args:
#         analysis (ray.tune.Analysis): Analysis object to get dataframes from.
#         to_match_configs (dict): Configs to specify the dataframes to get.
#         is_reducing_to_one_dataframe (bool): Whether return just one dataframe.
#             If True,
#                 If multiple matched dataframes are found, only one of them is returned and a
#                 warning will be raised.
#                 If just one matched dataframes are found, this one is returned.
#             If False,
#                 A dict with all matched dataframes will be returned
#         silent_broken_trials (bool): Whether to be silent when seeing broken trials.
#         is_rm_matched_trials (bool): Whether remove the matched trials.
#             This is used to clean your unwanted trials.

#     Returns:
#         dict of matched dataframes or just one dataframe.

#     """
#     dataframes = analysis.trial_dataframes

#     if not hasattr(analysis, 'flatten_configs'):
#         # logger.info('loading configs of trials.')
#         analysis.flatten_configs = {}
#         for logdir in dataframes.keys():
#             loaded_config = load_config_from_params(
#                 logdir, silent_broken_trials)
#             if loaded_config is not None:
#                 analysis.flatten_configs[logdir] = flatten_dict(loaded_config)
#             else:
#                 continue

#     in_dataframes_configs = analysis.flatten_configs

#     most_matched_infos = {
#         'num_matched_configs': 0
#     }
#     matched_dataframes = {}

#     iterator = in_dataframes_configs.items()
#     if is_log_progress:
#         iterator = tqdm.tqdm(iterator, leave=False)
#     for logdir, in_dataframe_configs in iterator:
#         if is_log_progress:
#             iterator.set_description('get_dataframes')

#         if 'training_iteration' in in_dataframe_configs.keys():
#             logger.warning(
#                 'Having training_iteration in config is deprecated.')
#             training_iteration = in_dataframe_configs['training_iteration']
#         elif 'run_kwargs: stop: training_iteration' in in_dataframe_configs.keys():
#             training_iteration = in_dataframe_configs['run_kwargs: stop: training_iteration']
#         else:
#             raise NotImplementedError

#         if dataframes[logdir].shape[0] != training_iteration:
#             if not silent_broken_trials:
#                 logger.warning('incompleted date found at: \n{} \ndataframe lenth: {} \ntraining_iteration: {} \n'.format(
#                     logdir,
#                     dataframes[logdir].shape[0],
#                     training_iteration,
#                 ))
#                 is_remove = u.inquire_confirm(
#                     "Would you like to remove the trial with incompleted data?"
#                 )
#                 if is_remove:
#                     shutil.rmtree(logdir)
#             continue

#         num_matched_configs = 0
#         mismatched_configs = []

#         for to_match_config_key, to_match_config in to_match_configs.items():

#             def check_simplied_config_key(to_match_config_key, in_dataframe_config_keys):
#                 num_simplied_to_match_config_key = 0
#                 for in_dataframe_config_key in in_dataframe_config_keys:
#                     if len(in_dataframe_config_key.split(': ')) > 1 and to_match_config_key == in_dataframe_config_key.split(': ')[-1]:
#                         num_simplied_to_match_config_key += 1
#                 if num_simplied_to_match_config_key > 1:
#                     raise Exception(
#                         'num_simplied_to_match_config_key should be at most 1, you have more than one because simplified config key is ambiguous.'
#                     )

#             check_simplied_config_key(
#                 to_match_config_key, in_dataframe_configs.keys())

#             def is_config_match(to_match_config_key, to_match_config, in_dataframe_configs):
#                 for in_dataframe_config_key in in_dataframe_configs.keys():
#                     if to_match_config_key in [in_dataframe_config_key, in_dataframe_config_key.split(': ')[-1]]:
#                         in_dataframe_config = {
#                             in_dataframe_config_key: in_dataframe_configs[in_dataframe_config_key]
#                         }
#                         if in_dataframe_configs[in_dataframe_config_key] == to_match_config:
#                             return True, in_dataframe_config
#                         else:
#                             return False, in_dataframe_config
#                 return False, 'to_match_config_key does not exsit in in_dataframe_config_key'

#             is_match, in_dataframe_config = is_config_match(
#                 to_match_config_key, to_match_config, in_dataframe_configs)

#             if is_match:
#                 num_matched_configs += 1

#             else:
#                 mismatched_configs.append({
#                     'to_match_config':  {to_match_config_key: to_match_config},
#                     'in_dataframe_config': in_dataframe_config,
#                 })

#         if num_matched_configs > most_matched_infos['num_matched_configs']:
#             most_matched_infos['num_matched_configs'] = num_matched_configs
#             most_matched_infos['in_dataframe_configs'] = in_dataframe_configs
#             most_matched_infos['mismatched_configs'] = copy.deepcopy(
#                 mismatched_configs
#             )

#         if num_matched_configs == len(to_match_configs.keys()):

#             if is_rm_matched_trials:
#                 is_remove = u.inquire_confirm(
#                     "Would you like to remove the matched trial?"
#                 )
#                 if is_remove:
#                     shutil.rmtree(logdir)
#                     continue

#             matched_dataframes[logdir] = dataframes[logdir].copy()

#             def add_checkpoint_path_to_dataframe():
#                 matched_dataframes[logdir]["checkpoint_path"] = 'NaN'
#                 num_checkpoints = int(
#                     training_iteration /
#                     in_dataframe_configs['checkpoint_freq']
#                 )
#                 # initial checkpoint
#                 matched_dataframes[logdir].at[0, "checkpoint_path"] = "{}/checkpoint_{}/model.pth".format(
#                     logdir,
#                     0,
#                 )
#                 # later checkpoints
#                 for checkpoint_i in range(num_checkpoints):
#                     checkpoint_train_iteration = int(
#                         (checkpoint_i + 1) *
#                         in_dataframe_configs['checkpoint_freq']
#                     )
#                     matched_dataframes[logdir].at[checkpoint_train_iteration, "checkpoint_path"] = "{}/checkpoint_{}/model.pth".format(
#                         logdir,
#                         checkpoint_train_iteration,
#                     )

#             if 'checkpoint_freq' in in_dataframe_configs.keys():
#                 if in_dataframe_configs['checkpoint_freq'] > 0:
#                     add_checkpoint_path_to_dataframe()

#     if len(matched_dataframes) == 0:
#         logger.warning('cannot find data for to_match_configs \n{} \nMost matched in_dataframe_configs are \n{} \nwhich matches {} out of {} configs. \nMismatched configs are \n{}'.format(
#             pp.pformat(to_match_configs),
#             pp.pformat(most_matched_infos['in_dataframe_configs']),
#             most_matched_infos['num_matched_configs'],
#             len(to_match_configs.keys()),
#             pp.pformat(most_matched_infos['mismatched_configs']),
#         ))
#     else:
#         logger.info('Successfully load dataframes for configs: \n{}'.format(
#             pp.pformat(to_match_configs)
#         ))

#     if is_reducing_to_one_dataframe:
#         if len(matched_dataframes) > 1:
#             logger.warning('Multiple matched dataframes found for to_match_configs: \n{} \nOnly one of them is returned. Consider deleting some of the folders or use more specific configs.'.format(
#                 pp.pformat(to_match_configs)
#             ))
#         if len(matched_dataframes) == 0:
#             raise Exception('No matched_dataframes')
#         else:
#             return matched_dataframes[
#                 list(matched_dataframes.keys())[0]
#             ]

#     else:
#         return matched_dataframes


# def get_results_to_compare(analysis, configs_to_compare, configs_to_reduce, way_to_reduce, metric, configs={}, is_log_progress=True):
#     """Get results to compare from dicts of configs_to_compare and configs_to_reduce.

#     Args:
#         analysis (ray.tune.Analysis): Analysis object to get results from.
#         configs_to_compare (dict): Configs to compare across.
#         configs (dict): Specified configs.
#         way_to_reduce (list): List of ways to reduce, each of which is a str that can be eval().

#     Returns:
#         dict of results to compare.

#     """
#     results_to_compare = {}
#     configs_to_compare = copy.deepcopy(configs_to_compare)
#     configs_to_reduce = copy.deepcopy(configs_to_reduce)
#     key_this_iter = copy.deepcopy(
#         list(configs_to_compare.keys())[0]
#     )
#     configs_to_compare_this_iter = copy.deepcopy(
#         configs_to_compare[key_this_iter]
#     )
#     configs_to_compare.pop(key_this_iter)

#     iterator = configs_to_compare_this_iter
#     if is_log_progress:
#         iterator = tqdm.tqdm(iterator, leave=False)
#     for config_to_compare_ in iterator:
#         if is_log_progress:
#             iterator.set_description('COMPARING [{}]'.format(key_this_iter))

#         config_to_compare_str_ = config_to_compare_
#         if key_this_iter in float_configs:
#             if isinstance(config_to_compare_, float):
#                 config_to_compare_str_ = float2item[config_to_compare_]
#         results_to_compare_key = '{}={}'.format(
#             key_this_iter, config_to_compare_str_)
#         configs[key_this_iter] = config_to_compare_
#         if len(configs_to_compare.keys()) == 0:
#             x = get_results_to_reduce(
#                 analysis=analysis,
#                 configs_to_reduce=configs_to_reduce,
#                 metric=metric,
#                 configs=configs,
#             )

#             results_to_compare[results_to_compare_key] = eval(way_to_reduce)

#         else:
#             results_to_compare[results_to_compare_key] = get_results_to_compare(
#                 analysis=analysis,
#                 configs_to_compare=configs_to_compare,
#                 configs_to_reduce=configs_to_reduce,
#                 way_to_reduce=way_to_reduce,
#                 metric=metric,
#                 configs=configs,
#             )
#     return results_to_compare


# def get_results_to_reduce(analysis, configs_to_reduce, metric, configs={}, is_log_progress=False):
#     """Get results to average from a dict of configs_to_reduce.

#     Args:
#         analysis (ray.tune.Analysis): Analysis object to get results from.
#         configs_to_reduce (dict): Configs to average across.
#         metric (list): List of metric, each of which is a str.
#         configs (dict): Specified configs.

#     Returns:
#         List of results to average.

#     """
#     results_to_reduce = []
#     configs_to_reduce = copy.deepcopy(configs_to_reduce)
#     key_this_iter = copy.deepcopy(
#         list(configs_to_reduce.keys())[0]
#     )
#     configs_to_reduce_this_iter = copy.deepcopy(
#         configs_to_reduce[key_this_iter]
#     )
#     configs_to_reduce.pop(key_this_iter)

#     iterator = configs_to_reduce_this_iter
#     if is_log_progress:
#         iterator = tqdm.tqdm(iterator, leave=False)
#     for config_to_reduce_ in configs_to_reduce_this_iter:
#         if is_log_progress:
#             iterator.set_description('REDUCING [{}]'.format(key_this_iter))

#         configs[key_this_iter] = config_to_reduce_
#         if len(configs_to_reduce.keys()) == 0:

#             if "None" in configs.keys():
#                 del configs["None"]

#             df = get_dataframes(
#                 to_match_configs=configs,
#                 analysis=analysis,
#             )

#             if df is not None:
#                 results_to_reduce.append(eval(metric))

#             else:
#                 logger.warning('failed to get data for \n{}'.format(
#                     pp.pformat(configs),
#                 ))

#         else:
#             results_to_reduce.extend(
#                 get_results_to_reduce(
#                     analysis=analysis,
#                     configs_to_reduce=configs_to_reduce,
#                     metric=metric,
#                     configs=configs,
#                 )
#             )
#     return results_to_reduce


# def get_kth_smallest(data, kth):
#     """Get the kth small value in data.
#     """
#     data_flatten = data.flatten()
#     kth_smallest = data_flatten[
#         np.argpartition(
#             data_flatten,
#             kth=kth,
#         )[kth]
#     ]
#     return kth_smallest


# def draw_heatmap(data, xlabel=None, ylabel=None, annot=False, xticklabels=None, yticklabels=None, norm=None, cmap='Blues'):
#     """Draws heatmap. Outdated.
#     """
#     normer = None
#     if norm == 'SymLogNorm':
#         if np.min(data) == 0.0:
#             kth = 1
#         elif np.min(data) > 0.0:
#             kth = 0
#         else:
#             raise NotImplementedError
#         normer = SymLogNorm(
#             linthresh=get_kth_smallest(data, kth=kth),
#         )
#     elif norm == 'LogNorm':
#         normer = LogNorm()
#     kwargs = {
#         'annot': annot,
#         'fmt': '.2e',
#         'cmap': cmap,
#     }
#     annot_kws = {
#         # "fontsize": 8
#     }
#     if normer is not None:
#         kwargs['norm'] = normer
#     ax = sns.heatmap(data, **kwargs, annot_kws=annot_kws)
#     if xlabel is not None:
#         ax.set(
#             xlabel=xlabel,
#         )
#     if ylabel is not None:
#         ax.set(
#             ylabel=ylabel,
#         )
#     if xticklabels is not None:
#         ax.set_xticklabels(xticklabels)
#     if yticklabels is not None:
#         ax.set_yticklabels(yticklabels)
