import pytest

import os
import copy
import tempfile
import pandas as pd

from ray import tune, air

import analysis_utils as au
import utils as u

from base_trainable import BaseTrainable


@pytest.fixture
def get_results_df():
    def get_results_df(analysis_df):
        """"Get dataframe with only config and metric columns, and without "('seed',)" column."""
        assert isinstance(analysis_df, au.AnalysisDataFrame)
        return analysis_df.get_dataframe_with_config_metric_columns_only(
        ).drop("('seed',)", axis=1)
    return get_results_df


@pytest.fixture
def run_with_seed():
    def run_with_seed(seed):
        local_dir = tempfile.mkdtemp()
        tune.Tuner(
            BaseTrainable,
            run_config=air.RunConfig(
                local_dir=local_dir,
                stop={"training_iteration": 1},
            ),
            param_space={
                "seed": seed,
                "step_code": "result_dict['metric'] = np.random.random_sample() * torch.rand(1).item() * random.random()",
            },
        ).fit()
        analysis_df = au.get_analysis_df(
            au.Analysis(local_dir),
            metrics=[
                "df['metric'].iloc[-1]",
            ],
        )
        return analysis_df
    return run_with_seed


def test_randomness_control(get_results_df, run_with_seed):

    if 'RAY_ADDRESS' in os.environ:
        del os.environ['RAY_ADDRESS']

    analysis_df = run_with_seed(3242)

    analysis_df_different_seed = run_with_seed(3241)

    analysis_df_same_seed = run_with_seed(3242)

    with pytest.raises(AssertionError):
        u.assert_frame_equal(
            get_results_df(
                analysis_df
            ), get_results_df(
                analysis_df_different_seed
            )
        )

    u.assert_frame_equal(
        get_results_df(
            analysis_df
        ), get_results_df(
            analysis_df_same_seed
        )
    )


def test_num_iterations():

    if 'RAY_ADDRESS' in os.environ:
        del os.environ['RAY_ADDRESS']

    local_dir = tempfile.mkdtemp()
    tune.Tuner(
        BaseTrainable,
        run_config=air.RunConfig(
            local_dir=local_dir,
            stop={"is_num_iterations_reached": 1},
        ),
        param_space={
            "num_iterations": tune.grid_search([1, 2]),
            "step_code": "result_dict['metric'] = self._iteration",
        },
    ).fit()
    analysis_df = au.get_analysis_df(
        au.Analysis(local_dir),
        metrics=[
            "df['metric'].iloc[-1]",
        ],
    )
    # u.print_df_as_dict(
    #     analysis_df.get_dataframe_with_config_metric_columns_only()
    # )
    u.assert_frame_equal(
        analysis_df.get_dataframe_with_config_metric_columns_only().drop(
            "('step_code',)", axis=1
        ),
        pd.DataFrame({
            "('num_iterations',)": [
                2,
                1
            ],
            "df['metric'].iloc[-1]": [
                1,
                0
            ]
        })
    )
