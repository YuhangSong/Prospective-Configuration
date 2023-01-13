import os
import copy
import tempfile
import pandas as pd

from ray import tune, air

import analysis_utils as au
import utils as u


def test_analysis_v1():

    if 'RAY_ADDRESS' in os.environ:
        del os.environ['RAY_ADDRESS']

    """Test all neccessary functions/pipeline for analysis_v1.py
    """

    local_dir = tempfile.mkdtemp()

    class Trainable(tune.Trainable):
        def step(self):
            return {'metric': self._iteration*self.config['learning_rate']*self.config['seed']}

        def save_checkpoint(self, tmp_checkpoint_dir):
            return tmp_checkpoint_dir

    excpected_df_from_get_analysis_df = pd.DataFrame({
        "('learning_rate',)": [1.0, 1.0, 2.0, 2.0],
        "('seed',)": [1, 2, 1, 2],
        "df['metric'].max()": [1.0, 2.0, 2.0, 4.0],
    })

    # Test step: get_analysis_df

    # # Test case: basic

    tune.Tuner(
        Trainable,
        run_config=air.RunConfig(
            local_dir=local_dir,
            stop={"training_iteration": 2},
        ),
        param_space={
            "learning_rate": tune.grid_search([1.0, 2.0]),
            "seed": tune.grid_search([1, 2]),
        },
    ).fit()

    analysis_df = au.get_analysis_df(
        au.Analysis(local_dir),
        metrics=[
            "df['metric'].max()",
        ],
    )

    u.assert_frame_equal(
        analysis_df.get_dataframe_with_config_metric_columns_only(),
        excpected_df_from_get_analysis_df,
    )

    # # Test case: multiple metrics

    analysis_df = au.get_analysis_df(
        au.Analysis(local_dir),
        metrics=[
            "df['metric'].max()",
            "df['metric'].min()",
        ],
    )

    u.assert_frame_equal(
        analysis_df.get_dataframe_with_config_metric_columns_only(),
        pd.DataFrame({
            "('learning_rate',)": [1.0, 1.0, 2.0, 2.0],
            "('seed',)": [1, 2, 1, 2],
            "df['metric'].max()": [1.0, 2.0, 2.0, 4.0],
            "df['metric'].min()": [0.0, 0.0, 0.0, 0.0],
        }),
    )

    # # Test case: compress_plot and extract_plot

    analysis_df = au.get_analysis_df(
        au.Analysis(local_dir),
        metrics=[
            "compress_plot('metric','training_iteration')"
        ],
    )

    u.assert_frame_equal(
        analysis_df.get_dataframe_with_config_metric_columns_only(),
        pd.DataFrame({
            "('learning_rate',)": [1.0, 1.0, 2.0, 2.0],
            "('seed',)": [1, 2, 1, 2],
            "metric-along-training_iteration": [
                [[0.0, 1.0], [1.0, 2.0]],
                [[0.0, 1.0], [2.0, 2.0]],
                [[0.0, 1.0], [2.0, 2.0]],
                [[0.0, 1.0], [4.0, 2.0]],
            ],
        }),
    )

    df = au.extract_plot(
        analysis_df.get_dataframe_with_config_metric_columns_only(),
        'metric', 'training_iteration',
    )

    u.assert_frame_equal(
        df,
        pd.DataFrame({
            "('learning_rate',)": [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0],
            "('seed',)": [1, 2, 1, 2, 1, 2, 1, 2],
            "metric": [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 4.0],
            "training_iteration": [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
        }),
    )

    # # Test case: there are duplicate trials + analysis_df.one_row_per_config()

    tune.Tuner(
        Trainable,
        run_config=air.RunConfig(
            local_dir=local_dir,
            stop={"training_iteration": 2},
        ),
        param_space={
            "learning_rate": tune.grid_search([1.0, 2.0]),
            "seed": tune.grid_search([1, 2]),
        },
    ).fit()

    analysis_df = au.get_analysis_df(
        au.Analysis(local_dir),
        metrics=[
            "df['metric'].max()",
        ],
    )

    analysis_df.one_row_per_config()

    u.assert_frame_equal(
        analysis_df.get_dataframe_with_config_metric_columns_only(),
        excpected_df_from_get_analysis_df,
    )

    # # Test case: there are incompleted trials
    # TODO: add this test case, need to catch loggings

    # # Test case: there are missing trials
    # TODO: add this test case, need to catch loggings

    # Test step: purge_analysis_df

    # # Test case: basic

    tune.Tuner(
        Trainable,
        run_config=air.RunConfig(
            local_dir=local_dir,
            stop={"training_iteration": 2},
        ),
        param_space={
            "learning_rate": tune.grid_search([0.0]),
            "seed": tune.grid_search([1]),
        },
    ).fit()

    analysis_df = au.get_analysis_df(
        au.Analysis(local_dir),
        metrics=[
            "df['metric'].max()",
        ],
    )

    analysis_df.one_row_per_config()

    config_dict = {
        "learning_rate": {
            "grid_search": [1.0, 2.0],
        },
        "seed": {
            "grid_search": [1, 2],
        },
    }

    analysis_df = au.purge_analysis_df(
        copy.deepcopy(analysis_df),
        config_dict,
    )

    u.assert_frame_equal(
        analysis_df.get_dataframe_with_config_metric_columns_only(),
        excpected_df_from_get_analysis_df,
    )

    # Test step: purge_analysis_df_with_config_dicts

    # # Test case: basic

    # add another run, so that to test purge_analysis_df_with_config_dicts
    tune.Tuner(
        Trainable,
        run_config=air.RunConfig(
            local_dir=local_dir,
            stop={"training_iteration": 2},
        ),
        param_space={
            "learning_rate": tune.grid_search([3]),
            "seed": tune.grid_search([1]),
        },
    ).fit()

    analysis_df = au.get_analysis_df(
        au.Analysis(local_dir),
        metrics=[
            "df['metric'].max()",
        ],
    )

    analysis_df.one_row_per_config()

    config_dicts = [
        {
            "learning_rate": {
                "grid_search": [1.0, 2.0],
            },
            "seed": {
                "grid_search": [1, 2],
            }
        },
        {
            "learning_rate": {
                "grid_search": [3.0],
            },
            "seed": {
                "grid_search": [1],
            },
        },
    ]

    analysis_df = au.purge_analysis_df_with_config_dicts(
        copy.deepcopy(analysis_df),
        config_dicts=config_dicts,
    )

    u.assert_frame_equal(
        analysis_df.get_dataframe_with_config_metric_columns_only(),
        pd.DataFrame({
            "('learning_rate',)": [1.0, 1.0, 2.0, 2.0, 3.0],
            "('seed',)": [1, 2, 1, 2, 1],
            "df['metric'].max()": [1.0, 2.0, 2.0, 4.0, 3.0],
        })
    )

    # # Test case: two runs with the same config but different names

    # add another run, so that to test purge_analysis_df_with_config_dicts

    class TrainableLr(tune.Trainable):
        def step(self):
            return {'metric': self._iteration*self.config['lr']*self.config['seed']}

        def save_checkpoint(self, tmp_checkpoint_dir):
            return tmp_checkpoint_dir

    tune.Tuner(
        TrainableLr,
        run_config=air.RunConfig(
            local_dir=local_dir,
            stop={"training_iteration": 2},
        ),
        param_space={
            "lr": tune.grid_search([4]),
            "seed": tune.grid_search([1]),
        },
    ).fit()

    analysis_df = au.get_analysis_df(
        au.Analysis(local_dir),
        metrics=[
            "df['metric'].max()",
        ],
    )

    analysis_df.one_row_per_config()

    config_dicts = [
        {
            "learning_rate": {
                "grid_search": [1.0, 2.0],
            },
            "seed": {
                "grid_search": [1, 2],
            }
        },
        {
            "lr": {
                "grid_search": [4],
            },
            "seed": {
                "grid_search": [1],
            },
        },
    ]

    analysis_df = au.purge_analysis_df_with_config_dicts(
        copy.deepcopy(analysis_df),
        config_dicts=config_dicts,
        rename_dict={
            "('lr',)": "('learning_rate',)",
        }
    )

    u.assert_frame_equal(
        analysis_df.get_dataframe_with_config_metric_columns_only(),
        pd.DataFrame({
            "('learning_rate',)": [1.0, 1.0, 2.0, 2.0, 4.0],
            "('seed',)": [1, 2, 1, 2, 1],
            "df['metric'].max()": [1.0, 2.0, 2.0, 4.0, 4.0],
        })
    )
