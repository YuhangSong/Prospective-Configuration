import pytest

import os
import copy
import tempfile
import numpy as np
import pandas as pd

from ray import tune, air

import analysis_utils as au
import utils as u

from dataset_learning_trainable import DatasetLearningTrainable


class TestTrainable(DatasetLearningTrainable):
    def iteration_step(
        self,
        data_pack_key,
        batch_idx,
        batch,
        do_key,
    ):
        data, target = batch
        if do_key == 'learn':
            self.metric += (data + target).item()*self._iteration
        elif do_key == 'predict':
            self.metric += (data * target).item()*self._iteration
        else:
            raise ValueError('do_key must be learn or predict')


@pytest.fixture
def run_with():
    def run_with(param_space,
                 metrics=[
                     "df['train__metric'].iloc[0]",
                     "df['train__metric'].iloc[1]",
                 ],
                 ):
        if 'RAY_ADDRESS' in os.environ:
            del os.environ['RAY_ADDRESS']
        param_space.update({
            "before_iteration_code": "self.metric = 0",
        })
        local_dir = tempfile.mkdtemp()
        tune.Tuner(
            TestTrainable,
            run_config=air.RunConfig(
                local_dir=local_dir,
                stop={"training_iteration": 2},
            ),
            param_space=param_space,
        ).fit()
        analysis_df = au.get_analysis_df(
            au.Analysis(local_dir),
            metrics=metrics,
        )
        return analysis_df
    return run_with


def test_basic(run_with):

    analysis_df = run_with({
        "data_packs": {
            "train": {
                "data_loader": "u.tensordataset_data_loader([0, 1, 2], [0, 1, 2])",
            },
        },
        "log_packs": {
            "metric": {
                "log": "self.metric",
            },
        },
    })

    # u.print_df_as_dict(
    #     analysis_df.get_dataframe_with_metric_columns_only()
    # )
    u.assert_frame_equal(
        analysis_df.get_dataframe_with_metric_columns_only(),
        pd.DataFrame({
            "df['train__metric'].iloc[0]": [
                # ... * 0 = 0
                0.0
            ],
            "df['train__metric'].iloc[1]": [
                # [0, 1, 2]+[0, 1, 2] = [0, 2, 4]
                # [0, 1, 2]*[0, 1, 2] = [0, 1, 4]
                # [0, 2, 4]+[0, 1, 4] = [0, 3, 8]
                # sum([0, 3, 8])/3 = 3.6666666666666667
                # 3.6666666666666667 * 1 = 3.6666666666666667
                3.6666666666666667
            ],
        })
    )


def test_data_packs_at_iteration_0(run_with):

    analysis_df = run_with({
        "data_packs": {
            "train": {
                "data_loader": "u.tensordataset_data_loader([0, 1, 2], [0, 1, 2])",
                "at_iteration": "[0]",
            },
        },
        "log_packs": {
            "metric": {
                "log": "self.metric",
            },
        },
    })

    # u.print_df_as_dict(
    #     analysis_df.get_dataframe_with_metric_columns_only()
    # )
    u.assert_frame_equal(
        analysis_df.get_dataframe_with_metric_columns_only(),
        pd.DataFrame({
            "df['train__metric'].iloc[0]": [
                # ... * 0 = 0
                0.0
            ],
            "df['train__metric'].iloc[1]": [
                np.nan
            ],
        })
    )


def test_data_packs_at_iteration_1(run_with):

    analysis_df = run_with({
        "data_packs": {
            "train": {
                "data_loader": "u.tensordataset_data_loader([0, 1, 2], [0, 1, 2])",
                "at_iteration": "[1]",
            },
        },
        "log_packs": {
            "metric": {
                "log": "self.metric",
            },
        },
        "log_key_holders": "['train__metric']",
    })

    # u.print_df_as_dict(
    #     analysis_df.get_dataframe_with_metric_columns_only()
    # )
    u.assert_frame_equal(
        analysis_df.get_dataframe_with_metric_columns_only(),
        pd.DataFrame({
            "df['train__metric'].iloc[0]": [
                np.nan
            ],
            "df['train__metric'].iloc[1]": [
                # [0, 1, 2]+[0, 1, 2] = [0, 2, 4]
                # [0, 1, 2]*[0, 1, 2] = [0, 1, 4]
                # [0, 2, 4]+[0, 1, 4] = [0, 3, 8]
                # sum([0, 3, 8])/3 = 3.6666666666666667
                # 3.6666666666666667 * 1 = 3.6666666666666667
                3.6666666666666667
            ],
        })
    )


def test_data_packs_do_predict(run_with):

    analysis_df = run_with({
        "data_packs": {
            "train": {
                "data_loader": "u.tensordataset_data_loader([0, 1, 2], [0, 1, 2])",
                "do": "['predict']",
            },
        },
        "log_packs": {
            "metric": {
                "log": "self.metric",
            },
        },
    })

    # u.print_df_as_dict(
    #     analysis_df.get_dataframe_with_metric_columns_only()
    # )
    u.assert_frame_equal(
        analysis_df.get_dataframe_with_metric_columns_only(),
        pd.DataFrame({
            "df['train__metric'].iloc[0]": [
                # ... * 0 = 0
                0.0
            ],
            "df['train__metric'].iloc[1]": [
                # [0, 1, 2]*[0, 1, 2] = [0, 1, 4]
                # sum([0, 1, 4])/3 = 1.6666666666666667
                # 1.6666666666666667 * 1 = 1.6666666666666667
                1.6666666666666667
            ],
        })
    )


def test_data_packs_do_learn(run_with):

    analysis_df = run_with({
        "data_packs": {
            "train": {
                "data_loader": "u.tensordataset_data_loader([0, 1, 2], [0, 1, 2])",
                "do": "['learn']",
            },
        },
        "log_packs": {
            "metric": {
                "log": "self.metric",
            },
        },
    })

    # u.print_df_as_dict(
    #     analysis_df.get_dataframe_with_metric_columns_only()
    # )
    u.assert_frame_equal(
        analysis_df.get_dataframe_with_metric_columns_only(),
        pd.DataFrame({
            "df['train__metric'].iloc[0]": [
                # ... * 0 = 0
                0.0
            ],
            "df['train__metric'].iloc[1]": [
                # [0, 1, 2]+[0, 1, 2] = [0, 2, 4]
                # sum([0, 2, 4])/3 = 2.0
                # 2.0 * 1 = 2.0
                2.0
            ],
        })
    )


def test_log_fn(run_with):

    analysis_df = run_with({
        "data_packs": {
            "train": {
                "data_loader": "u.tensordataset_data_loader([0, 1, 2], [0, 1, 2])",
            },
        },
        "log_packs": {
            "metric": {
                "log_fn": "lambda sf: sf.metric",
            },
        },
    })

    # u.print_df_as_dict(
    #     analysis_df.get_dataframe_with_metric_columns_only()
    # )
    u.assert_frame_equal(
        analysis_df.get_dataframe_with_metric_columns_only(),
        pd.DataFrame({
            "df['train__metric'].iloc[0]": [
                # ... * 0 = 0
                0.0
            ],
            "df['train__metric'].iloc[1]": [
                # [0, 1, 2]+[0, 1, 2] = [0, 2, 4]
                # [0, 1, 2]*[0, 1, 2] = [0, 1, 4]
                # [0, 2, 4]+[0, 1, 4] = [0, 3, 8]
                # sum([0, 3, 8])/3 = 3.6666666666666667
                # 3.6666666666666667 * 1 = 3.6666666666666667
                3.6666666666666667
            ],
        })
    )


def test_log_packs_at_iteration_0(run_with):

    analysis_df = run_with({
        "data_packs": {
            "train": {
                "data_loader": "u.tensordataset_data_loader([0, 1, 2], [0, 1, 2])",
            },
        },
        "log_packs": {
            "metric": {
                "log": "self.metric",
                "at_iteration": "[0]",
            },
        },
    })

    # u.print_df_as_dict(
    #     analysis_df.get_dataframe_with_metric_columns_only()
    # )
    u.assert_frame_equal(
        analysis_df.get_dataframe_with_metric_columns_only(),
        pd.DataFrame({
            "df['train__metric'].iloc[0]": [
                # ... * 0 = 0
                0.0
            ],
            "df['train__metric'].iloc[1]": [
                np.nan
            ],
        })
    )


def test_log_packs_at_iteration_1(run_with):

    analysis_df = run_with({
        "data_packs": {
            "train": {
                "data_loader": "u.tensordataset_data_loader([0, 1, 2], [0, 1, 2])",
            },
        },
        "log_packs": {
            "metric": {
                "log": "self.metric",
                "at_iteration": "[1]",
            },
        },
        "log_key_holders": "['train__metric']",
    })

    # u.print_df_as_dict(
    #     analysis_df.get_dataframe_with_metric_columns_only()
    # )
    u.assert_frame_equal(
        analysis_df.get_dataframe_with_metric_columns_only(),
        pd.DataFrame({
            "df['train__metric'].iloc[0]": [
                # ... * 0 = 0
                np.nan
            ],
            "df['train__metric'].iloc[1]": [
                # [0, 1, 2]+[0, 1, 2] = [0, 2, 4]
                # [0, 1, 2]*[0, 1, 2] = [0, 1, 4]
                # [0, 2, 4]+[0, 1, 4] = [0, 3, 8]
                # sum([0, 3, 8])/3 = 3.6666666666666667
                # 3.6666666666666667 * 1 = 3.6666666666666667
                3.6666666666666667
            ],
        })
    )


def test_log_packs_at_data_pack_default(run_with):

    analysis_df = run_with(
        {
            "data_packs": {
                "train": {
                    "data_loader": "u.tensordataset_data_loader([0, 1, 2], [0, 1, 2])",
                },
                "test": {
                    "data_loader": "u.tensordataset_data_loader([1, 2, 3], [1, 2, 3])",
                },
            },
            "log_packs": {
                "metric": {
                    "log": "self.metric",
                },
            },
        },
        metrics=[
            "df['train__metric'].iloc[0] if 'train__metric' in df.columns else np.nan",
            "df['train__metric'].iloc[1] if 'train__metric' in df.columns else np.nan",
            "df['test__metric'].iloc[0] if 'test__metric' in df.columns else np.nan",
            "df['test__metric'].iloc[1] if 'test__metric' in df.columns else np.nan",
        ]
    )

    # u.print_df_as_dict(
    #     analysis_df.get_dataframe_with_metric_columns_only()
    # )
    u.assert_frame_equal(
        analysis_df.get_dataframe_with_metric_columns_only(),
        pd.DataFrame({
            "df['train__metric'].iloc[0] if 'train__metric' in df.columns else np.nan": [
                # ... * 0 = 0
                0.0
            ],
            "df['train__metric'].iloc[1] if 'train__metric' in df.columns else np.nan": [
                # [0, 1, 2]+[0, 1, 2] = [0, 2, 4]
                # [0, 1, 2]*[0, 1, 2] = [0, 1, 4]
                # [0, 2, 4]+[0, 1, 4] = [0, 3, 8]
                # sum([0, 3, 8])/3 = 3.6666666666666667
                # 3.6666666666666667 * 1 = 3.6666666666666667
                3.6666666666666667
            ],
            "df['test__metric'].iloc[0] if 'test__metric' in df.columns else np.nan": [
                # ... * 0 = 0
                0.0
            ],
            "df['test__metric'].iloc[1] if 'test__metric' in df.columns else np.nan": [
                # [1, 2, 3]+[1, 2, 3] = [2, 4, 6]
                # [1, 2, 3]*[1, 2, 3] = [1, 4, 9]
                # [2, 4, 6]+[1, 4, 9] = [3, 8, 15]
                # sum([3, 8, 15])/3 = 8.666666666666668
                # 8.666666666666668 * 1 = 8.666666666666668
                8.666666666666668
            ],
        })
    )


def test_log_packs_at_data_pack_train(run_with):

    analysis_df = run_with(
        {
            "data_packs": {
                "train": {
                    "data_loader": "u.tensordataset_data_loader([0, 1, 2], [0, 1, 2])",
                },
                "test": {
                    "data_loader": "u.tensordataset_data_loader([1, 2, 3], [1, 2, 3])",
                },
            },
            "log_packs": {
                "metric": {
                    "log": "self.metric",
                    "at_data_pack": "['train']",
                },
            },
        },
        metrics=[
            "df['train__metric'].iloc[0] if 'train__metric' in df.columns else np.nan",
            "df['train__metric'].iloc[1] if 'train__metric' in df.columns else np.nan",
            "df['test__metric'].iloc[0] if 'test__metric' in df.columns else np.nan",
            "df['test__metric'].iloc[1] if 'test__metric' in df.columns else np.nan",
        ]
    )

    # u.print_df_as_dict(
    #     analysis_df.get_dataframe_with_metric_columns_only()
    # )
    u.assert_frame_equal(
        analysis_df.get_dataframe_with_metric_columns_only(),
        pd.DataFrame({
            "df['train__metric'].iloc[0] if 'train__metric' in df.columns else np.nan": [
                # ... * 0 = 0
                0.0
            ],
            "df['train__metric'].iloc[1] if 'train__metric' in df.columns else np.nan": [
                # [0, 1, 2]+[0, 1, 2] = [0, 2, 4]
                # [0, 1, 2]*[0, 1, 2] = [0, 1, 4]
                # [0, 2, 4]+[0, 1, 4] = [0, 3, 8]
                # sum([0, 3, 8])/3 = 3.6666666666666667
                # 3.6666666666666667 * 1 = 3.6666666666666667
                3.6666666666666667
            ],
            "df['test__metric'].iloc[0] if 'test__metric' in df.columns else np.nan": [
                np.nan
            ],
            "df['test__metric'].iloc[1] if 'test__metric' in df.columns else np.nan": [
                np.nan
            ],
        })
    )


def test_log_packs_summarize_over_batch_idx_fn(run_with):

    analysis_df = run_with({
        "data_packs": {
            "train": {
                "data_loader": "u.tensordataset_data_loader([0, 1, 2], [0, 1, 2])",
            },
        },
        "log_packs": {
            "metric": {
                "log": "self.metric",
                "summarize_over_batch_idx_fn": "lambda x: np.max(x)",
            },
        },
    })

    # u.print_df_as_dict(
    #     analysis_df.get_dataframe_with_metric_columns_only()
    # )
    u.assert_frame_equal(
        analysis_df.get_dataframe_with_metric_columns_only(),
        pd.DataFrame({
            "df['train__metric'].iloc[0]": [
                # ... * 0 = 0
                0.0
            ],
            "df['train__metric'].iloc[1]": [
                # max(2+2, ...) + max(2*2, ...) = 4+4 = 8
                # 8 * 1 = 8
                8.0
            ],
        })
    )


def test_log_packs_at_batch_idx(run_with):

    analysis_df = run_with({
        "data_packs": {
            "train": {
                "data_loader": "u.tensordataset_data_loader([0, 1, 2], [0, 1, 2])",
            },
        },
        "log_packs": {
            "metric": {
                "log": "self.metric",
                "at_batch_idx": "[2]",
            },
        },
    })

    # u.print_df_as_dict(
    #     analysis_df.get_dataframe_with_metric_columns_only()
    # )
    u.assert_frame_equal(
        analysis_df.get_dataframe_with_metric_columns_only(),
        pd.DataFrame({
            "df['train__metric'].iloc[0]": [
                # ... * 0 = 0
                0.0
            ],
            "df['train__metric'].iloc[1]": [
                # (2+2) + (2*2) = 4+4 = 8
                # 8 * 1 = 8
                8.0
            ],
        })
    )
