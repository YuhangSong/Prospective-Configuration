import pytest
import os
import uuid
import copy
import tempfile
import pandas as pd
import numpy as np

import utils as u

from analysis_utils import *


@pytest.fixture
def typical_list():

    return [1, 2]


def test_typical_list(typical_list):

    a = copy.deepcopy(typical_list)
    assert a == [1, 2]
    a.remove(1)
    assert a == [2]
    b = copy.deepcopy(typical_list)
    assert b == [1, 2]


@pytest.fixture
def typical_dataframe():

    return pd.DataFrame({
        'rule': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        'learning_rate': [3, 3, 4, 4, 3, 3, 4, 4],
        'seed': [1, 2, 1, 2, 1, 2, 1, 2],
        'metric': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    })


def test_resolve_nested_dict():

    # Test with a simple nested dict
    nested_dict = {'a': 1, 'b': {'c': 2, 'd': 3}}
    expected_output = {('a',): 1, ('b', 'c'): 2, ('b', 'd'): 3}
    assert resolve_nested_dict(nested_dict) == expected_output

    # Test with an empty dict
    nested_dict = {}
    expected_output = {}
    assert resolve_nested_dict(nested_dict) == expected_output

    # Test with a dict with just one level of nesting
    nested_dict = {'a': {'b': 1, 'c': 2}, 'd': 3}
    expected_output = {('a', 'b'): 1, ('a', 'c'): 2, ('d',): 3}
    assert resolve_nested_dict(nested_dict) == expected_output

    # Test with a dict with multiple levels of nesting
    nested_dict = {'a': {'b': {'c': 1, 'd': 2}, 'e': 3}, 'f': 4}
    expected_output = {('a', 'b', 'c'): 1, ('a', 'b', 'd'): 2, ('a', 'e'): 3, ('f',): 4}
    assert resolve_nested_dict(nested_dict) == expected_output


def test_format_friendly_string():

    # Test with a simple string
    string = "this is a test"
    expected_output = "this_is_a_test"
    assert format_friendly_string(string) == expected_output

    # Test with an empty string
    string = ""
    expected_output = ""
    assert format_friendly_string(string) == expected_output

    # Test with a string that contains special characters, punctuation, and spaces
    string = "this! is a test"
    expected_output = "this_is_a_test"
    assert format_friendly_string(string) == expected_output

    # Test with the is_abbr flag set to True
    string = "this is a test"
    expected_output = "this_is_a_test"
    assert format_friendly_string(string, is_abbr=True) == expected_output

    # Test with a string that is all special characters and punctuation
    string = "!!!"
    expected_output = ""
    assert format_friendly_string(string) == expected_output


def test_save_load_dict():

    uid = str(uuid.uuid4())

    # Test saving and loading with default parameters
    d = {'a': 1, 'b': 2}
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dict(d, tmpdir, f'test-{uid}')
        loaded_d = load_dict(logdir=tmpdir, title=f'test-{uid}')
        assert d == loaded_d

    # Test saving and loading with different formats
    d = {'a': 1, 'b': 2}
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dict(d, tmpdir, f'test-{uid}', formats=['yaml', 'json', 'pickle'])
        loaded_d_yaml = load_dict(
            logdir=tmpdir, title=f'test-{uid}', format='yaml')
        loaded_d_json = load_dict(
            logdir=tmpdir, title=f'test-{uid}', format='json')
        loaded_d_pickle = load_dict(
            logdir=tmpdir, title=f'test-{uid}', format='pickle')
        assert d == loaded_d_yaml == loaded_d_json == loaded_d_pickle

    # Test saving and loading with a specified path
    d = {'a': 1, 'b': 2}
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, f'test-{uid}.yaml')
        save_dict(d, logdir=tmpdir, title=f'test-{uid}', formats=['yaml'])
        loaded_d = load_dict(path=save_path)
        assert d == loaded_d


def test_load_config_from_params():

    # Test loading a config from params.json
    config = {'a': 1, 'b': 2}
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, 'params.json'), 'w') as f:
            json.dump(config, f)
        loaded_config = load_config_from_params(tmpdir)
        assert config == loaded_config

    # Test loading a config from a nonexistent params.json
    with tempfile.TemporaryDirectory() as tmpdir:
        loaded_config = load_config_from_params(tmpdir)
        assert loaded_config is None


def test_summarize_select(typical_dataframe):

    # Create a test dataframe
    df = copy.deepcopy(typical_dataframe)

    # Define the configuration columns
    config_columns = ['rule', 'learning_rate', 'seed']

    # Define the groups to summarize over and the summarize function
    group_summarize = ['seed']
    def summarize_fn(df): return df['metric'].mean()

    # Define the groups to select over and the select function
    group_select = ['learning_rate']
    def select_fn(df): return df['summarize'] == df['summarize'].min()

    # Test the summarize_select function
    result_df = summarize_select(
        df, config_columns, group_summarize, summarize_fn, group_select, select_fn
    )

    # Assert that the resulting dataframe is correct
    u.assert_frame_equal(
        result_df, pd.DataFrame({
            'rule': ['A', 'A', 'B', 'B'],
            'learning_rate': [3, 3, 3, 3],
            'seed': [1, 2, 1, 2],
            'metric': [0.1, 0.2, 0.5, 0.6]
        }),
    )


def test_select_best_lr(typical_dataframe):

    # Test case: basic
    df = copy.deepcopy(typical_dataframe)
    config_columns = ['rule', 'learning_rate', 'seed']
    u.assert_frame_equal(
        select_best_lr(
            df, config_columns, 'metric',
        ), pd.DataFrame({
            'rule': ['A', 'A', 'B', 'B'],
            'learning_rate': [3, 3, 3, 3],
            'seed': [1, 2, 1, 2],
            'metric': [0.1, 0.2, 0.5, 0.6]
        }),
    )

    # Test case: is_min=True
    df = copy.deepcopy(typical_dataframe)
    u.assert_frame_equal(
        select_best_lr(
            df, config_columns, 'metric', is_min=False,
        ), pd.DataFrame({
            'rule': ['A', 'A', 'B', 'B'],
            'learning_rate': [4, 4, 4, 4],
            'seed': [1, 2, 1, 2],
            'metric': [0.3, 0.4, 0.7, 0.8]
        }),
    )


def test_one_row_per_config():

    # Test case: basic
    df = pd.DataFrame({
        'config_col_1': ['A', 'A', 'B', 'B'],
        'config_col_2': [1, 1, 2, 2],
        'metric': [0.1, 0.2, 0.3, 0.4],
    })
    metric_columns = ['metric']
    config_columns = ['config_col_1', 'config_col_2']
    result_df = one_row_per_config(df, metric_columns, config_columns)
    assert (result_df['config_col_1'] == ['A', 'B']).all()
    assert (result_df['config_col_2'] == [1, 2]).all()
    assert (result_df['metric'] == [0.1, 0.3]).all()

    # Test case: keep_small=False
    df = pd.DataFrame({
        'config_col_1': ['A', 'A', 'B', 'B'],
        'config_col_2': [1, 1, 2, 2],
        'metric': [0.1, 0.2, 0.3, 0.4],
    })
    result_df = one_row_per_config(
        df, metric_columns, config_columns, keep_small=False
    )
    assert (result_df['config_col_1'] == ['B', 'A']).all()
    assert (result_df['config_col_2'] == [2, 1]).all()
    assert (result_df['metric'] == [0.4, 0.2]).all()

    # Test case: with nan values
    df = pd.DataFrame({
        'config_col_1': ['A', 'A', 'B', 'B'],
        'config_col_2': [1, 1, 2, 2],
        'metric': [np.nan, 0.2, 0.3, 0.4]
    })
    result_df = one_row_per_config(df, metric_columns, config_columns)
    assert (result_df['config_col_1'] == ['A', 'B']).all()
    assert (result_df['config_col_2'] == [1, 2]).all()
    assert (result_df['metric'] == [0.2, 0.3]).all()

    # Test case: with nan values, keep_small=False
    df = pd.DataFrame({
        'config_col_1': ['A', 'A', 'B', 'B'],
        'config_col_2': [1, 1, 2, 2],
        'metric': [0.1, 0.2, 0.3, np.nan]
    })
    result_df = one_row_per_config(
        df, metric_columns, config_columns, keep_small=False
    )
    assert (result_df['config_col_1'] == ['B', 'A']).all()
    assert (result_df['config_col_2'] == [2, 1]).all()
    assert (result_df['metric'] == [0.3, 0.2]).all()


def test_auto_phrase_config_for_analysis_df():
    # Create a test DataFrame
    df = pd.DataFrame({
        'column_1': [1, 2, 3],
        'column_2': [4, 5, 6],
        "('a', 'b')": [7, 8, 9],
        "('c', 'd')": [10, 11, 12],
    })

    # Create an AnalysisDataFrame
    config_columns = ['column_1', "('a', 'b')"]
    metric_columns = ['column_2']
    analysis_df = AnalysisDataFrame(df, config_columns, metric_columns)

    # Test the auto_phrase_config_for_analysis_df function
    result = auto_phrase_config_for_analysis_df(analysis_df)
    expected = AnalysisDataFrame(pd.DataFrame({
        'column_1': [1, 2, 3],
        'column_2': [4, 5, 6],
        'a: b': [7, 8, 9],
        'c: d': [10, 11, 12],
    }), ['column_1', 'a: b'], ['column_2'])
    assert result.dataframe.equals(expected.dataframe)
    assert result.config_columns == expected.config_columns
    assert result.metric_columns == expected.metric_columns


def test_AnalysisDateFrame_rename():

    # create a test dataframe
    df = pd.DataFrame({
        "col_1": [1, 2, 3],
        "col_2": [4, 5, 6],
        "col_3": [7, 8, 9],
        "col_4": [10, 11, 12]
    })
    analysis_df = AnalysisDataFrame(df, ["col_1", "col_2"], ["col_3", "col_4"])

    # create a mapper to rename the columns
    mapper = {
        "col_1": "new_col_1",
        "col_2": "new_col_2",
        "col_3": "new_col_3",
        "col_4": "new_col_4"
    }

    # test the rename method
    analysis_df.rename(mapper)

    # ensure that the columns have been renamed in the dataframe
    assert list(analysis_df.dataframe.columns) == list(mapper.values())

    # ensure that the config_columns and metric_columns attributes have been updated
    assert analysis_df.config_columns == list(mapper.values())[:2]
    assert analysis_df.metric_columns == list(mapper.values())[2:]


def test_process_dkeys():

    # Test case: basic
    d = {'a': 1, 'b': 2, 'c': 3}
    expected_output = {'A': 1, 'B': 2, 'C': 3}

    def process(key):
        return key.upper()
    assert process_dkeys(d, process) == expected_output

    # Test case: empty dictionary
    d = {}
    expected_output = {}
    assert process_dkeys(d, process) == expected_output

    # Test case: string keys
    d = {'a': 1, 'b': 2, 'c': 3}
    expected_output = {'A': 1, 'B': 2, 'C': 3}

    def process(key):
        return key.upper()
    assert process_dkeys(d, process) == expected_output

    # Test case: non-string keys
    d = {1: 'a', 2: 'b', 3: 'c'}
    expected_output = {2: 'a', 4: 'b', 6: 'c'}

    def process(key):
        return key * 2
    assert process_dkeys(d, process) == expected_output

    # Test case: AssertionError on non-dict input
    with pytest.raises(AssertionError):
        process_dkeys([1, 2, 3], process)

    # Test case: AssertionError on non-callable process function
    with pytest.raises(AssertionError):
        process_dkeys(d, 'invalid_function')


def test_filter_dataframe_by_dict(typical_dataframe):

    # Test case: filter with only one key
    df = copy.deepcopy(typical_dataframe)
    d = {'rule': 'B'}
    expected_output = pd.DataFrame({
        'rule': ['B', 'B', 'B', 'B'],
        'learning_rate': [3, 3, 4, 4],
        'seed': [1, 2, 1, 2],
        'metric': [0.5, 0.6, 0.7, 0.8]
    })
    u.assert_frame_equal(
        filter_dataframe_by_dict(df, d), expected_output
    )

    # Test case: filter with multiple keys
    df = copy.deepcopy(typical_dataframe)
    d = {'rule': 'A', 'learning_rate': 4, 'seed': 1}
    expected_output = pd.DataFrame({
        'rule': ['A'],
        'learning_rate': [4],
        'seed': [1],
        'metric': [0.3]
    })
    u.assert_frame_equal(
        filter_dataframe_by_dict(df, d), expected_output
    )

    # Test case: return empty dataframe when no match
    df = copy.deepcopy(typical_dataframe)
    d = {'rule': 'A', 'learning_rate': 4, 'seed': 3}
    assert filter_dataframe_by_dict(df, d).empty

    # Test case: return None when keys are not in dataframe
    df = copy.deepcopy(typical_dataframe)
    d = {'rule': 'A', 'learning_rate': 4, 'hidden_size': 2}
    assert filter_dataframe_by_dict(df, d) is None


def test_drop_same_columns():

    # Test case: basic
    df = pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': [6, 7, 8, 9, 10],
        'col3': [10, 10, 10, 10, 10]
    })
    expected_output = pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': [6, 7, 8, 9, 10]
    })
    u.assert_frame_equal(drop_same_columns(df), expected_output)

    # Test case: all columns have unique values
    df = pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': [6, 7, 8, 9, 10],
        'col3': [11, 12, 13, 14, 15]
    })
    expected_output = copy.deepcopy(df)
    u.assert_frame_equal(drop_same_columns(df), expected_output)

    # Test case: all columns have the same value
    df = pd.DataFrame({
        'col1': [1, 1, 1, 1, 1],
        'col2': [1, 1, 1, 1, 1],
        'col3': [1, 1, 1, 1, 1]
    })
    assert drop_same_columns(df).empty


def test_drop_cols():

    typical_dataframe_ = pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': [6, 7, 8, 9, 10],
        'col3': [11, 12, 13, 14, 15]
    })

    # Test case: basic
    df = copy.deepcopy(typical_dataframe_)
    cols = ['col1', 'col3']
    expected_output = pd.DataFrame(
        {'col2': [6, 7, 8, 9, 10]}
    )
    assert drop_cols(df, cols).equals(expected_output)

    # Test case: drop all columns
    df = copy.deepcopy(typical_dataframe_)
    cols = ['col1', 'col2', 'col3']
    assert drop_cols(df, cols).empty

    # Test case: drop no columns
    df = copy.deepcopy(typical_dataframe_)
    cols = []
    expected_output = copy.deepcopy(df)
    assert drop_cols(df, cols).equals(expected_output)


def test_reduce(typical_dataframe):

    # Test case: basic
    df = copy.deepcopy(typical_dataframe)
    by = ['rule']
    def reduce_func(df): return {'metric': df['metric'].mean()}
    expected_output = pd.DataFrame(
        {
            'rule': ['A', 'B'],
            'metric': [0.25, 0.65]
        },
    )
    u.assert_frame_equal(
        reduce(df, by, reduce_func), expected_output
    )

    # Test case: multiple columns
    df = copy.deepcopy(typical_dataframe)
    by = ['rule', 'learning_rate']
    def reduce_func(df): return {'metric': df['metric'].mean()}
    expected_output = pd.DataFrame(
        {
            'rule': ['A', 'A', 'B', 'B'],
            'learning_rate': [3, 4, 3, 4],
            'metric': [0.15, 0.35, 0.55, 0.75]
        },
    )
    u.assert_frame_equal(
        reduce(df, by, reduce_func), expected_output
    )


def test_add_metric_per_group(typical_dataframe):

    # Test case: basic
    df = copy.deepcopy(typical_dataframe)
    by = ['rule', 'learning_rate']
    def metric_fun(df): return ('metric_new', df['metric'].mean())
    expected_output = pd.DataFrame(
        {
            'rule': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
            'learning_rate': [3, 3, 4, 4, 3, 3, 4, 4],
            'seed': [1, 2, 1, 2, 1, 2, 1, 2],
            'metric': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            'metric_new': [0.15, 0.15, 0.35, 0.35, 0.55, 0.55, 0.75, 0.75],
        },
    )
    u.assert_frame_equal(
        add_metric_per_group(df, by, metric_fun), expected_output
    )


def test_select_rows_per_group(typical_dataframe):

    # Test case: basic
    df = copy.deepcopy(typical_dataframe)
    by = ['rule', 'learning_rate']
    def select_fn(df): return df['metric'] == df['metric'].min()
    expected_output = pd.DataFrame({
        'rule': ['A', 'A', 'B', 'B'],
        'learning_rate': [3, 4, 3, 4],
        'seed': [1, 1, 1, 1],
        'metric': [0.1, 0.3, 0.5, 0.7],
    })
    u.assert_frame_equal(
        select_rows_per_group(df, by, select_fn), expected_output
    )


def test_new_col(typical_dataframe):

    # Test case: basic
    df = copy.deepcopy(typical_dataframe)
    def apply_fn(row): return row['seed']*row['metric']
    expected_output = pd.DataFrame(
        {
            'rule': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
            'learning_rate': [3, 3, 4, 4, 3, 3, 4, 4],
            'seed': [1, 2, 1, 2, 1, 2, 1, 2],
            'metric': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            'metric_new': [0.1, 0.4, 0.3, 0.8, 0.5, 1.2, 0.7, 1.6],
        },
    )
    u.assert_frame_equal(
        new_col(df, 'metric_new', apply_fn), expected_output
    )


def test_combine_cols(typical_dataframe):

    # Test case: basic
    df = copy.deepcopy(typical_dataframe)
    expected_output = pd.DataFrame(
        {
            'rule': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
            'learning_rate': [3, 3, 4, 4, 3, 3, 4, 4],
            'seed': [1, 2, 1, 2, 1, 2, 1, 2],
            'metric': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            'id': ['learning_rate=3, seed=1', 'learning_rate=3, seed=2', 'learning_rate=4, seed=1', 'learning_rate=4, seed=2', 'learning_rate=3, seed=1', 'learning_rate=3, seed=2', 'learning_rate=4, seed=1', 'learning_rate=4, seed=2'],
        },
    )
    u.assert_frame_equal(
        combine_cols(df, 'id', ['learning_rate', 'seed']), expected_output
    )

    # Test case: combine_opt='values'
    df = copy.deepcopy(typical_dataframe)
    expected_output = pd.DataFrame(
        {
            'rule': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
            'learning_rate': [3, 3, 4, 4, 3, 3, 4, 4],
            'seed': [1, 2, 1, 2, 1, 2, 1, 2],
            'metric': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            'id': ['3, 1', '3, 2', '4, 1', '4, 2', '3, 1', '3, 2', '4, 1', '4, 2'],
        },
    )
    u.assert_frame_equal(
        combine_cols(df, 'id', ['learning_rate', 'seed'],
                     combine_opt='values'), expected_output
    )


def test_reduce_metric(typical_dataframe):

    # Test case: basic
    df = copy.deepcopy(typical_dataframe)
    config_cols = ['rule', 'learning_rate']
    metric_col = 'metric'
    expected_output = pd.DataFrame(
        {
            'rule': ['A', 'A', 'B', 'B'],
            'learning_rate': [3, 4, 3, 4],
            'metric (mean)': [0.15, 0.35, 0.55, 0.75],
            'metric (min)': [0.1, 0.3, 0.5, 0.7],
        },
    )
    u.assert_frame_equal(
        reduce_metric(
            df, config_cols, metric_col,
            reduce_with=['mean', 'min']
        ), expected_output
    )


def test_get_non_numeric_cols():

    # Test case: basic
    df = pd.DataFrame({'col1': ['foo', 'bar', 'baz'], 'col2': [1, 2, 3]})
    result = get_non_numeric_cols(df)
    assert result == ['col1']

    # Test case: multiple non-numeric columns
    df = pd.DataFrame({'col1': ['foo', 'bar', 'baz'], 'col2': [
                      1, 2, 3], 'col3': ['a', 'b', 'c']})
    result2 = get_non_numeric_cols(df)
    assert result2 == ['col1', 'col3']

    # Test case: no non-numeric columns
    df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    result3 = get_non_numeric_cols(df)
    assert result3 == []

    # Test case: non-numeric column with numeric values and boolean values
    df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [True, 1, 0]})
    result3 = get_non_numeric_cols(df)
    assert result3 == ['col2']


def test_dict2str():

    # Test case: basic
    d = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
    result = dict2str(d)
    assert result == 'key1=value1, key2=value2, key3=value3'

    # Test case: empty dictionary
    d = {}
    result = dict2str(d)
    assert result == ''

    # Test case: dictionary with one key-value pair
    d = {'key1': 'value1'}
    result = dict2str(d)
    assert result == 'key1=value1'

    # Test case: dictionary with numeric values
    d = {'key1': 1, 'key2': 2.5, 'key3': 3}
    result = dict2str(d)
    assert result == 'key1=1, key2=2.5, key3=3'

    # Test case: dictionary with boolean values
    d = {'key1': True, 'key2': False}
    result = dict2str(d)
    assert result == 'key1=True, key2=False'


def test_list2str():

    # Test case: basic
    l = ['value1', 'value2', 'value3']
    result = list2str(l)
    assert result == 'value1, value2, value3'

    # Test case: empty list
    l2 = []
    result2 = list2str(l2)
    assert result2 == ''

    # Test case: list with one value
    l3 = ['value1']
    result3 = list2str(l3)
    assert result3 == 'value1'

    # Test case: list with numeric values
    l4 = [1, 2.5, 3]
    result4 = list2str(l4)
    assert result4 == '1, 2.5, 3'

    # Test case: list with boolean values
    l5 = [True, False]
    result5 = list2str(l5)
    assert result5 == 'True, False'
