import pytest

import os
import tempfile
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import uuid

from utils import *


def test_init_xavier_uniform_reciprocal_friendly_():

    f_in = 1
    f_out = 4
    C = 0.1
    gain = 2.0

    w = torch.empty(f_out, f_in)

    w_fb_product = init_xavier_uniform_reciprocal_friendly_(
        w, gain=gain, C=C, is_test=True
    )

    w = w.view(-1)

    w_positive = w[w > 0]
    assert torch.allclose(
        torch.max(
            w_positive
        ),
        torch.max(
            w_fb_product/w_positive
        )
    )
    assert torch.allclose(
        torch.min(
            w_positive
        ),
        torch.min(
            w_fb_product/w_positive
        )
    )

    w_negative = w[w < 0]
    assert torch.allclose(
        torch.max(
            w_negative
        ),
        torch.max(
            w_fb_product/w_negative
        )
    )
    assert torch.allclose(
        torch.min(
            w_negative
        ),
        torch.min(
            w_fb_product/w_negative
        )
    )


def test_init_yuhang():

    f_in = 100
    f_out = 20
    C = 0.1
    A = 3.2

    w = torch.empty(f_out, f_in)

    init_yuhang(w, C=C, A=A)

    w = w.view(-1)

    w_positive = w[w > 0]
    assert w_positive.size(0) > 0
    assert w_positive.dim() == 1
    assert w_positive.max() < A/f_in
    assert w_positive.min() > A/f_in*C
    w_negative = w[w < 0]
    assert w_negative.size(0) > 0
    assert w_negative.dim() == 1
    assert w_negative.max() < -A/f_in*C
    assert w_negative.min() > -A/f_in


def test_get_error_undershot_only():
    target = torch.tensor([[1.0, 2.0, 1.0], [4.0, 3.0, 3.0]])
    prediction = torch.tensor([[1.5, 1.5, 1.5], [4.5, 2.0, 3.5]])
    expected_output = torch.tensor([[-0.5, 0.5, -0.5], [0.0, 0.0, -0.5]])
    assert torch.allclose(
        get_error_undershot_only(target=target, prediction=prediction),
        expected_output
    )


def test_dict_values():
    test_dict = {
        'a': 1,
        'b': {
            'c': 2,
            'd': {
                'e': 3
            }
        },
        'f': 4
    }
    expected_output = [1, 2, 3, 4]
    assert list(dict_values(test_dict)) == expected_output


def test_hardcode_w_update_pre():

    w = torch.randn(3, 4)
    x_pre = torch.randn(3, 3)
    e_post = torch.randn(3, 4)
    lr = 0.1
    acf = torch.nn.Sigmoid()
    acf_at = 'pre'

    w_init = w.clone()

    expected_update = - torch.bmm(
        acf(x_pre).unsqueeze(2),
        e_post.unsqueeze(1)
    ).sum(dim=0).mul(lr)

    hardcode_w_update(w, x_pre, e_post, lr, acf, acf_at)

    assert not torch.allclose(w_init, w)

    assert torch.allclose(w_init + expected_update, w)


def test_hardcode_w_update_pre_optimizer():

    w_optimizer_update = torch.nn.parameter.Parameter(torch.randn(3, 4))
    w_hardcode_w_update = w_optimizer_update.data.clone()
    x_pre = torch.randn(3, 3)
    target = torch.randn(3, 4)
    lr = 0.1
    acf = torch.nn.Sigmoid()
    acf_at = 'pre'

    def forward_fn(w):
        return torch.matmul(acf(x_pre), w)

    optimizer = torch.optim.SGD([w_optimizer_update], lr=lr, weight_decay=0.01)
    optimizer.zero_grad()
    loss = (forward_fn(w_optimizer_update)-target).pow(2).sum()*0.5
    loss.backward()
    optimizer.step()

    hardcode_w_update(w_hardcode_w_update, x_pre,
                      forward_fn(w_hardcode_w_update)-target, lr, acf, acf_at, weight_decay=0.01)

    assert torch.allclose(w_optimizer_update.data, w_hardcode_w_update)


def test_hardcode_w_update_post():

    w = torch.randn(3, 4)
    x_pre = torch.randn(3, 3)
    e_post = torch.randn(3, 4)
    lr = 0.1
    acf = torch.nn.Sigmoid()
    acf_at = 'post'

    w_init = w.clone()

    expected_update = - torch.bmm(
        x_pre.unsqueeze(2),
        (e_post * grad(acf)(torch.matmul(x_pre, w))).unsqueeze(1)
    ).sum(dim=0).mul(lr)

    hardcode_w_update(w, x_pre, e_post, lr, acf, acf_at)

    assert not torch.allclose(w_init, w)

    assert torch.allclose(w_init + expected_update, w)


def test_hardcode_w_update_post_optimizer():

    w_optimizer_update = torch.nn.parameter.Parameter(torch.randn(3, 4))
    w_hardcode_w_update = w_optimizer_update.data.clone()
    x_pre = torch.randn(3, 3)
    target = torch.randn(3, 4)
    lr = 0.1
    acf = torch.nn.Sigmoid()
    acf_at = 'post'

    def forward_fn(w):
        return acf(torch.matmul(x_pre, w))

    optimizer = torch.optim.SGD([w_optimizer_update], lr=lr)
    optimizer.zero_grad()
    loss = (forward_fn(w_optimizer_update)-target).pow(2).sum()*0.5
    loss.backward()
    optimizer.step()

    hardcode_w_update(w_hardcode_w_update, x_pre,
                      forward_fn(w_hardcode_w_update)-target, lr, acf, acf_at)

    assert torch.allclose(w_optimizer_update.data, w_hardcode_w_update)


def test_hardcode_w_update_both():

    w = torch.randn(3, 4)
    x_pre = torch.randn(3, 3)
    e_post = torch.randn(3, 4)
    lr = 0.1
    acf = torch.nn.Sigmoid()
    acf_at = 'both'

    w_init = w.clone()

    expected_update = - torch.bmm(
        acf(x_pre).unsqueeze(2),
        (e_post * grad(acf)(torch.matmul(acf(x_pre), w))).unsqueeze(1)
    ).sum(dim=0).mul(lr)

    hardcode_w_update(w, x_pre, e_post, lr, acf, acf_at)

    assert not torch.allclose(w_init, w)

    assert torch.allclose(w_init + expected_update, w)


def test_hardcode_w_update_both_optimizer():

    w_optimizer_update = torch.nn.parameter.Parameter(torch.randn(3, 4))
    w_hardcode_w_update = w_optimizer_update.data.clone()
    x_pre = torch.randn(3, 3)
    target = torch.randn(3, 4)
    lr = 0.1
    acf = torch.nn.Sigmoid()
    acf_at = 'both'

    def forward_fn(w):
        return acf(torch.matmul(acf(x_pre), w))

    optimizer = torch.optim.SGD([w_optimizer_update], lr=lr)
    optimizer.zero_grad()
    loss = (forward_fn(w_optimizer_update) -
            target).pow(2).sum()*0.5
    loss.backward()
    optimizer.step()

    hardcode_w_update(w_hardcode_w_update, x_pre,
                      forward_fn(w_hardcode_w_update)-target, lr, acf, acf_at)

    assert torch.allclose(w_optimizer_update.data, w_hardcode_w_update)


def test_get_lists_overlap():
    # Test case with lists that have overlap
    list1 = [1, 2, 3, 4, 5]
    list2 = [4, 5, 6, 7, 8]
    expected_output = [4, 5]
    assert get_lists_overlap(list1, list2) == expected_output

    # Test case with lists that have no overlap
    list1 = [1, 2, 3, 4, 5]
    list2 = [6, 7, 8, 9, 10]
    expected_output = []
    assert get_lists_overlap(list1, list2) == expected_output

    # Test case with empty lists
    list1 = []
    list2 = []
    expected_output = []
    assert get_lists_overlap(list1, list2) == expected_output

    # Test case with duplicate elements
    list1 = [1, 2, 2, 3, 4, 5]
    list2 = [5, 5, 4, 4, 4, 4]
    expected_output = [4, 5]
    assert get_lists_overlap(list1, list2) == expected_output

    # Test case with non-list inputs
    list1 = "1, 2, 3, 4, 5"
    list2 = [6, 7, 8, 9, 10]
    with pytest.raises(AssertionError) as error:
        get_lists_overlap(list1, list2)
    assert str(error.value) == "Expected list1 to be a list, but got <class 'str'>"


def test_is_lists_overlap():
    # Test case with overlapping lists
    list1 = [1, 2, 3, 4]
    list2 = [3, 4, 5, 6]
    assert is_lists_overlap(list1, list2) == True

    # Test case with non-overlapping lists
    list1 = [1, 2, 3, 4]
    list2 = [5, 6, 7, 8]
    assert is_lists_overlap(list1, list2) == False

    # Test case with empty lists
    list1 = []
    list2 = []
    assert is_lists_overlap(list1, list2) == False

    # Test case with one list is empty
    list1 = [1, 2, 3, 4]
    list2 = []
    assert is_lists_overlap(list1, list2) == False

    # Test case where inputs are not lists
    list1 = 'not a list'
    list2 = [5, 6, 7, 8]
    with pytest.raises(AssertionError):
        is_lists_overlap(list1, list2)


def test_reflect_scale():
    scale = 'MNIST/28/True/True/-0.1/1.0/10/6000/32/1000/10/accuracy'
    expected_data_loader_kwargs = {
        'dataset_name': 'MNIST',
        'data_image_size': 28,
        'data_is_gray': True,
        'data_is_flatten': True,
        'target_min': -0.1,
        'target_max': 1.0,
        'partial_targets_num': 10,
        'partial_num': 6000,
        'batch_size': 32,
    }
    expected_scheduler_kwargs = {
        'max_t': 1000,
        'grace_period': 10,
        'metric': 'accuracy',
    }
    data_loader_kwargs, scheduler_kwargs = reflect_scale(scale)
    assert data_loader_kwargs == expected_data_loader_kwargs
    assert scheduler_kwargs == expected_scheduler_kwargs


def test_get_file_extensions_in_dir():

    # Test the case where the directory is empty
    with tempfile.TemporaryDirectory() as tempdir:
        extensions = get_file_extensions_in_dir(tempdir)
        assert len(extensions) == 0

    # Test the case where the directory has one file with an extension
    with tempfile.TemporaryDirectory() as tempdir:
        test_file = 'test.txt'
        open(os.path.join(tempdir, test_file), 'w').close()
        extensions = get_file_extensions_in_dir(tempdir)
        assert len(extensions) == 1
        assert os.path.splitext(test_file)[1] in extensions

    # Test the case where the directory has multiple files with different extensions
    with tempfile.TemporaryDirectory() as tempdir:
        test_files = [
            'test1.txt',
            'test2.csv',
            'test3.pdf',
            'test4.png',
            'test5.docx',
        ]
        for test_file in test_files:
            open(os.path.join(tempdir, test_file), 'w').close()

        extensions = get_file_extensions_in_dir(tempdir)

        for test_file in test_files:
            extension = os.path.splitext(test_file)[1]
            assert extension in extensions

        assert len(extensions) == len(test_files)

    # Test the case where the directory has multiple files with the same extension
    with tempfile.TemporaryDirectory() as tempdir:
        test_files = [
            'test1.txt',
            'test2.txt',
            'test3.txt',
            'test4.txt',
            'test5.txt',
        ]
        for test_file in test_files:
            open(os.path.join(tempdir, test_file), 'w').close()

        extensions = get_file_extensions_in_dir(tempdir)

        for test_file in test_files:
            extension = os.path.splitext(test_file)[1]
            assert extension in extensions

        assert len(extensions) == 1

    # Test the case where the given path is not a directory
    with tempfile.TemporaryDirectory() as tempdir:
        test_file = 'test.txt'
        open(os.path.join(tempdir, test_file), 'w').close()
        with pytest.raises(AssertionError):
            get_file_extensions_in_dir(os.path.join(tempdir, test_file))

    # Test the case where there are subdirectories and files in the subdirectories
    with tempfile.TemporaryDirectory() as tempdir:
        subdir1 = os.path.join(tempdir, 'subdir1')
        subdir2 = os.path.join(tempdir, 'subdir2')
        os.makedirs(subdir1)
        os.makedirs(subdir2)

        subdir1_files = [
            'test1.txt',
            'test2.csv',
            'test3.pdf',
        ]
        for subdir1_file in subdir1_files:
            open(os.path.join(subdir1, subdir1_file), 'w').close()

        subdir2_files = [
            'test1.png',
            'test2.docx',
            'test3.xlsx',
        ]
        for subdir2_file in subdir2_files:
            open(os.path.join(subdir2, subdir2_file), 'w').close()

        extensions = get_file_extensions_in_dir(tempdir)

        for subdir1_file in subdir1_files:
            extension = os.path.splitext(subdir1_file)[1]
            assert extension in extensions

        for subdir2_file in subdir2_files:
            extension = os.path.splitext(subdir2_file)[1]
            assert extension in extensions

        assert len(extensions) == len(subdir1_files + subdir2_files)

    # Test the case with ignore_empty_extension=True
    with tempfile.TemporaryDirectory() as tempdir:
        test_files = [
            'test1.txt',
            'test2',
            'test3.pdf',
        ]
        for test_file in test_files:
            open(os.path.join(tempdir, test_file), 'w').close()

        extensions = get_file_extensions_in_dir(
            tempdir, ignore_empty_extension=True
        )

        test_files.remove('test2')
        for test_file in test_files:
            extension = os.path.splitext(test_file)[1]
            assert extension in extensions

        assert len(extensions) == len(test_files)


def test_get_file_extension():
    assert get_file_extension('myfile.txt') == '.txt'
    assert get_file_extension('myfile.pdf') == '.pdf'
    assert get_file_extension('myfile') == ''
    assert get_file_extension('/path/to/myfile.zip') == '.zip'
    assert get_file_extension('/path/to/myfile') == ''


def test_hashify():

    # Test case: basic
    x = [1, 2, 3]
    expected_output = "[1, 2, 3]"
    assert hashify(x) == expected_output

    # Test case: tuple input
    x = (1, 2, 3)
    expected_output = (1, 2, 3)
    assert hashify(x) == expected_output

    # Test case: string input
    x = "abc"
    expected_output = "abc"
    assert hashify(x) == expected_output

    # Test case: integer input
    x = 123
    expected_output = 123
    assert hashify(x) == expected_output

    # Test case: float input
    x = 123.456
    expected_output = 123.456
    assert hashify(x) == expected_output


def test_assert_frame_equal():

    # Test case: basic
    df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df2 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    assert_frame_equal(df1, df2)
    assert_frame_equal(df1, df2, ignore_order=True)
    assert_frame_equal(df1, df2, ignore_order=False)

    # Test case: different index
    df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}, index=[1, 2, 3])
    df2 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}, index=[4, 5, 6])
    assert_frame_equal(df1, df2)
    assert_frame_equal(df1, df2, ignore_order=True)
    assert_frame_equal(df1, df2, ignore_order=False)

    # Test case: different order of rows
    df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df2 = pd.DataFrame({'a': [2, 1, 3], 'b': [5, 4, 6]})
    assert_frame_equal(df1, df2)
    assert_frame_equal(df1, df2, ignore_order=True)
    with pytest.raises(AssertionError):
        assert_frame_equal(df1, df2, ignore_order=False)

    # Test case: different order of columns
    df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df2 = pd.DataFrame({'b': [4, 5, 6], 'a': [1, 2, 3]})
    assert_frame_equal(df1, df2)
    assert_frame_equal(df1, df2, ignore_order=True)
    with pytest.raises(AssertionError):
        assert_frame_equal(df1, df2, ignore_order=False)

    # Test case: different order of rows and columns
    df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df2 = pd.DataFrame({'b': [5, 4, 6], 'a': [2, 1, 3]})
    assert_frame_equal(df1, df2)
    assert_frame_equal(df1, df2, ignore_order=True)
    with pytest.raises(AssertionError):
        assert_frame_equal(df1, df2, ignore_order=False)

    # Test case: different order of rows and columns, but values are not matched
    df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df2 = pd.DataFrame({'b': [5, 4, 6], 'a': [1, 2, 3]})
    with pytest.raises(AssertionError):
        assert_frame_equal(df1, df2)
    with pytest.raises(AssertionError):
        assert_frame_equal(df1, df2, ignore_order=True)
    with pytest.raises(AssertionError):
        assert_frame_equal(df1, df2, ignore_order=False)

    # Test case: with unhashable elements
    df1 = pd.DataFrame({'a': [[1, 2], [3]], 'b': [[4, 5], [6]]})
    df2 = pd.DataFrame({'a': [[1, 2], [3]], 'b': [[4, 5], [6]]})
    assert_frame_equal(df1, df2)
    assert_frame_equal(df1, df2, ignore_order=True)
    assert_frame_equal(df1, df2, ignore_order=False)


def test_concatenate_dicts():

    # Test case: basic
    dict_list = [{'a': 1, 'b': 2}, {'c': 3, 'd': 4}]
    expected_output = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    assert concatenate_dicts(dict_list) == expected_output

    # Test case: empty list
    dict_list = [{}, {}, {}]
    expected_output = {}
    assert concatenate_dicts(dict_list) == expected_output

    # Test case: with overlapping keys
    dict_list = [{'a': 1, 'b': 2}, {'b': 3, 'c': 4}]
    expected_output = {'a': 1, 'b': 3, 'c': 4}
    assert concatenate_dicts(dict_list) == expected_output

    # Test case: with non dict input
    dict_list = [{'a': 1, 'b': 2}, 'c']
    with pytest.raises(AssertionError):
        concatenate_dicts(dict_list)


def test_fig_to_pil():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    image = fig_to_pil(fig)
    assert isinstance(image, Image.Image)


def test_fig_to_pil_with_empty_figure():
    fig, ax = plt.subplots()
    image = fig_to_pil(fig)
    assert isinstance(image, Image.Image)


def test_fig_to_pil_with_invalid_input():
    with pytest.raises(AssertionError):
        fig_to_pil('foo')


def test_fig_to_numpy():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    arr = fig_to_numpy(fig)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3, 480, 640)


def test_fig_to_numpy_with_empty_figure():
    fig, ax = plt.subplots()
    arr = fig_to_numpy(fig)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3, 480, 640)


def test_fig_to_numpy_with_invalid_input():
    with pytest.raises(AssertionError):
        fig_to_numpy('foo')


def test_remove_items():

    # Test case: basic
    list_ = [1, 2, 3, 4, 5]
    list_remove = [3, 5]
    expected_output = [1, 2, 4]
    assert remove_items(list_, list_remove) == expected_output

    # Test case: remove non-existent items
    list_ = [1, 2, 3, 4, 5]
    list_remove = [6, 7]
    expected_output = [1, 2, 3, 4, 5]
    assert remove_items(list_, list_remove) == expected_output

    # Test case: remove all items
    list_ = [1, 2, 3, 4, 5]
    list_remove = [1, 2, 3, 4, 5]
    expected_output = []
    assert remove_items(list_, list_remove) == expected_output

    # Test case: with in_place=True
    list_ = [1, 2, 3, 4, 5]
    list_remove = [3, 5]
    expected_output = [1, 2, 4]
    assert remove_items(list_, list_remove, in_place=True) == expected_output
    assert list_ == expected_output


def test_remove_all_items():
    list_ = [1, 2, 3, 4, 5]
    list_remove = [1, 2, 3, 4, 5]
    expected_output = []
    assert remove_items(list_, list_remove) == expected_output


@pytest.fixture
def temp_dir_and_files():
    temp_dir = tempfile.TemporaryDirectory()
    path = temp_dir.name
    extensions = ["txt", "pdf"]
    files = ["file1.txt", "file2.pdf", "file3.txt", "file4.pdf"]
    for file_ in files:
        open(os.path.join(path, file_), "w").close()
    return temp_dir, path, extensions, files


def test_clean_files_with_extension(temp_dir_and_files):
    temp_dir, path, extensions, files = temp_dir_and_files
    # Test cleaning files with the given extensions
    clean_files_with_extension(path, extensions)
    remaining_files = os.listdir(path)
    assert len(remaining_files) == 0
    temp_dir.cleanup()


def test_torch_max_onehot():
    # Test with a 2D tensor
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    expected = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    onehot = torch_max_onehot(x, dim=1)
    assert torch.allclose(onehot, expected)


def test_get_classification_error():
    # Test binary classification
    prediction = torch.tensor([[0.1], [0.9], [0.8], [0.2]])
    target = torch.tensor([[0], [1], [1], [1]])
    error = get_classification_error(prediction, target, is_binary=True)
    assert error == 0.25

    # Test multi-class classification
    prediction = torch.tensor([[0.1, 0.9, 0.0], [0.9, 0.1, 0.0], [
                              0.8, 0.2, 0.0], [0.2, 0.8, 0.0]])
    target = torch.tensor([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    error = get_classification_error(prediction, target)
    assert error == 0.75

    # Test binary classification with non-default threshold
    prediction = torch.tensor([[0.1], [0.9], [0.8], [0.2]])
    target = torch.tensor([[0], [1], [1], [1]])
    error = get_classification_error(
        prediction, target, is_binary=True, binary_threshold=0.15)
    assert error == 0.0


def test_tensordataset_data_loader():

    input_ = [[1, 2], [3, 4]]
    target_ = [[5, 6], [7, 8]]
    batch_size = 2

    # Test with default noise_std
    data_loader = tensordataset_data_loader(
        input_, target_, batch_size=batch_size
    )
    assert isinstance(data_loader, DataLoader)
    assert len(data_loader) == 1
    input_batch, target_batch = next(iter(data_loader))
    assert input_batch.tolist() == input_
    assert target_batch.tolist() == target_

    # Test with specified noise_std
    data_loader = tensordataset_data_loader(
        input_, target_, noise_std=0.1, batch_size=batch_size,
    )
    assert isinstance(data_loader, DataLoader)
    assert len(data_loader) == 1
    input_batch, target_batch = next(iter(data_loader))
    assert input_batch.tolist() != input_
    assert target_batch.tolist() != target_


def test_torch_normal():
    # Test std = 0.0
    mean = torch.tensor(0.0)
    std = 0.0
    result = torch_normal(mean, std)
    assert result == mean, f'Error: {result}'

    # Test std > 0.0
    mean = torch.tensor(0.0)
    std = 1.0
    result = torch_normal(mean, std)
    assert isinstance(result, torch.Tensor), f'Error: {result}'
    assert result != mean, f'Error: {result}'


def test_np_idx2onehot():
    # Test conversion of a labelled index
    idx = 2
    size = 5
    onehot = np_idx2onehot(idx, size)
    expected = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    assert np.allclose(onehot, expected)

    # Test conversion of an unlabelled index
    idx = -1
    size = 5
    onehot = np_idx2onehot(idx, size)
    expected = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    assert np.allclose(onehot, expected)

    # Test conversion with a custom fill function for unlabelled index
    def fill_fn(size):
        return 1.0 / size**2
    idx = -1
    size = 5
    onehot = np_idx2onehot(idx, size, fill_fn)
    expected = np.array([0.04, 0.04, 0.04, 0.04, 0.04])
    assert np.allclose(onehot, expected)

    # Test conversion with a non-default target range
    idx = 2
    size = 5
    onehot = np_idx2onehot(idx, size, target_min=-0.233184, target_max=0.6324)
    expected = np.array([-0.233184, -0.233184, 0.6324, -0.233184, -0.233184])
    assert np.allclose(onehot, expected)


def test_grad():
    # Test that the grad function returns the correct gradient
    def f(x):
        return x**2
    grad_f = grad(f)
    x = torch.tensor([2.0], requires_grad=True)
    y = f(x)
    y.backward()
    assert torch.allclose(x.grad, grad_f(x))

    # Test that the grad function works with multiple inputs
    def g(x):
        return x[0]**2 + x[1]**2
    grad_g = grad(g)
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    y = g(x)
    y.backward()
    assert torch.allclose(x.grad, grad_g(x))


def test_sigmoid_inverse():
    x = torch.rand(3, 3)
    assert torch.allclose(
        sigmoid_inverse(
            torch.nn.functional.sigmoid(x),
            eps=0.0,
        ),
        x,
    )


def test_hardtanh_inverse():
    x = torch.rand(3, 3)
    assert torch.allclose(
        hardtanh_inverse(
            torch.nn.functional.hardtanh(x),
            eps=0.0,
        ),
        x,
    )


def test_tanh_inverse():
    x = torch.rand(3, 3)
    assert torch.allclose(
        tanh_inverse(
            torch.nn.functional.tanh(x),
            eps=0.0,
        ),
        x,
    )


def test_identity():
    # Test that identity returns the input unchanged
    x = torch.randn(10, 10)
    assert torch.allclose(x, identity(x))


def test_identity_inverse():
    # Test that identity_inverse returns the input unchanged
    x = torch.randn(10, 10)
    assert torch.allclose(x, identity_inverse(x))


def test_SCO():
    # Test that SCO applies the shift, scale, and offset correctly
    module = nn.Linear(10, 10)
    sco = SCO(module, s=1.0, c=2.0, o=3.0)
    x = torch.randn(5, 10)
    assert torch.allclose(module(x + 1.0) * 2.0 + 3.0, sco(x))

    # Test that SCO applies the shift, scale, and offset correctly for a non-linear module
    module = nn.Sigmoid()
    sco = SCO(module, s=1.0, c=2.0, o=3.0)
    x = torch.randn(5, 10)
    assert torch.allclose(module(x + 1.0) * 2.0 + 3.0, sco(x))


def test_flatten_dict():
    # Test a simple dictionary
    d = {'a': 1, 'b': 2, 'c': 3}
    assert flatten_dict(d) == d

    # Test a nested dictionary
    d = {'a': 1, 'b': {'c': 2, 'd': 3}, 'e': 4}
    expected = {'a': 1, 'b: c': 2, 'b: d': 3, 'e': 4}
    assert flatten_dict(d) == expected

    # Test a deeper nested dictionary
    d = {'a': 1, 'b': {'c': {'d': 2, 'e': 3}, 'f': 4}, 'g': 5}
    expected = {'a': 1, 'b: c: d': 2, 'b: c: e': 3, 'b: f': 4, 'g': 5}
    assert flatten_dict(d) == expected

    # Test a dictionary with a level_limit
    d = {'a': 1, 'b': {'c': {'d': 2, 'e': 3}, 'f': 4}, 'g': 5}
    expected = {'a': 1, 'b: c': {'d': 2, 'e': 3}, 'b: f': 4, 'g': 5}
    assert flatten_dict(d, level_limit=2) == expected

    # Test an empty dictionary
    d = {}
    assert flatten_dict(d) == d


def test_assert_config_all_valid():
    config = {"a": 1, "b": "hello", "c": True, "d": {"e": 2.5}}
    assert_config_all_valid(config)

    config = {"a": 1, "b": "hello", "c": [True], "d": {"e": 2.5}}
    with pytest.raises(AssertionError) as error:
        assert_config_all_valid(config)

    config = {"a": {"b": {"c": {"d": {"e": {"f": {"g": {"h": 1}}}}}}}}
    assert_config_all_valid(config)


# def test_controlled_sigmoid():
#     cs = ControlledSigmoid(l=2.0, o=0.5, c=3.0)
#     assert cs.slope == 2.0
#     assert cs.offset == 0.5
#     assert cs.scale == 3.0
#     input_tensor = torch.tensor([-1.0, 0.0, 1.0])
#     expected_output = torch.tensor([-1.5, 0.5, 1.5])
#     output = cs(input_tensor)
#     assert output.allclose(expected_output)


# def test_param_delta_fn():
#     param1 = torch.tensor([1.0, 2.0, 3.0])
#     param2 = torch.tensor([4.0, 5.0, 6.0])
#     expected_output = 5.196152422706632
#     output = param_delta_fn("key", param1, param2)
#     assert output == expected_output


def test_prepare_path():

    uid = str(uuid.uuid4())

    # Test creating a new path
    path = f'/tmp/test/new/path-{uid}'
    prepare_path(path)
    assert os.path.exists(os.path.dirname(path))

    # Test creating a nested path
    path = '/tmp/test/new/nested/path'
    prepare_path(path)
    assert os.path.exists(os.path.dirname(path))

    # Test a path that already exists
    path = f'/tmp/test/new/path-{uid}'
    prepare_path(path)
    assert os.path.exists(os.path.dirname(path))

    # Test creating a path with special characters
    path = f'/tmp/te#st/new/path-{uid}'
    prepare_path(path)
    assert os.path.exists(os.path.dirname(path))

    # Test creating a path with a file name
    path = f'/tmp/test/new/path-{uid}/file.txt'
    prepare_path(path)
    assert os.path.exists(os.path.dirname(path))


def test_map_list_by_dict():
    # Test a simple mapping
    l = ['a', 'b', 'c']
    d = {'a': 'A', 'b': 'B'}
    assert map_list_by_dict(l, d) == ['A', 'B', 'c']

    # Test a full mapping
    l = ['a', 'b', 'c']
    d = {'a': 'A', 'b': 'B', 'c': 'C'}
    assert map_list_by_dict(l, d) == ['A', 'B', 'C']

    # Test a partial mapping
    l = ['a', 'b', 'c']
    d = {'a': 'A', 'b': 'B'}
    assert map_list_by_dict(l, d) == ['A', 'B', 'c']

    # Test a mapping with no changes
    l = ['a', 'b', 'c']
    d = {'d': 'D', 'e': 'E'}
    assert map_list_by_dict(l, d) == ['a', 'b', 'c']

    # Test an empty list
    l = []
    d = {'a': 'A', 'b': 'B'}
    assert map_list_by_dict(l, d) == []

    # Test an empty dictionary
    l = ['a', 'b', 'c']
    d = {}
    assert map_list_by_dict(l, d) == ['a', 'b', 'c']


def test_prepare_dir():

    uid = str(uuid.uuid4())

    # Test creating a new directory
    dir = f'/tmp/test/new/dir-{uid}/'
    prepare_dir(dir)
    assert os.path.exists(dir)

    # Test creating a nested directory
    dir = f'/tmp/test/new/nested/dir-{uid}/'
    prepare_dir(dir)
    assert os.path.exists(dir)

    # Test a directory that already exists
    dir = f'/tmp/test/new/dir-{uid}/'
    prepare_dir(dir)
    assert os.path.exists(dir)

    # Test creating a directory with special characters
    dir = f'/tmp/te#st/new/dir-{uid}/'
    prepare_dir(dir)
    assert os.path.exists(dir)

    # Test creating a directory with a trailing slash
    dir = f'/tmp/test/new/dir-{uid}/'
    prepare_dir(dir)
    assert os.path.exists(dir)

    # Test creating a directory with a missing trailing slash
    dir = f'/tmp/test/new/dir-{uid}'
    prepare_dir(dir)
    assert os.path.exists(dir + '/')


def test_list_intersection():

    # Test case: lists with no intersection
    assert list_intersection([1, 2, 3], [4, 5, 6]) == []
    assert list_intersection([], [4, 5, 6]) == []
    assert list_intersection([1, 2, 3], []) == []

    # Test case: lists with intersection
    assert list_intersection([1, 2, 3], [3, 2, 1]) == [1, 2, 3]
    assert list_intersection([1, 2, 2, 3], [2, 2, 3, 4]) == [2, 3]

    # Test case: lists with duplicates
    assert list_intersection([1, 1, 2, 3], [3, 2, 1, 1]) == [1, 2, 3]
