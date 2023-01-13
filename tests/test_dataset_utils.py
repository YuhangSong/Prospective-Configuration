import pytest
import torch

from dataset_utils import *


def test_partial_dateset_v1():

    # create a mock dataset object with 10 datapoints and 2 classes
    class TestDataset:
        def __init__(self):
            self.data = torch.Tensor(
                [[0], [1], [2], [3], [4], [5], [6]]
            )
            self.targets = torch.Tensor(
                [0, 0, 0, 1, 1, 1, 1]
            )

    data_0 = [0, 1, 2]
    data_1 = [3, 4, 5, 6]

    def assert_partial_dataset(partial_dataset, target, data, num):

        assert partial_dataset.targets.size(0) == partial_dataset.data.size(0)

        unique_targets = torch.unique(
            partial_dataset.targets
        ).tolist()
        for unique_target in unique_targets:
            assert unique_target in [-1, 0, 1]

        for i in range(partial_dataset.targets.size(0)):
            assert partial_dataset.data[i][0] in data_0+data_1

        # assert if number of data points with <target> in <data> is <num>
        assert sum(
            (
                partial_dataset.targets[i] == target
            ) and (
                partial_dataset.data[i][0] in data
            ) for i in range(partial_dataset.targets.size(0))
        ) == num

    # Test case: default
    partial_dataset = partial_dateset_v1(TestDataset())
    assert_partial_dataset(partial_dataset, 0, data_0, 3)
    assert_partial_dataset(partial_dataset, -1, data_0, 0)
    assert_partial_dataset(partial_dataset, 1, data_1, 4)
    assert_partial_dataset(partial_dataset, -1, data_1, 0)

    # Test case: partial_targets
    partial_dataset = partial_dateset_v1(TestDataset(), partial_targets=[0])
    assert_partial_dataset(partial_dataset, 0, data_0, 3)
    assert_partial_dataset(partial_dataset, -1, data_0, 0)
    assert_partial_dataset(partial_dataset, 1, data_1, 0)
    assert_partial_dataset(partial_dataset, -1, data_1, 0)

    # Test case: partial_num
    partial_dataset = partial_dateset_v1(TestDataset(), partial_num=2)
    assert_partial_dataset(partial_dataset, 0, data_0, 2)
    assert_partial_dataset(partial_dataset, -1, data_0, 0)
    assert_partial_dataset(partial_dataset, 1, data_1, 2)
    assert_partial_dataset(partial_dataset, -1, data_1, 0)

    # Test case: unlabelled_ratio
    partial_dataset = partial_dateset_v1(TestDataset(), unlabelled_ratio=0.5)
    assert_partial_dataset(partial_dataset, 0, data_0, 2)
    assert_partial_dataset(partial_dataset, -1, data_0, 1)
    assert_partial_dataset(partial_dataset, 1, data_1, 2)
    assert_partial_dataset(partial_dataset, -1, data_1, 2)

    # Test case: all
    partial_dataset = partial_dateset_v1(
        TestDataset(), partial_targets=[1], partial_num=3, unlabelled_ratio=0.5
    )
    assert_partial_dataset(partial_dataset, 0, data_0, 0)
    assert_partial_dataset(partial_dataset, -1, data_0, 0)
    assert_partial_dataset(partial_dataset, 1, data_1, 2)
    assert_partial_dataset(partial_dataset, -1, data_1, 1)


def test_map_dataset_targets():

    # create a mock dataset object with 4 datapoints and 3 classes
    class TestDataset:
        def __init__(self):
            self.data = torch.Tensor(
                [[0], [1], [2], [3]]
            )
            self.targets = torch.Tensor(
                [0, 1, 0, 2]
            )

    # Test case: mapper is None
    dataset = TestDataset()
    mapper = None
    mapped_dataset = map_dataset_targets(dataset, mapper)
    assert mapped_dataset == dataset

    # Test case: mapper is a dict
    dataset = TestDataset()
    mapper = {0: 1, 1: 0, 2: 2}
    mapped_dataset = map_dataset_targets(dataset, mapper)
    assert (mapped_dataset.data == torch.tensor([[0], [1], [2], [3]])).all()
    assert (mapped_dataset.targets == torch.tensor([1, 0, 1, 2])).all()


def test_data_loader_fn_defualt():

    data_loader = data_loader_fn()
    num_batches = 0
    for _, (data, target) in enumerate(data_loader):
        assert data.size(0) == 4
        assert data.size(1) == 2
        assert target.size(0) == 4
        assert target.size(1) == 1
        num_batches += 1
    assert num_batches == 1


def test_data_loader_fn_dataset_name_MNIST():

    data_loader = data_loader_fn(
        dataset_name="MNIST",
        batch_size=600,
    )
    num_batches = 0
    for _, (data, target) in enumerate(data_loader):
        assert data.size(0) == 600
        assert data.size(1) == 784
        assert target.size(0) == 600
        assert target.size(1) == 10
        num_batches += 1
    assert num_batches == 98


def test_data_loader_fn_dataset_name_FashionMNIST():

    data_loader = data_loader_fn(
        dataset_name="FashionMNIST",
        batch_size=600,
    )
    num_batches = 0
    for _, (data, target) in enumerate(data_loader):
        assert data.size(0) == 600
        assert data.size(1) == 784
        assert target.size(0) == 600
        assert target.size(1) == 10
        num_batches += 1
    assert num_batches == 100


def test_data_loader_fn_dataset_name_CIFAR10():

    data_loader = data_loader_fn(
        dataset_name="CIFAR10",
        batch_size=600,
    )
    num_batches = 0
    for _, (data, target) in enumerate(data_loader):
        assert data.size(0) == 600
        assert data.size(1) == 784
        assert target.size(0) == 600
        assert target.size(1) == 10
        num_batches += 1
    assert num_batches == 83


def test_data_loader_fn_train():

    data_loader = data_loader_fn(
        dataset_name="MNIST",
        train=False,
        batch_size=100,
    )
    num_batches = 0
    for _, (data, target) in enumerate(data_loader):
        assert data.size(0) == 100
        assert data.size(1) == 784
        assert target.size(0) == 100
        assert target.size(1) == 10
        num_batches += 1
    assert num_batches == 100


def test_data_loader_fn_data_image_size():

    data_loader = data_loader_fn(
        dataset_name="CIFAR10",
        data_image_size=5,
    )
    for _, (data, target) in enumerate(data_loader):
        assert data.size()[1:] == torch.Size([25])


def test_data_loader_fn_data_is_gray():

    data_loader = data_loader_fn(
        dataset_name="CIFAR10",
        data_is_gray=False,
    )
    for _, (data, target) in enumerate(data_loader):
        assert data.size()[1:] == torch.Size([2352])


def test_data_loader_fn_data_is_flatten():

    data_loader = data_loader_fn(
        dataset_name="CIFAR10",
        data_is_flatten=False,
    )
    for _, (data, target) in enumerate(data_loader):
        assert data.size()[1:] == torch.Size([1, 28, 28])


def test_data_loader_fn_data_is_gray_is_flatten():

    data_loader = data_loader_fn(
        dataset_name="CIFAR10",
        data_is_gray=False,
        data_is_flatten=False,
    )
    for _, (data, target) in enumerate(data_loader):
        assert data.size()[1:] == torch.Size([3, 28, 28])


def test_data_loader_fn_target_min_max():

    data_loader = data_loader_fn(
        target_min=-0.2842,
        target_max=1.3526,
    )
    for _, (data, target) in enumerate(data_loader):
        assert target.min() == -0.2842
        assert target.max() == 1.3526
