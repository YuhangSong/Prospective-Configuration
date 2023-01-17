import os
import random

import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

import utils as u

logger = u.getLogger(__name__)


class AngleDataset(TensorDataset):

    def __init__(
        self,
        num_samples=100,
        num_targets=2,
        is_positive_only=False,
    ):

        assert isinstance(num_samples, int), (
            f"num_samples must be an integer, not {type(num_samples)}"
        )
        assert num_samples > 0, (
            f"num_samples must be greater than 0, not {num_samples}"
        )

        assert isinstance(num_targets, int), (
            f"num_targets must be an integer, not {type(num_targets)}"
        )
        assert num_targets > 0, (
            f"num_targets must be greater than 0, not {num_targets}"
        )

        assert isinstance(is_positive_only, bool), (
            f"is_positive_only must be a boolean, not {type(is_positive_only)}"
        )

        def get_target(angle):

            target = torch.zeros(num_targets)

            if not is_positive_only:
                if angle < 0:
                    angle = 180 + angle
                target[
                    int(angle / (180/num_targets))
                ] = 1
            else:
                if angle > 45:
                    angle = angle - 45
                target[
                    int(angle / (45/num_targets))
                ] = 1

            return target

        datas = []
        targets = []

        for i in range(num_samples):

            if not is_positive_only:
                x = np.random.uniform(-1, 1)
                y = np.random.uniform(-1, 1)
            else:
                x = np.random.uniform(0, 1)
                y = np.random.uniform(0, 1)

            datas.append(torch.Tensor([x, y]))

            angle = np.rad2deg(np.arctan2(y, x))

            targets.append(get_target(angle))

        super().__init__(
            torch.stack(datas),
            torch.stack(targets),
        )


def pre_dataset(dataset):
    '''Unify the dataset to data (Tensor) and targets (Tensor).
        This is for processing datasets in unified code.
        Note that targets are idx instead of onehot.
    '''

    if type(dataset) in [datasets.SVHN]:
        dataset.data = torch.Tensor(dataset.data)
        dataset.targets = torch.Tensor(dataset.labels)
    elif type(dataset) in [datasets.CIFAR10, datasets.CIFAR100]:
        dataset.data = torch.Tensor(dataset.data)
        dataset.targets = torch.Tensor(dataset.targets)

    assert hasattr(dataset, 'data')
    assert hasattr(dataset, 'targets')
    assert isinstance(dataset.data, torch.Tensor)
    assert isinstance(dataset.targets, torch.Tensor)

    assert dataset.data.dim() >= 2
    assert dataset.targets.dim() == 1
    assert dataset.data.size(0) == dataset.targets.size(0)

    return dataset


def post_dataset(dataset):
    '''Reverse of pre_dataset.
        This converts dataset from the unified format to its original format.
    '''

    if type(dataset) in [datasets.SVHN]:
        dataset.data = dataset.data.byte().numpy()
        dataset.labels = dataset.targets.long().numpy()
    elif type(dataset) in [datasets.CIFAR10, datasets.CIFAR100]:
        dataset.data = dataset.data.byte().numpy()
        dataset.targets = dataset.targets.long().tolist()

    return dataset


def partial_dateset_v1(dataset, partial_targets=None, partial_num=None, unlabelled_ratio=0.0):
    """Select a subset of the dataset.

        dataset: the original dataset to create the partial dataset from.
        partial_targets: a list of targets (classes) to include in the partial dataset. If set to None, all targets will be included.
        partial_num: the number of data points to include for each target in the partial dataset. If set to None, all data points for each target will be included.
        unlabelled_ratio: a float value between 0.0 and 1.0 indicating the ratio of unlabelled data points to include in the partial dataset. Unlabelled data points are indicated with a target value of -1.
    """

    dataset = pre_dataset(dataset)

    original_dataset_unique_targets = torch.unique(dataset.targets).tolist()

    if partial_targets is None:
        # None means not applied, so get all targets and set as partial_targets
        partial_targets = original_dataset_unique_targets

    assert isinstance(partial_targets, list), 'partial_targets must be a list'
    for partial_target in partial_targets:
        assert partial_target in original_dataset_unique_targets, (
            f'partial_targets must be a subset of the original dataset targets, but {partial_target} is not in the original dataset targets {original_dataset_unique_targets}'
        )

    assert (
        isinstance(partial_num, int) and partial_num > 0
    ) or (partial_num is None), f'partial_num must be a positive integer or None, but got {partial_num}'

    assert isinstance(
        unlabelled_ratio, float
    ) and (0 <= unlabelled_ratio <= 1), (
        f'unlabelled_ratio must be a float value between 0.0 and 1.0, but got {unlabelled_ratio}'
    )

    partial_target_num_targets = []
    partial_target_num_datas = []

    for partial_target in partial_targets:

        # mask after applies partial_target
        partial_target_mask = (dataset.targets == partial_target)
        # partial_target_mask: tensor([ True, False, False,  ..., False, False, False])

        # idx after applies partial_target
        partial_target_idx = partial_target_mask.nonzero(
            as_tuple=False
        ).squeeze(1)
        # partial_target_idx: tensor([    1,     2,     4,  ..., 59974, 59985, 59998])

        if partial_num is not None:
            if partial_num < partial_target_idx.size()[0]:
                partial_num_ = partial_num
            else:
                partial_num_ = None
        else:
            partial_num_ = partial_num

        # idx after applies partial_num and partial_target
        partial_target_num_idx = partial_target_idx[
            torch.randperm(
                partial_target_idx.size()[0]
            )[:partial_num_]
        ]
        # partial_target_num_idx: tensor([43017, 45843, 43440,  3620, 25991,  6417, 40425, 38389,  5477, 46967,
        # 54384, 43164, 32175, 13065, 36788, 24928, 59168, 17996, 21545, 55107,
        # 56483, 26094, 25783,  7961, 10844, 37258, 33559, 48178, 23700, 53589,
        # 53788, 59204, 40906, 38727, 11447, 20832, 17366, 55911,  3644, 57911,
        # 23990, 39825, 25863,  4492, 55385, 33223, 57882, 31879, 35712, 34674])

        # mask after applies partial_num and partial_target
        partial_target_num_mask = torch.zeros(
            dataset.targets.size()
        ).bool().fill_(False)
        partial_target_num_mask[partial_target_num_idx] = True
        # partial_target_num_mask: tensor([ True, False, False,  ..., False, False, False])

        # the targets selected by partial_target_num_mask
        partial_target_num_target = dataset.targets[
            partial_target_num_mask
        ].clone()
        partial_target_num_data = dataset.data[
            partial_target_num_mask
        ].clone()

        if unlabelled_ratio > 0.0:
            # fill some targets with -1, indicating that they are unlabelled datapoints
            partial_target_num_target[
                # these unlabelled datapoints are randomly selected
                torch.randperm(
                    partial_target_num_target.size(0)
                )[
                    # the number of these unlabelled datapoints is determined by unlabelled_ratio
                    :int(unlabelled_ratio * partial_target_num_target.size(0))
                ]
            ] = -1.0

        # append the selected targets and data
        partial_target_num_targets.append(partial_target_num_target)
        partial_target_num_datas.append(partial_target_num_data)

    dataset.targets = torch.cat(partial_target_num_targets, dim=0)
    dataset.data = torch.cat(partial_target_num_datas, dim=0)

    dataset = post_dataset(dataset)

    return dataset


def partial_dateset(dataset, partial_targets=None, partial_num=-1, unlabelled_ratio=0.0):
    """Use only part of the dataset.

    Args:
        dataset (Dataset): The dataset.

        The following arguments are applied in a nested manner.

        partial_targets (list): A list of targets. The returned dataset only contains these targets.
            None means not applied.
        partial_num (int): The number of datapoints to extract from each class.
            Negative value means not applied.
        unlabelled_ratio (float): The ratio of unlabelled datapoints.
    """

    logger.warning(
        "partial_dateset is deprecated, use partial_dataset_v1 instead."
    )

    dataset = pre_dataset(dataset)

    if partial_targets is None:
        # None means not applied, so get all targets and set as partial_targets
        partial_targets = torch.unique(dataset.targets).tolist()

    else:

        assert isinstance(partial_targets, list)

    assert isinstance(partial_num, int)

    targets = []
    data = []

    for partial_target in partial_targets:

        # mask after applies partial_target
        partial_target_mask = (dataset.targets == partial_target)
        # partial_target_mask: tensor([ True, False, False,  ..., False, False, False])

        # idx after applies partial_target
        partial_target_idx = partial_target_mask.nonzero(
            as_tuple=False
        ).squeeze(1)
        # partial_target_idx: tensor([    1,     2,     4,  ..., 59974, 59985, 59998])

        # idx after applies partial_num and partial_target
        partial_target_num_idx = partial_target_idx[
            torch.randperm(
                partial_target_idx.size()[0]
            )[:partial_num if partial_num < partial_target_idx.size()[0] else -1]
        ]
        # partial_target_num_idx: tensor([43017, 45843, 43440,  3620, 25991,  6417, 40425, 38389,  5477, 46967,
        # 54384, 43164, 32175, 13065, 36788, 24928, 59168, 17996, 21545, 55107,
        # 56483, 26094, 25783,  7961, 10844, 37258, 33559, 48178, 23700, 53589,
        # 53788, 59204, 40906, 38727, 11447, 20832, 17366, 55911,  3644, 57911,
        # 23990, 39825, 25863,  4492, 55385, 33223, 57882, 31879, 35712, 34674])

        # mask after applies partial_num and partial_target
        partial_target_num_mask = torch.zeros(
            dataset.targets.size()
        ).bool().fill_(False)
        partial_target_num_mask[partial_target_num_idx] = True
        # partial_target_num_mask: tensor([ True, False, False,  ..., False, False, False])

        # the targets selected by partial_target_num_mask
        partial_target_num_target = dataset.targets[
            partial_target_num_mask
        ].clone()

        if unlabelled_ratio > 0.0:
            # fill some targets with -1, indicating that they are unlabelled datapoints
            partial_target_num_target[
                # these unlabelled datapoints are randomly selected
                torch.randperm(
                    partial_target_num_target.size(0)
                )[
                    # the number of these unlabelled datapoints is determined by unlabelled_ratio
                    :int(unlabelled_ratio * partial_target_num_target.size(0))
                ]
            ] = -1.0

        # append the selected targets and data
        targets.append(partial_target_num_target)
        data.append(dataset.data[partial_target_num_mask].clone())

    dataset.targets = torch.cat(targets, dim=0)
    dataset.data = torch.cat(data, dim=0)

    dataset = post_dataset(dataset)

    return dataset


def map_dataset_targets(dataset, mapper=None):
    """Map the targets of the dateset.

    Args:
        dataset: The dataset.
        mapper (dict): A dict, the key of which is the original targets and the value is the mapped targets.
    """

    if mapper is None:

        return dataset

    else:

        assert isinstance(mapper, dict), "mapper must be a dict."

        dataset = pre_dataset(dataset)

        original_targets = dataset.targets.clone()
        for original_target in mapper.keys():
            mapped_target = mapper[original_target]
            dataset.targets[
                (original_targets == original_target)
            ] = mapped_target

        dataset = post_dataset(dataset)

        return dataset


def data_loader_fn(dataset_name="XOR",
                   train=True,
                   download=False,
                   # data
                   data_image_size=28,
                   data_is_gray=True,
                   data_is_flatten=True,
                   # target
                   target_min=-0.1,
                   target_max=1.0,
                   # partial dataset
                   partial_targets_num=10,
                   partial_num=6000,
                   # kwargs for DataLoader
                   batch_size=4,
                   shuffle=True,
                   drop_last=True,
                   num_workers=0,
                   pin_memory=True,
                   data_pil_transforms=[],
                   ):

    assert isinstance(dataset_name, str), (
        f"dataset_name must be a str, but got {type(dataset_name)}."
    )
    assert isinstance(data_is_gray, bool), (
        f"data_is_gray must be a bool, but got {type(data_is_gray)}."
    )
    assert isinstance(data_image_size, int), (
        f"data_image_size must be a int, but got {type(data_image_size)}."
    )
    assert isinstance(target_min, (int, float)), (
        f"target_min must be a int or float, but got {type(target_min)}."
    )
    assert isinstance(target_max, (int, float)), (
        f"target_max must be a int or float, but got {type(target_max)}."
    )
    assert isinstance(partial_targets_num, int), (
        f"partial_targets_num must be a int, but got {type(partial_targets_num)}."
    )
    assert isinstance(data_pil_transforms, list), (
        f"data_pil_transforms must be a list, but got {type(data_pil_transforms)}."
    )

    if dataset_name in ['XOR']:

        if dataset_name == 'XOR':

            data = [
                [-1, -1],
                [-1, 1],
                [1, -1],
                [1, 1],
            ]
            target = [
                [target_min],
                [target_max],
                [target_max],
                [target_min],
            ]

        else:
            raise ValueError(f"Unknown dataset_name: {dataset_name}")

        dataset = TensorDataset(
            torch.Tensor(data),
            torch.Tensor(target),
        )

    elif dataset_name in ['MNIST', 'FashionMNIST', 'CIFAR10']:

        transform = []
        if dataset_name in ['CIFAR10']:
            if data_is_gray:
                transform.append(transforms.Grayscale())
        transform.append(transforms.Resize(data_image_size))
        transform.extend(data_pil_transforms)
        transform.append(transforms.ToTensor())
        if data_is_flatten:
            transform.append(u.transforms_flatten)

        target_transform = []
        target_transform.append(
            transforms.Lambda(
                lambda idx: u.np_idx2onehot(
                    idx,
                    size=partial_targets_num,
                    target_min=target_min,
                    target_max=target_max,
                )
            )
        )

        dataset = partial_dateset_v1(
            eval(
                'datasets.{}'.format(dataset_name)
            )(
                os.environ.get('DATA_DIR'),
                train=train,
                download=download,
                transform=transforms.Compose(transform),
                target_transform=transforms.Compose(target_transform)
            ),
            partial_num=partial_num,
            partial_targets=list(range(partial_targets_num)),
        )

    else:

        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
