import random
import numpy as np
import torch
import torch.nn as nn
import pytest

import predictive_coding as pc


def test_train_default_silent():

    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)

    input_size = 2
    hidden_size = 2
    output_size = 2
    batch_size = 2

    data, target = (
        torch.rand(batch_size, input_size),
        torch.rand(batch_size, output_size)
    )

    linears = [
        nn.Linear(input_size, hidden_size),
        nn.Linear(hidden_size, hidden_size),
        nn.Linear(hidden_size, output_size),
    ]

    model = nn.Sequential(
        linears[0],
        pc.PCLayer(),
        nn.Sigmoid(),
        linears[1],
        pc.PCLayer(),
        nn.Sigmoid(),
        linears[2],
    )

    pc_trainer = pc.PCTrainer(
        model,
        plot_progress_at=[],
    )

    model.train()

    def loss_fn(output, _target):
        return (output - _target).pow(2).sum() * 0.5

    for _ in range(2):
        pc_trainer.train_on_batch(
            inputs=data,
            loss_fn=loss_fn,
            loss_fn_kwargs={
                '_target': target
            },
            is_log_progress=False,
            is_return_results_every_t=False,
        )

    np.testing.assert_equal(
        linears[0].weight.detach().numpy(),
        np.array(
            [[1.0914857387542725, -0.32670697569847107],
                [0.01766992174088955, -0.036283332854509354]]
        )
    )
    np.testing.assert_equal(
        linears[0].bias.detach().numpy(),
        np.array(
            [-0.5729419589042664, -0.26975196599960327]
        )
    )
    np.testing.assert_equal(
        linears[1].weight.detach().numpy(),
        np.array(
            [[-0.12819968163967133, 0.3927641808986664],
                [-0.7660729885101318, 0.35872143507003784]]
        )
    )
    np.testing.assert_equal(
        linears[1].bias.detach().numpy(),
        np.array(
            [0.95953369140625, 0.2396860271692276]
        )
    )
    np.testing.assert_equal(
        linears[2].weight.detach().numpy(),
        np.array(
            [[0.7276633977890015, 0.048212602734565735],
                [0.988935112953186, -0.4032057225704193]]
        )
    )
    np.testing.assert_equal(
        linears[2].bias.detach().numpy(),
        np.array(
            [0.028494490310549736, -0.12354260683059692]
        )
    )
