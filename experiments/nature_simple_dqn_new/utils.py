import collections
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import predictive_coding as pc

import analysis_utils as au


def plot(df, plot="Reward"):

    df = au.nature_pre(df)

    groups = ['Env', 'Rule', 'pc_learning_rate']

    df = au.add_metric_per_group(
        df, groups,
        lambda df: (
            'mean per group', df['Mean of episode reward'].mean()
        ),
    )

    groups.pop(-1)

    df = au.select_rows_per_group(
        df, groups,
        lambda df: df['mean per group'] == df['mean per group'].max()
    )

    df = au.drop_cols(df, ['mean per group'])

    df = au.extract_plot(df, f'Episode {plot}', 'training_iteration')

    df = df[df['training_iteration'].isin(list(range(0, 10000, 100)))]

    g = au.nature_relplot_curve(
        data=df,
        x='training_iteration',
        y=f'Episode {plot}',
        hue='Rule', style='Rule',
        hue_order=['PC', 'BP'],
        style_order=['PC', 'BP'],
        col='Env',
        aspect=0.8,
        sharey=False
    )

    au.nature_post(g, is_grid=True)

    return df


class RunningStats(object):
    """Computes running mean and standard deviation
    Url: https://gist.github.com/wassname/a9502f562d4d3e73729dc5b184db2501
    Adapted from:
        *
        <http://stackoverflow.com/questions/1174984/how-to-efficiently-\
calculate-a-running-standard-deviation>
        * <http://mathcentral.uregina.ca/QQ/database/QQ.09.02/carlos1.html>
        * <https://gist.github.com/fvisin/5a10066258e43cf6acfa0a474fcdb59f>

    Usage:
        rs = RunningStats()
        for i in range(10):
            rs += np.random.randn()
            print(rs)
        print(rs.mean, rs.std)
    """

    def __init__(self, n=0., m=None, s=None, per_dim=True):
        self.n = n
        self.m = m
        self.s = s
        self.per_dim = per_dim

    def clear(self):
        self.n = 0.

    def push(self, x):
        # process input
        if self.per_dim:
            self.update_params(x)
        else:
            for el in x.flatten():
                self.update_params(el)

    def update_params(self, x):
        self.n += 1
        if self.n == 1:
            self.m = x
            self.s = 0.
        else:
            prev_m = self.m.copy()
            self.m += (x - self.m) / self.n
            self.s += (x - prev_m) * (x - self.m)

    def __add__(self, other):
        if isinstance(other, RunningStats):
            sum_ns = self.n + other.n
            prod_ns = self.n * other.n
            delta2 = (other.m - self.m) ** 2.
            return RunningStats(sum_ns,
                                (self.m * self.n + other.m * other.n) / sum_ns,
                                self.s + other.s + delta2 * prod_ns / sum_ns)
        else:
            self.push(other)
            return self

    @property
    def mean(self):
        return self.m if self.n else 0.0

    def variance(self):
        return self.s / (self.n) if self.n else 0.0

    @property
    def std(self):
        return np.sqrt(self.variance())

    def __repr__(self):
        return '<RunningMean(mean={: 2.4f}, std={: 2.4f}, n={: 2f}, m={: 2.4f}, s={: 2.4f})>'.format(self.mean, self.std, self.n, self.m, self.s)

    def __str__(self):
        return 'mean={: 2.4f}, std={: 2.4f}'.format(self.mean, self.std)

    def normalize(self, x):
        return (
            x - self.mean
        ) / (
            self.std if np.all(self.std) else 1.0
        )


class ReplayBuffer():
    def __init__(self, buffer_limit, sample_to_device):

        self.buffer = collections.deque(maxlen=buffer_limit)
        self.sample_to_device = sample_to_device

    def put(self, transition):

        self.buffer.append(transition)

    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float).to(self.sample_to_device), torch.tensor(a_lst).to(self.sample_to_device), \
            torch.tensor(r_lst).to(self.sample_to_device), torch.tensor(s_prime_lst, dtype=torch.float).to(self.sample_to_device), \
            torch.tensor(done_mask_lst).to(self.sample_to_device)

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):

    def __init__(self, predictive_coding, num_obs, num_act, bias=True, pc_layer_at='before_acf', hidden_size=128, num_hidden=1, acf='Sigmoid'):

        super(Qnet, self).__init__()

        self.predictive_coding = predictive_coding
        self.num_act = num_act

        model = []

        # input layer
        model.append(nn.Linear(num_obs, hidden_size, bias=bias))
        if self.predictive_coding and pc_layer_at == 'before_acf':
            model.append(pc.PCLayer())
        model.append(eval('nn.{}()'.format(acf)))
        if self.predictive_coding and pc_layer_at == 'after_acf':
            model.append(pc.PCLayer())

        for i in range(num_hidden):

            # hidden layer
            model.append(nn.Linear(hidden_size, hidden_size, bias=bias))
            if self.predictive_coding and pc_layer_at == 'before_acf':
                model.append(pc.PCLayer())
            model.append(eval('nn.{}()'.format(acf)))
            if self.predictive_coding and pc_layer_at == 'after_acf':
                model.append(pc.PCLayer())

        # output layer
        model.append(nn.Linear(hidden_size, num_act, bias=bias))

        self.model = nn.Sequential(*model)

    def forward(self, x):

        return self.model(x)

    def sample_action(self, obs, epsilon):

        if random.random() < epsilon:
            return random.randint(0, self.num_act - 1)

        else:
            return self.forward(obs).argmax().item()
