from __future__ import print_function

import copy
import pprint
import torch
import numpy as np
import matplotlib.pyplot as plt

import utils
import analysis_utils

from dataset_learning_trainable import DatasetLearningTrainable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import matmul


class AnyEnergyTrainable(DatasetLearningTrainable):
    """AnyEnergyTrainable.

        Manage:
            - any energy model.
    """

    def setup(self, config):

        super(AnyEnergyTrainable, self).setup(config)

        exec(self.config.get("before_AnyEnergyTrainable_setup_code", "pass"))

        # config
        self.energy_fn = self.config["energy_fn"]
        self.is_with_negative_phase = self.config["is_with_negative_phase"]
        self.connectivity = self.config["connectivity"]
        self.inference_duration = self.config["inference_duration"]
        self.batch_size = self.config["batch_size"]

        # network structure
        self.ns = eval(self.config["ns"])
        if self.connectivity.split(' ')[0] == 'Layered':
            pass
        elif self.connectivity.split(' ')[0] == 'Fully-connected':
            # a fully-connected network with each neuron connected to each of other xs and itself
            # the initialization of ns is describing the weight matrix
            self.ns = [sum(self.ns), sum(self.ns)]
        else:
            raise NotImplementedError
        self.l_start = 0
        self.l_end = len(self.ns) - 1

        # Ws

        # # create Ws
        self.Ws = {}
        for l in range(self.l_start, self.l_end):
            self.Ws[l] = nn.Parameter(
                torch.FloatTensor(
                    self.ns[l], self.ns[l + 1]
                ).to(self.device)
            )
            torch.nn.init.xavier_normal_(self.Ws[l])

        # # create Ws_backward
        self.Ws_backward = {}
        if self.connectivity.split(' ')[0] == 'Layered':
            # # # only layered network applies Ws_backward

            if self.connectivity.split(' ')[1] == 'Recurrent':
                # # # only recurrent layered network applies Ws_backward

                for l in range(self.l_start, self.l_end):

                    if self.connectivity.split(' ')[2] == 'Asymmetric':

                        # independent backward connection
                        self.Ws_backward[l] = nn.Parameter(
                            torch.FloatTensor(
                                self.ns[l], self.ns[l + 1]
                            ).to(self.device)
                        )
                        torch.nn.init.xavier_normal_(self.Ws_backward[l])

                    elif self.connectivity.split(' ')[2] == 'Symmetric':

                        # backward connection is the same as the forward connection
                        self.Ws_backward[l] = self.Ws[l]

                    else:

                        # no backward connection
                        pass

        # # initialize Ws
        exec(self.config.get("init_code", "pass"))

        # # init is a kind of update
        self.after_Ws_updated()

        # # create optimizer for Ws
        self.optimizer_learning = eval(self.config['optimizer_learning_fn'])(
            list(self.Ws.values()) + list(self.Ws_backward.values()),
            **self.config['optimizer_learning_kwargs']
        )

        # create xs
        self.xs = {}
        for l in range(self.l_start, self.l_end + 1):
            self.xs[l] = nn.Parameter(
                torch.FloatTensor(
                    self.batch_size, self.ns[l]
                ).normal_(0.0, 1.0).to(self.device)
            )

        # # for fully-connected network
        if self.connectivity.split(' ')[0] == 'Fully-connected':
            # # # for fully-connected network, the second layer of xs are not needed (as it is the first layer of xs themselves)
            self.xs.pop(1)
            assert len(self.xs) == 1
            assert list(self.xs.keys())[0] == 0

        # # create optimizer for xs
        self._create_optimizer_inference()

        exec(self.config.get("after_AnyEnergyTrainable_setup_code", "pass"))

    def _create_optimizer_inference(self):
        """(Re)create optimizer for xs (inference).
        """
        self.optimizer_inference = eval(self.config['optimizer_inference_fn'])(
            list(self.xs.values()),
            **self.config['optimizer_inference_kwargs']
        )

    def compute_energy(self):
        """Compute energy.
        """

        energy = []

        # energy from forward direction
        for l in range(self.l_start, self.l_end):

            # x_pre, x_post and w
            x_pre = self.xs[l]
            if self.connectivity.split(' ')[0] == 'Layered':
                # for layered network, x_post is the x in the next layer
                x_post = self.xs[l + 1]
            elif self.connectivity.split(' ')[0] == 'Fully-connected':
                # for fully-connected network, x_post and x_pre are the same set of xs
                x_post = x_pre
            else:
                raise NotImplementedError
            w = self.Ws[l]

            energy.append(
                eval(self.energy_fn)
            )

        # energy from backward direction for layered-structured network
        if self.connectivity.split(' ')[0] == 'Layered':

            if self.connectivity.split(' ')[1] == 'Recurrent':

                for l in reversed(range(self.l_start, self.l_end)):

                    # x_pre, x_post and w
                    x_pre = self.xs[l + 1]
                    x_post = self.xs[l]
                    w = self.Ws_backward[l].t()

                    energy.append(
                        eval(self.energy_fn)
                    )

        return sum(energy)

    def clamp_xs(self, clamp):
        """Clamp input or/and output xs to self.s_in or/and self.s_out.
        """

        assert isinstance(clamp, list)

        if 's_in' in clamp:
            self.get_xs_input().data.copy_(self.s_in)
        if 's_out' in clamp:
            self.get_xs_output().data.copy_(self.s_out)

    def get_xs_input(self):
        """Get xs that is considered to be input xs.
        """

        if self.connectivity.split(' ')[0] == 'Layered':

            return self.xs[self.l_start]

        elif self.connectivity.split(' ')[0] == 'Fully-connected':

            return self.xs[0][:, :self.s_in.size(1)]

        else:

            raise NotImplementedError

    def get_xs_output(self):
        """Get xs that is considered to be output xs.
        """

        if self.connectivity.split(' ')[0] == 'Layered':

            return self.xs[self.l_end]

        elif self.connectivity.split(' ')[0] == 'Fully-connected':

            return self.xs[0][:, -self.s_out.size(1):]

        else:

            raise NotImplementedError

    def get_error(self):
        """Get error between self.s_out and output xs.
        """

        return (self.get_xs_output() - self.s_out).pow(2).sum() * 0.5

    def _multiply_inference_rate(self, multiplier):
        for param_group_i in range(len(self.optimizer_inference.param_groups)):
            self.optimizer_inference.param_groups[param_group_i][
                'lr'
            ] = self.optimizer_inference.param_groups[param_group_i][
                'lr'
            ] * multiplier

    def _multiply_learning_rate(self, multiplier):
        for param_group_i in range(len(self.optimizer_learning.param_groups)):
            self.optimizer_learning.param_groups[param_group_i][
                'lr'
            ] = self.optimizer_learning.param_groups[param_group_i][
                'lr'
            ] * multiplier

    def inference(self, clamp):
        """Update xs.
        """

        # every inference in an independent optimization problem
        # thus, starts over
        self._create_optimizer_inference()

        self.clamp_xs(clamp)

        last_energy = None

        for inference_i in range(self.config['inference_duration']):

            # update xs
            self.optimizer_inference.zero_grad()
            energy = self.compute_energy()

            # control inference rate
            if last_energy is not None:
                if energy < last_energy:
                    # amplify learning rate of x if energy does decrease during inference
                    self._multiply_inference_rate(
                        self.config['inference_rate_amplifier']
                    )
                else:
                    # discount learning rate of x if energy does NOT decrease during inference
                    self._multiply_inference_rate(
                        self.config['inference_rate_discount']
                    )
            last_energy = energy

            energy.backward()
            self.optimizer_inference.step()

            self.clamp_xs(clamp)

    def learning(self, is_negative):
        """Update Ws.
        """

        last_energy = None

        for learn_i in range(self.config['learning_duration']):

            # update Ws
            self.optimizer_learning.zero_grad()
            energy = self.compute_energy()

            # control learning rate
            if last_energy is not None:
                if energy < last_energy:
                    # amplify learning rate of x if energy does decrease during learning
                    self._multiply_learning_rate(
                        self.config['learning_rate_amplifier']
                    )
                else:
                    # discount learning rate of x if energy does NOT decrease during learning
                    self._multiply_learning_rate(
                        self.config['learning_rate_discount']
                    )
            last_energy = energy

            if is_negative:
                (-energy).backward()
            else:
                energy.backward()

            self.optimizer_learning.step()

            self.after_Ws_updated()

    def after_Ws_updated(self):
        """Applied every time Ws gets updated.
        """

        if self.connectivity.split(' ')[0] == 'Fully-connected':

            if self.connectivity.split(' ')[1] == 'None-self-recurrent':
                self.Ws[0].data.fill_diagonal_(0.0)

            if self.connectivity.split(' ')[2] == 'Symmetric':
                self.Ws[0].data = (
                    self.Ws[0].data + self.Ws[0].data.t()
                ) / 2.0

    def iteration_step(
        self,
        data_pack_key,
        batch_idx,
        batch,
        do_key,
    ):

        # unpack batch
        self.s_in, self.s_out = batch

        if do_key == 'predict':

            # inference with only input clamped
            self.inference(clamp=['s_in'])

            if self.is_with_negative_phase:

                # with negative phase, Ws is updated to increase the energy
                self.learning(is_negative=True)

        elif do_key == 'learn':

            # inference with both input and output clamped
            self.inference(clamp=['s_in', 's_out'])

            # Ws is updated to decrease the energy
            self.learning(is_negative=False)

        else:

            raise NotImplementedError
