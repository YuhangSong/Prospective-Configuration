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

import utils as u

logger = u.getLogger(__name__)


class AnyEnergyTrainable(DatasetLearningTrainable):
    """AnyEnergyTrainable.

        Manage:
            - any energy model.
    """

    def setup(self, config):

        super(AnyEnergyTrainable, self).setup(config)

        exec(self.config.get("before_AnyEnergyTrainable_setup_code", "pass"))

        logger.warning("HandCodedRulesTrainable is not maintained anymore.")

        # config
        self.energy_fn = {
            "both": self.config.get("energy_fn_both", self.config.get("energy_fn", None)),
            "inference": self.config.get("energy_fn_inference", self.config.get("energy_fn", None)),
            "learning":  self.config.get("energy_fn_learning",  self.config.get("energy_fn", None)),
        }
        self.learning_do = eval(
            self.config.get(
                "learning_do", "['inference','learning']"
            )
        )
        self.is_manual_xs_dynamic = self.config.get(
            "is_manual_xs_dynamic", False
        )
        self.is_with_negative_phase = self.config["is_with_negative_phase"]
        self.connectivity = self.config["connectivity"]
        self.batch_size = self.config["batch_size"]

        # network structure
        self.ns = eval(self.config["ns"])
        if self.connectivity.split(' ')[0] == 'Layered':
            pass
        elif self.connectivity.split(' ')[0] == 'Fully-connected':
            # a fully-connected network with each neuron connected to each of other xs and itself
            # the initialization of ns is describing the weight matrix
            self.ns_original = copy.deepcopy(self.ns)
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
        if self.config.get("init_code", None) is not None:
            raise Exception("init_code has been renamed to init_Ws_code")
        exec(self.config.get("init_Ws_code", "pass"))

        # # init is a kind of update
        self.clamp_Ws()

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

        exec(self.config.get("init_xs_code", "pass"))

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

    def compute_energies(self, phase):
        """Compute energies, returns a list of energies of different layers.
        """

        energies = []

        # energies from forward direction
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

            energies.append(
                eval(self.energy_fn[phase])
            )

        # energies from backward direction for layered-structured network
        if self.connectivity.split(' ')[0] == 'Layered':

            if self.connectivity.split(' ')[1] == 'Recurrent':

                for l in reversed(range(self.l_start, self.l_end)):

                    # x_pre, x_post and w
                    x_pre = self.xs[l + 1]
                    x_post = self.xs[l]
                    w = self.Ws_backward[l].t()

                    energies.append(
                        eval(self.energy_fn[phase])
                    )

        return torch.stack(energies)

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

        # debug
        # print(self.optimizer_inference.param_groups[param_group_i][
        #     'lr'
        # ])

    def _multiply_learning_rate(self, multiplier):
        for param_group_i in range(len(self.optimizer_learning.param_groups)):
            self.optimizer_learning.param_groups[param_group_i][
                'lr'
            ] = self.optimizer_learning.param_groups[param_group_i][
                'lr'
            ] * multiplier

    def step_dynamic(self, do_key, clamp, is_negative=False):
        """Update xs.
        """

        is_manual_xs = (
            (
                do_key in ['inference', 'both']
            ) and (
                self.is_manual_xs_dynamic
            )
        )
        is_optimize_xs = (
            (
                do_key in ['inference', 'both']
            ) and (
                not self.is_manual_xs_dynamic
            )
        )
        is_optimize_Ws = (
            do_key in ['learning', 'both']
        )
        is_optimize_anything = (
            is_optimize_xs or is_optimize_Ws
        )

        if is_optimize_anything:
            if is_optimize_xs:
                # every inference in an independent optimization problem
                # thus, starts over
                self._create_optimizer_inference()

        if do_key == 'inference':
            duration = self.config['inference_duration']
        elif do_key == 'learning':
            duration = self.config['learning_duration']
        elif do_key == 'both':
            duration = self.config['both_duration']
        else:
            raise NotImplementedError

        self.clamp_xs(clamp)
        self.clamp_Ws()

        if is_optimize_anything:
            last_energy = None

        energies_history = []
        for step_i in range(duration):

            if is_manual_xs:
                exec(self.config.get("manual_xs_dynamic_code", "pass"))
                self.clamp_xs(clamp)

            if is_optimize_anything:

                if is_optimize_xs:
                    self.optimizer_inference.zero_grad()
                if is_optimize_Ws:
                    self.optimizer_learning.zero_grad()

                energies_history.append(
                    self.compute_energies(phase=do_key)
                )
                energy = energies_history[-1].sum()

                # control inference rate
                if last_energy is not None:
                    if energy < last_energy:
                        if is_optimize_xs:
                            # amplify learning rate of x if energy does decrease during inference
                            self._multiply_inference_rate(
                                self.config['inference_rate_amplifier']
                            )
                        if is_optimize_Ws:
                            # amplify learning rate of x if energy does decrease during learning
                            self._multiply_learning_rate(
                                self.config['learning_rate_amplifier']
                            )
                    else:
                        if is_optimize_xs:
                            # discount learning rate of x if energy does NOT decrease during inference
                            self._multiply_inference_rate(
                                self.config['inference_rate_discount']
                            )
                        if is_optimize_Ws:
                            # discount learning rate of x if energy does NOT decrease during learning
                            self._multiply_learning_rate(
                                self.config['learning_rate_discount']
                            )
                last_energy = energy

                if is_negative:
                    (-energy).backward()
                else:
                    energy.backward()

                if is_optimize_xs:
                    self.optimizer_inference.step()
                    self.clamp_xs(clamp)
                if is_optimize_Ws:
                    self.optimizer_learning.step()
                    self.clamp_Ws()

            # debug
            # input(energy.item())

        # debug
        # input(energy.item())
        # input(clamp)

        if is_optimize_anything:
            return energy, torch.stack(energies_history)

    def clamp_Ws(self):
        """Applied every time Ws gets updated.
        """

        if self.connectivity.split(' ')[0] == 'Fully-connected':

            if self.connectivity.split(' ')[1] == 'None-self-recurrent':
                self.Ws[0].data.fill_diagonal_(0.0)

            if self.connectivity.split(' ')[2] == 'Symmetric':
                self.Ws[0].data = (
                    self.Ws[0].data + self.Ws[0].data.t()
                ) / 2.0

        exec(self.config.get("after_clamp_Ws", "pass"))

    def iteration_step(
        self,
        batch,
        do_key,
        data_pack_key=None,
        batch_idx=None,
    ):

        # unpack batch
        self.s_in, self.s_out = batch

        if do_key == 'predict':

            # inference with only input clamped
            self.step_dynamic(
                do_key='inference',
                clamp=['s_in'],
                is_negative=False,
            )

            self.prediction = self.get_xs_output().data.clone()

            if self.is_with_negative_phase:

                # with negative phase, Ws is updated to increase the energy
                self.step_dynamic(
                    do_key='learning',
                    clamp=['s_in'],
                    is_negative=True,
                )

        elif do_key == 'learn':

            for learning_do_key in self.learning_do:

                self.step_dynamic(
                    do_key=learning_do_key,
                    clamp=['s_in', 's_out'],
                    is_negative=False,
                )

        else:

            exec(self.config.get("iteration_step_else", "pass"))
