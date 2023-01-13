from __future__ import print_function

import os
import subprocess
import glob
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

import utils as u

logger = u.getLogger(__name__)


class HandCodedRulesTrainable(DatasetLearningTrainable):
    """HandCodedRulesTrainable.

        Manage:
            - different learning rules coded by hand.
    """

    def setup(self, config):

        super(HandCodedRulesTrainable, self).setup(config)

        exec(self.config.get("before_HandCodedRulesTrainable_setup_code", "pass"))

        logger.warning("HandCodedRulesTrainable is not maintained anymore.")

        # depreciation warnings
        if self.config.get("is_reset_variables", None) is not None:
            raise RuntimeError(
                "is_reset_variables has been deprecated in favor of data_packs.do"
            )
        if self.config.get("summarize_over_batch_idx", None) is not None:
            raise RuntimeError(
                "summarize_over_batch_idx has been deprecated in favor of summarize_over_batch_idx_fn. "
                "and the default behavior has changed to not deviding batch size, as it should be done at the level of log. "
            )
        if self.config.get("setup_code", None) is not None:
            raise RuntimeError(
                "setup_code is ambiguous so deprecated, use <before_(Trainable)_setup_code> or <after_(Trainable)_setup_code>. "
            )

        # config
        self.rule = self.config["rule"]
        self.ns = eval(self.config["ns"])
        self.l_start = 0
        self.l_end = len(self.ns) - 1
        self.loss_coefficient = self.config.get("loss_coefficient", 1.0)
        self.inference_rate = self.config["inference_rate"]
        self.inference_duration = self.config["inference_duration"]
        self.inference_rate_discount = self.config["inference_rate_discount"]
        assert self.inference_rate_discount <= 1.0
        self.learning_rate = self.config["learning_rate"]
        self.batch_size = self.config["batch_size"]

        if self.rule in ["Almeida-Pineda", "GeneRec", "Hopfield"]:
            assert self.loss_coefficient == 1.0, NotImplementedError

        # create f and related functional
        self.f = eval(self.config["f"])
        self.f_grad = utils.grad(self.f)
        self.f_inverse = eval(self.config["f_inverse"])

        # Ws
        # # create Ws
        self.Ws = {}
        for l in range(self.l_start, self.l_end):
            self.Ws[l] = torch.FloatTensor(
                self.ns[l], self.ns[l + 1]
            ).to(self.device)
            torch.nn.init.xavier_normal_(self.Ws[l])
        # # initialize Ws
        exec(self.config.get("init_code", "pass"))
        # # hold Ws_start
        self.Ws_start = copy.deepcopy(self.Ws)

        # create xs
        self.xs = {}
        for l in range(self.l_start, self.l_end + 1):
            self.xs[l] = torch.FloatTensor(
                self.batch_size, self.ns[l]
            ).normal_(0.0, 1.0).to(self.device)

        # create ys
        if self.rule in ["Almeida-Pineda"]:
            self.ys = copy.deepcopy(self.xs)
        # create varepsilons and mus
        if self.rule in ["Predictive-Coding"]:
            self.mus = copy.deepcopy(self.xs)
            self.varepsilons = copy.deepcopy(self.xs)
        # create deltas
        if self.rule in ["Back-Propagation"]:
            self.deltas = copy.deepcopy(self.xs)

        # for visualize
        self.is_visualize_setup_done = False
        self.visualize_frame_i = 0

        exec(self.config.get("after_HandCodedRulesTrainable_setup_code", "pass"))

    def _inference_step(self, include_output):
        """Update xs for one step.
        """

        assert self.rule in [
            "GeneRec",
            "Hopfield",
            "Almeida-Pineda",
            "Predictive-Coding"
        ]

        exec(self.config.get("before_inference_step_code", "pass"))

        if self.rule in ["Predictive-Coding"]:
            # compute self.varepsilons
            for l in range(self.l_start, self.l_end + 1):
                if l > self.l_start:
                    self.mus[l] = self.f(
                        self.xs[l - 1]
                    ).matmul(
                        self.Ws[l - 1]
                    )
                self.varepsilons[l] = self.xs[l] - self.mus[l]

        # get l_start_
        if self.config.get("inference_include_input", False):
            l_start_ = self.l_start
        else:
            # # inference normally excludes self.l_start as input neurons are clamped to self.s_in
            l_start_ = self.l_start + 1

        # get l_end_
        if include_output:
            # # inference includes self.l_end
            l_end_ = self.l_end + 1
        else:
            # # inference excludes self.l_end
            l_end_ = self.l_end

        # inference on each layer
        for l in range(l_start_, l_end_):

            # dx contains several terms, so hold the terms in a list
            dx = []

            if self.rule in ["GeneRec", "Hopfield"]:

                if self.rule in ["GeneRec"]:
                    # dx term: self supress
                    dx.append(
                        - self.xs[l]
                    )

                if l > self.l_start:
                    # dx term: connections to l-1
                    dx.append(
                        self.f_grad(
                            self.xs[l]
                        ) * self.f(
                            self.xs[l - 1]
                        ).matmul(
                            self.Ws[l - 1]
                        )
                    )

                if l < self.l_end:
                    # dx term: connections to l+1
                    dx.append(
                        self.f_grad(
                            self.xs[l]
                        ) * self.f(
                            self.xs[l + 1]
                        ).matmul(
                            self.Ws[l].t()
                        )
                    )

            elif self.rule in ["Almeida-Pineda"]:

                # dx term: self supress
                dx.append(
                    - self.xs[l]
                )

                if l > self.l_start:
                    # dx term: connections to l-1
                    dx.append(
                        self.f(
                            self.xs[l - 1]
                        ).matmul(
                            self.Ws[l - 1]
                        )
                    )

                if l < self.l_end:
                    # dx term: connections to l+1
                    dx.append(
                        self.f(
                            self.xs[l + 1]
                        ).matmul(
                            self.Ws[l].t()
                        )
                    )

            elif self.rule in ["Predictive-Coding"]:

                if (l) == self.l_end:
                    # error is from last_layer
                    coefficient = self.loss_coefficient
                else:
                    coefficient = 1.0
                # dx term: self supress
                dx.append(
                    -self.varepsilons[l] * coefficient
                )

                if l < self.l_end:
                    if (l + 1) == self.l_end:
                        # error is from last_layer
                        coefficient = self.loss_coefficient
                    else:
                        coefficient = 1.0
                    # dx term: connections to l+1
                    dx.append(
                        self.f_grad(
                            self.xs[l]
                        ) * (
                            self.varepsilons[l + 1]
                        ).matmul(
                            self.Ws[l].t()
                        ) * (coefficient**2)
                    )

            else:
                raise NotImplementedError

            # update x from dx, which is a list holding all terms of dx
            self.xs[l] += (
                self.inference_rate * sum(dx)
            )

        total_energy = None

        if self.rule in ["Predictive-Coding"]:
            layer_energies = []
            for l in range(self.l_start, self.l_end + 1):
                layer_energies.append(
                    (self.varepsilons[l].pow(2) * 0.5).sum().item()
                )
            total_energy = sum(layer_energies)

        exec(self.config.get("after_inference_step_code", "pass"))

        return total_energy

    def _propagation_step(self, J_l_end):
        """Update ys for one step.
        """

        assert self.rule in ["Almeida-Pineda"]

        # get l_start_
        # # propagation always includes self.l_start
        l_start_ = self.l_start

        # get l_end_
        # # inference includes self.l_end
        l_end_ = self.l_end + 1

        # propagation on each layer
        for l in range(l_start_, l_end_):

            # dy contains several terms, so hold the terms in a list
            dy = []

            # self supress
            dy.append(
                - self.ys[l]
            )

            if self.l_start < l < self.l_end:
                # connections to l-1
                dy.append(
                    self.f_grad(
                        self.xs[l]
                    ) * self.ys[l - 1].matmul(
                        self.Ws[l - 1]
                    )
                )

            if l < self.l_end:
                # connections to l+1
                dy.append(
                    self.f_grad(
                        self.xs[l]
                    ) * self.ys[l + 1].matmul(
                        self.Ws[l].t()
                    )
                )

            if l == self.l_end:
                # if there is not an l+1 layer, use J_l_end instead
                dy.append(
                    J_l_end
                )

            # update self.ys (propagation)
            self.ys[l] += (
                self.inference_rate * sum(dy)
            )

    def _backpropagation(self, J_l_end):
        """Update deltas.
        """

        assert self.rule in ["Back-Propagation"]

        # the output layer
        self.deltas[self.l_end] = J_l_end

        for l in reversed(range(self.l_start + 1, self.l_end)):
            # connections to l+1
            self.deltas[l] = self.f_grad(
                self.xs[l]
            ) * self.deltas[l + 1].matmul(
                self.Ws[l].t()
            )

    def _get_dWs(self, phase="p"):
        """Get dWs aligned with l.
        """

        dWs = {}

        for l in range(self.l_start, self.l_end):

            if self.rule in ["GeneRec", "Hopfield"]:

                dW = torch.matmul(
                    self.f(
                        self.xs[l].t()
                    ),
                    self.f(
                        self.xs[l + 1]
                    )
                )

            elif self.rule in ["Almeida-Pineda", "Predictive-Coding", "Back-Propagation"]:

                # these rules all use error_term to update the weights

                if self.rule in ["Almeida-Pineda"]:
                    error_term = self.ys[l + 1]
                elif self.rule in ["Predictive-Coding"]:
                    error_term = self.varepsilons[l + 1]
                elif self.rule in ["Back-Propagation"]:
                    error_term = self.deltas[l + 1]
                else:
                    raise NotImplementedError

                dW = torch.matmul(
                    self.f(
                        self.xs[l].t()
                    ),
                    error_term
                )

                assert phase in ["p"]

            else:
                raise NotImplementedError

            dW = dW / self.batch_size

            if phase in ["p"]:
                dWs[l] = dW
            elif phase in ["n"]:
                dWs[l] = -dW
            else:
                raise NotImplementedError

        return dWs

    def _learning(self, dWss):
        """Update weights from dWss, which is a list of dWs, which is a dict of dW aligned with each l.
        """

        for dWs in dWss:
            for l in dWs.keys():
                self.Ws[l] += (
                    self.learning_rate * dWs[l]
                )

    def _clamp_input(self):
        """Clamp input neurons to s_in.
        """
        self.xs[self.l_start].copy_(self.s_in)
        if self.rule in ["Predictive-Coding"]:
            self.mus[self.l_start].copy_(self.s_in)

    def _prediction(self):
        """Prediction.

            This involves:
                - clamp input neurons to s_in
                - run network to make predictions
        """

        # clamp input neurons to s_in.
        self._clamp_input()

        if self.rule in ["Back-Propagation"]:
            # prediction by forward
            for l in range(self.l_start + 1, self.l_end + 1):
                self.xs[l] = self.f(
                    self.xs[l - 1]
                ).matmul(
                    self.Ws[l - 1]
                )

        elif self.rule in ["GeneRec", "Hopfield", "Almeida-Pineda", "Predictive-Coding"]:
            # prediction by inference (include output layer)
            self._inference(
                include_output=True
            )

        else:
            raise NotImplementedError

        prediction = self.xs[self.l_end]

        exec(self.config.get("after_prediction_code", "pass"))

        # return prediction
        return prediction

    def _reset_variables(self):
        """Reset variables
        """

        for l in range(self.l_start, self.l_end + 1):
            self.xs[l].fill_(0.0)
            if self.rule in ["Almeida-Pineda"]:
                self.ys[l].fill_(0.0)
            if self.rule in ["Predictive-Coding"]:
                self.mus[l].fill_(0.0)
                self.varepsilons[l].fill_(0.0)

    def _inference(self, include_output):
        """Update xs for inference_duration steps.
        """

        self.total_energies = []

        # variables for for self.inference_rate_discount < 1.0
        if self.inference_rate_discount < 1.0:
            inference_rate_discount_times = 0

        for t in range(self.inference_duration):

            total_energy = self._inference_step(
                include_output
            )

            self.total_energies.append(total_energy)

            # controls for self.inference_rate_discount < 1.0
            if self.inference_rate_discount < 1.0:
                # checks for self.inference_rate_discount < 1.0
                if total_energy is None:
                    raise RuntimeError(
                        "self.inference_rate_discount is applied according to total_energy returned from _inference_step, but None is returned"
                    )
                # controls for self.inference_rate_discount < 1.0
                if len(self.total_energies) > 1:
                    if self.total_energies[-1] >= self.total_energies[-2]:
                        inference_rate_discount_times += 1
                        self.inference_rate *= self.inference_rate_discount

        if self.inference_rate_discount < 1.0:
            # checks for self.inference_rate_discount < 1.0
            if (self.total_energies[1] > self.total_energies[0] * 10) or np.isnan(np.asarray(self.total_energies)).any():
                raise RuntimeError(
                    "total_energy increases by 10 times at the start of inference (or there are nan(s) in self.total_energies), please use a smaller inference_rate, self.total_energies[:10]:\n{}".format(
                        self.total_energies[:10]
                    )
                )
            if inference_rate_discount_times < self.config["minimal_inference_rate_discount_times"]:
                raise RuntimeError(
                    "inference_rate_discount_times is not enough, increase inference_rate, or increase inference_duration"
                )

        # set back inference_rate as it could be discounted
        self.inference_rate = self.config["inference_rate"]

        exec(self.config.get("after_inference_code", "pass"))

    def iteration_step(
        self,
        data_pack_key,
        batch_idx,
        batch,
        do_key,
    ):

        # unpack batch
        self.s_in, self.s_out = batch

        # see argument dWss of self._learning()
        dWss = []

        if do_key == 'reset':

            self._reset_variables()

        elif do_key == 'predict':

            self.prediction = self._prediction()

            # loss is computed between prediction and s_out
            # note that it might not be used for training
            self.loss_per_neuron_per_datapoint = (
                (
                    self.prediction - self.s_out
                ).pow(
                    2
                ) * 0.5
            )
            self.loss_per_neuron = self.loss_per_neuron_per_datapoint.sum(
                # reduce the datapoint dimension
                dim=0, keepdim=False
            )
            self.loss = self.loss_per_neuron.sum(
                # reduce the neuron dimension
                dim=0, keepdim=False
            )

            if self.rule in ["GeneRec", "Hopfield"]:
                # negative phase, append dWs to dWss
                dWss.append(
                    self._get_dWs(phase="n")
                )

        elif do_key == 'learn':

            self._clamp_input()

            if self.rule in ["Almeida-Pineda", "Back-Propagation"]:

                # these two learning rules start from J_l_end
                J_l_end = (
                    self.s_out - self.xs[self.l_end]
                ) * self.loss_coefficient

                if self.rule in ["Almeida-Pineda"]:

                    # propagation with multiple steps (configed by inference_duration)
                    for t in range(self.inference_duration):
                        self._propagation_step(J_l_end)

                elif self.rule in ["Back-Propagation"]:

                    # backpropagation
                    self._backpropagation(J_l_end)

                else:

                    raise NotImplementedError

            elif self.rule in ["GeneRec", "Hopfield", "Predictive-Coding"]:

                # clamp to s_out
                self.xs[self.l_end].copy_(self.s_out)

                # inference (exclude output layer)
                self._inference(
                    include_output=False
                )

            else:
                raise NotImplementedError

            # positive phase, append dWs to dWss
            dWss.append(
                self._get_dWs(phase="p")
            )

            # append dWs to dWss
            self._learning(dWss)

        else:

            raise NotImplementedError

    def setup_visualize(self):
        plt.ion()
        assert os.path.isabs(self.save_visualize_path)
        self.clean_visualize_results()
        self.is_visualize_setup_done = True

    def clean_visualize_results(self):
        os.chdir(self.save_visualize_path)
        for file_name in glob.glob("*.png"):
            os.remove(file_name)
        for file_name in glob.glob("*.mp4"):
            os.remove(file_name)

    def make_video(self):
        os.chdir(self.save_visualize_path)
        subprocess.call([
            'ffmpeg', '-framerate', '8', '-i', 'frame%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
            'video.mp4'
        ])
        self.visualize_frame_i = 0

    def visualize(self, title, fix_output):

        if not self.is_visualize_setup_done:
            self.setup_visualize()

        plt.figure(title)
        plt.clf()
        plt.axis('off')

        val_min = self.config.get("val_min", 0.0)
        val_max = self.config.get("val_max", 1.0)

        # for Predictive-Coding, compute mus
        if self.rule in ["Predictive-Coding"]:
            for l in range(self.l_start, self.l_end + 1):
                if l > self.l_start:
                    self.mus[l] = self.f(
                        self.xs[l - 1]
                    ).matmul(
                        self.Ws[l - 1]
                    )

        # holding variables needed for visualization
        coordinates = {
            "x": {
                "x": {},
                "y": {},
                "y_min": {},
                "y_max": {},
            },
            "mu": {
                "x": {},
                "y": {},
                "y_min": {},
                "y_max": {},
            },
            "fix": {
                "x": {},
                "y": {},
            },
        }

        for b in range(self.batch_size):

            for _, coordinate in coordinates.items():
                for _, axis in coordinate.items():
                    axis[b] = {}

            for l in range(self.l_start, self.l_end + 1):

                for _, coordinate in coordinates.items():
                    for _, axis in coordinate.items():
                        axis[b][l] = {}

                for i in range(self.ns[l]):

                    # the are all on the same post so the same x
                    x = l

                    # visualize post
                    y_min = b + (i) / self.ns[l]
                    y_max = b + (i + 0.8) / self.ns[l]
                    plt.text(x, y_min, val_min,
                             color='lightgray',
                             )
                    plt.text(x, y_max, val_max,
                             color='lightgray',
                             )
                    coordinates["x"]["y_min"][b][l][i] = y_min
                    coordinates["x"]["y_max"][b][l][i] = y_max

                    # visualize x
                    x_val = self.xs[l][b][i].item()
                    y = (
                        x_val - val_min
                    ) / (
                        val_max - val_min
                    ) * (
                        y_max - y_min
                    ) + y_min
                    coordinates["x"]["x"][b][l][i] = x
                    coordinates["x"]["y"][b][l][i] = y
                    plt.text(
                        x,
                        y,
                        "{:.4f}".format(x_val),
                        color='black',
                    )

                    # visualize fix
                    if l == 0:
                        coordinates["fix"]["x"][b][l][i] = x
                        coordinates["fix"]["y"][b][l][i] = y
                    if (l == self.l_end) and fix_output and (self.rule in ["Predictive-Coding"]):
                        coordinates["fix"]["x"][b][l][i] = x
                        coordinates["fix"]["y"][b][l][i] = y

                    # visualize mu
                    if self.rule in ["Predictive-Coding"]:

                        if l > 0:

                            mu_val = self.mus[l][b][i].item()
                            y = (
                                mu_val -
                                val_min
                            ) / (
                                val_max - val_min
                            ) * (
                                y_max - y_min
                            ) + y_min
                            coordinates["mu"]["x"][b][l][i] = x
                            coordinates["mu"]["y"][b][l][i] = y
                            plt.text(
                                x,
                                y,
                                "{:.4f}".format(mu_val),
                                color='black',
                            )

        array_coordinates = copy.deepcopy(coordinates)
        for _, array_coordinate in array_coordinates.items():
            for k, _ in array_coordinate.items():
                array_coordinate[k] = np.array(
                    list(
                        utils.dict_values(
                            array_coordinate[k]
                        )
                    )
                )

        # visualize post
        plt.errorbar(
            x=array_coordinates["x"]["x"],
            y=(array_coordinates["x"]["y_max"] +
               array_coordinates["x"]["y_min"]) / 2,
            yerr=(array_coordinates["x"]["y_max"] -
                  array_coordinates["x"]["y_min"]) / 2,
            capsize=5,
            linestyle='None',
            elinewidth=1,
            ecolor='lightgray',
        )

        # visualize mu
        if self.rule in ["Predictive-Coding"]:
            plt.scatter(
                x=array_coordinates["mu"]["x"],
                y=array_coordinates["mu"]["y"],
                linestyle='None',
                color='white',
                s=200,
                edgecolors='black',
                marker='H',
            )

        # visualize x
        plt.scatter(
            x=array_coordinates["x"]["x"],
            y=array_coordinates["x"]["y"],
            linestyle='None',
            color='lightskyblue',
            s=200,
            edgecolors='black',
        )

        # visualize fix
        plt.scatter(
            x=array_coordinates["fix"]["x"],
            y=array_coordinates["fix"]["y"],
            linestyle='None',
            color='black',
            s=10,
            marker='*',
        )

        # visualize weights
        connect_from = "x"
        if self.rule in ["Predictive-Coding"]:
            connect_to = "mu"
        elif self.rule in ["Back-Propagation"]:
            connect_to = "x"
        else:
            raise NotADirectoryError
        for b in range(self.batch_size):
            for l in range(self.l_start, self.l_end):
                for i in range(self.ns[l]):
                    for j in range(self.ns[l + 1]):
                        x_i, x_j = coordinates[connect_from]["x"][b][l][i], coordinates[connect_to]["x"][b][l + 1][j]
                        y_i, y_j = coordinates[connect_from]["y"][b][l][i], coordinates[connect_to]["y"][b][l + 1][j]
                        plt.plot(
                            [x_i, x_j],
                            [y_i, y_j],
                            color='black',
                        )
                        plt.text(
                            (x_i + x_j) / 2,
                            (y_i + y_j) / 2,
                            "{:.4f}".format(self.Ws[l][i][j].item()),
                            color='black',
                        )

        plt.draw()
        plt.pause(0.000001)

        if self.save_visualize_path is not None:
            if not os.path.exists(self.save_visualize_path):
                os.makedirs(self.save_visualize_path)
            plt.savefig(
                os.path.join(
                    self.save_visualize_path,
                    "frame%02d.png" % self.visualize_frame_i,
                )
            )
            self.visualize_frame_i += 1
