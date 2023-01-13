import os
import typing
import warnings
import tqdm
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim

from . import utils
from . import pc_layer


class PCTrainer(object):
    """A trainer for predictive-coding models that are implemented by means of
    :class:`pc_layer.PCLayer`s.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_x_fn: typing.Callable = optim.SGD,
        optimizer_x_kwargs: dict = {"lr": 0.1},
        manual_optimizer_x_fn: typing.Callable = None,
        x_lr_amplifier: float = 1.0,
        x_lr_discount: float = 0.5,
        loss_x_fn: typing.Callable = None,
        loss_inputs_fn: typing.Callable = None,
        optimizer_p_fn: typing.Callable = optim.Adam,
        optimizer_p_kwargs: dict = {"lr": 0.001},
        manual_optimizer_p_fn: typing.Callable = None,
        T: int = 512,
        update_x_at: typing.Union[str, typing.List[int]] = "all",
        update_p_at: typing.Union[str, typing.List[int]] = "all",
        energy_coefficient: float = 1.0,
        early_stop_condition: str = "False",
        update_p_at_early_stop: bool = True,
        plot_progress_at: typing.Union[str, typing.List[int]] = "all",
        is_disable_warning_energy_from_different_batch_sizes: bool = False,
    ):
        """Creates a new instance of ``PCTrainer``.

        Remind of notations:

            ------- h=0 ---------, ------- h=1 ---------
            t=0, t=1, ......, t=T, t=0, t=1, ......, t=T

            h: batch. In each h, the same batch of data is presented, i.e., data batch is changed when h is changed.
            t: iteration. The integration step of inference.

        Args:
            model: The predictive-coding model to train.

            optimizer_x_fn: Callable to create optimizer of x.
            optimizer_x_kwargs: Keyword arguments for optimizer_x_fn.
            manual_optimizer_x_fn: Manually create optimizer_x.
                This will override optimizer_x_fn and optimizer_x_kwargs.

                See:
                ```python
                input('Start from zil and then il?')
                ```
                in demo.py as an example.

            x_lr_discount: Discount of learning rate of x if the overall energy (energy of hidden layers + loss) does not decrease.
                Set to 1.0 to disable it.
                The goal of inference is to get things to convergence at the current
                batch of datapoints, which is different from the goal of updating parameters,
                which is to take a small step at the current batch of datapoints, so annealing
                the learning rate of x according to the overall energy (energy of hidden layers + loss) is generally benefiting.
                Also, having this parameter enabled generally means if the choice of learning rate of x is slightly larger than the reasonable value,
                the training would still be stable.
            x_lr_amplifier: Amplifier of learning rate of x if the overall energy decrease. This is problemetic, please use 1.0. In some test, this ruins the optimization.
            Note: It is recommended to set x_lr_discount = 0.9 and x_lr_amplifier = 1.1, which will significantly speed up inference.
                However, this is only verified to be benefitial when update_p_at = 'last' or 'never', not sure when update_p_at = 'all'. Please verify and report.
                This is not incorporate to default configurations to keep the library has stable default behavior.

            loss_x_fn: Use this function to compute a loss from xs.
                This can be used, for example, for applying sparsity penalty to x:
                    <loss_x_fn=lambda x: 0.001 * x.abs().sum()>
            loss_inputs_fn: Use this function to compute a loss from inputs.
                Only takes effect when <is_optimize_inputs=True> when calling <self.train_on_batch()>.
                This can be used, for example, for applying sparsity penalty (pooled inhibit in the following example) to x:
                    <loss_inputs_fn=F.relu(x.abs().sum(1)-1).sum(0)>

            optimizer_p_fn: See optimizer_x_fn.
            optimizer_p_kwargs: See optimizer_x_kwargs.
            manual_optimizer_p_fn: See manual_optimizer_x_fn.

            A search of optimizer_{x,p}_fn with learning rate of both as well as x_lr_discount can be found here https://github.com/YuhangSong/general-energy-nets/blob/master/experiments/same-time-full-batch-1/mean-test-error-1.md
                It should give some idea of what combinations are possibly good. But note that the experiment is only conducted in the following restricted configurations, so might not apply to other configurations.
                    - MLPs
                    - FashionMNIST
                    - full-batch training (batch size = size of training set)
                    - other detailed configurations see https://github.com/YuhangSong/general-energy-nets/blob/master/experiments/same-time-full-batch-1/bp-sazil-FashionMNIST-1.yaml

            T: Train on each sample for T times.
            update_x_at:
                If "all", update x during all t=0 to T-1.
                If "last", update x at t=T-1.
                If "last_half", update x during all t=T/2 to T-1.
                If "never", never update x during all t=0 to T-1.
                If list of int, update x at t in update_x_at.
            update_p_at: See update_x_at.

            energy_coefficient: The coefficient added to the energy.

            early_stop_condition: Early stop condition for <train_on_batch()>. It is a str and will be eval during and expected to produce a bool at the time.

            update_p_at_early_stop: When early stop is triggered, whether to update p at the iteration.

            plot_progress_at: Plot the progress of training at batchs (this will slow down training).
                It could be a list of batchs (int) at which you want to plot the progress.
                It could be "all", which means to plot progress for all batchs.

                This is useful at the initial stage of your coding, when you want to look into the dynamic within inference/energy minimization.

                Such plots will be saved to ```~/plot_progress/```. If you have set environment variable WORKING_HOME, this directory will be ```$WORKING_HOME/plot_progress/```

                A healthy dynamic should looks like this: https://github.com/YuhangSong/general-energy-nets/blob/master/plot_progress_example.md

                    The two figures are the same result presented in different ways. Taking the combined-{} figure as an example, several things you should identify to make sure the model is healthy:

                        For each h, loss decreases along t and energy (it is the energy of hidden layers) increases along t, meaning that the loss is being absorbed into the energy of hidden layers.

                        For each h, overall (it is the sum of loss and energy) decreases along t, meaning that that above effect is driven by reducing this overall.

                        As h increases, the curve of loss gets lower, meaning the weight update is taking in the energy of hidden layers, which is further taken from the loss.

                    The other figure named seperated-{} is the same as the above plot but seperate plots of different h.

            is_disable_warning_energy_from_different_batch_sizes: if disable warning when energy in the network is from different batch sizes.
        """

        assert isinstance(model, nn.Module)
        self._model = model

        assert callable(optimizer_x_fn)
        self._optimizer_x_fn = optimizer_x_fn

        assert isinstance(optimizer_x_kwargs, dict)
        self._optimizer_x_kwargs = optimizer_x_kwargs

        if manual_optimizer_x_fn is not None:
            assert callable(manual_optimizer_x_fn)
        self._manual_optimizer_x_fn = manual_optimizer_x_fn

        self._optimizer_x = None

        assert isinstance(x_lr_discount, float)
        assert x_lr_discount <= 1.0
        self._x_lr_discount = x_lr_discount

        assert isinstance(x_lr_amplifier, float)
        assert x_lr_amplifier >= 1.0
        self._x_lr_amplifier = x_lr_amplifier

        if loss_x_fn is not None:
            assert callable(loss_x_fn)
        self._loss_x_fn = loss_x_fn
        if self._loss_x_fn is not None:
            assert self.get_is_model_has_pc_layers(), (
                "<loss_x_fn> should only work with models with <PCLayer>. "
            )

        if loss_inputs_fn is not None:
            assert callable(loss_inputs_fn)
        self._loss_inputs_fn = loss_inputs_fn
        if self._loss_inputs_fn is not None:
            assert self.get_is_model_has_pc_layers(), (
                "<loss_inputs_fn> should only work with models with <PCLayer>. "
            )

        assert callable(optimizer_p_fn)
        self._optimizer_p_fn = optimizer_p_fn

        assert isinstance(optimizer_p_kwargs, dict)
        self._optimizer_p_kwargs = optimizer_p_kwargs

        if manual_optimizer_p_fn is not None:
            assert callable(manual_optimizer_p_fn)
        self._manual_optimizer_p_fn = manual_optimizer_p_fn

        self.recreate_optimize_p()

        assert isinstance(T, int)
        assert T > 0
        self._T = T

        if self.get_is_model_has_pc_layers():

            # ensure that T is compatible with the trained model
            if self._T < self.get_num_pc_layers() + 1:
                warnings.warn(
                    (
                        "You should always choose T such that T >= (<pc_trainer.get_num_pc_layers()> + 1), "
                        "as it ensures that the error can be PC-propagated through the network."
                    ),
                    category=RuntimeWarning
                )

            min_t = self.get_least_T()
            if self._T < min_t:
                warnings.warn(
                    (
                        f"If you have one pc_layer per layer, T={self._T} is too small. "
                        f"Please use a minimum T of {min_t}, which is just enough to PC-propagate the error through the network and have all weigths updated based on these PC-propagated errors. "
                        "In practice, you normally should have T much larger than this minimum T. "
                    ),
                    category=RuntimeWarning
                )

        update_x_at = self._preprocess_step_index_list(
            indices=update_x_at,
            T=self._T,
        )
        self._update_x_at = update_x_at

        update_p_at = self._preprocess_step_index_list(
            indices=update_p_at,
            T=self._T,
        )
        self._update_p_at = update_p_at

        assert isinstance(energy_coefficient, float)
        self._energy_coefficient = energy_coefficient

        assert isinstance(early_stop_condition, str)
        self._early_stop_condition = early_stop_condition

        assert isinstance(update_p_at_early_stop, bool)
        self._update_p_at_early_stop = update_p_at_early_stop

        if isinstance(plot_progress_at, str):
            assert plot_progress_at in ["all"]
        elif isinstance(plot_progress_at, list):
            for h in plot_progress_at:
                assert isinstance(h, int)
        else:
            raise NotImplementedError
        self._plot_progress_at = plot_progress_at
        self._is_plot_progress = not (isinstance(
            self._plot_progress_at, list) and len(self._plot_progress_at) == 0)
        if self._is_plot_progress:
            self.reset_plot_progress()

        assert isinstance(
            is_disable_warning_energy_from_different_batch_sizes, bool)
        self.is_disable_warning_energy_from_different_batch_sizes = is_disable_warning_energy_from_different_batch_sizes

    #  GETTERS & SETTERS  #####################################################################################################

    def get_T(self) -> int:
        return self._T

    def get_model(self) -> nn.Module:
        return self._model

    def get_optimizer_x(self) -> optim.Optimizer:
        return self._optimizer_x

    def set_optimizer_x(self, optimizer_x: optim.Optimizer) -> None:
        assert isinstance(optimizer_x, optim.Optimizer)
        self._optimizer_x = optimizer_x

    def get_optimizer_p(self) -> optim.Optimizer:
        return self._optimizer_p

    def set_optimizer_p(self, optimizer_p: optim.Optimizer) -> None:
        assert isinstance(optimizer_p, optim.Optimizer)
        self._optimizer_p = optimizer_p

    def get_is_model_has_pc_layers(self) -> bool:
        """Evaluates if the trained model contains :class:`pc_layer.PCLayer."""

        for _ in self.get_model_pc_layers():
            return True
        else:
            return False

    def get_model_pc_layers_training(self) -> list:
        """Get a list of <pc_layer.training>."""

        pc_layers_training = []
        for pc_layer in self.get_model_pc_layers():
            pc_layers_training.append(pc_layer.training)
        return pc_layers_training

    def get_is_model_training(self):
        """Get whether the model is in train mode. This indicates that the model is in train mode
        and also all child pc_layers are in train mode. The same applies for eval mode.

        Returns:
            (bool | None):
                (bool): Whether the model is in train mode (True) or eval mode (False).
                (None): The model is neither in train mode nor eval mode,
                    because the child pc_layers are not in a unified state.
                    Calling <model.train()> or <model.eval()> is needed to unify the children pc_layers' states.
        """

        if (self._model.training) and np.all(self.get_model_pc_layers_training()):
            return True
        elif (not self._model.training) and np.all([not training for training in self.get_model_pc_layers_training()]):
            return False
        else:
            return None

    def get_energies(self, is_per_datapoint: bool = False, named_layers: bool = False) -> typing.Union[list, typing.Dict[str, pc_layer.PCLayer]]:
        """Retrieves the energies held by each pc_layer.

        Args:
            is_per_datapoint: if get the per-datapoint energies.
            named_layers (bool): if True return a dict with the layer names as keys else a list of pc_layers.
                Analogous to `nn.Module.modules()` vs `nn.Module.named_modules()`

        Returns:
            (list | dict): The energies held by each pc_layer.
        """

        energies = {}
        batch_sizes = []
        for name, pc_layer in self.get_named_model_pc_layers():
            energy = pc_layer.energy_per_datapoint() if is_per_datapoint else pc_layer.energy()
            if energy is not None:
                energies[name] = energy
                batch_sizes += [
                    energy.size(0) if is_per_datapoint else energy.size()
                ]

        assert len(energies) > 0, (
            "You don't have any pc_layers or none of them is holding energy. "
        )

        if (not self.is_disable_warning_energy_from_different_batch_sizes) and (batch_sizes.count(batch_sizes[0]) != len(batch_sizes)):
            warnings.warn(
                (
                    f"You pc_layers hold energy of different batch_sizes: {batch_sizes}.\n"
                    "You can disable this warning by setting is_disable_warning_energy_from_different_batch_sizes in PCTrainer to True."
                ),
                category=RuntimeWarning
            )

        return energies if named_layers else list(energies.values())

    def get_model_parameters(self) -> typing.Generator[nn.Parameter, None, None]:
        """Retrieves the actual trainable parameters, which are all parameters except xs.
        """

        # fetch the xs
        all_model_xs = set(
            self.get_model_xs(
                is_warning_x_not_initialized=False,
            )
        )

        # iterate over all parameters in the trained model, and retrieve those that are actually trained (i.e., exclude xs)
        for param in self._model.parameters():
            if not any(param is x for x in all_model_xs):
                yield param

    def get_model_pc_layers(self) -> typing.Generator[pc_layer.PCLayer, None, None]:
        """Retrieves all :class:`pc_layer.PCLayer`s contained in the trained model."""

        for module in self._model.modules():
            if isinstance(module, pc_layer.PCLayer):
                yield module

    def get_named_model_pc_layers(self) -> typing.Generator[pc_layer.PCLayer, None, None]:
        """Retrieves all :class:`pc_layer.PCLayer`s contained in the trained model as named dictionary."""

        for name, module in self._model.named_modules():
            if isinstance(module, pc_layer.PCLayer):
                yield name, module

    def get_model_xs(self, is_warning_x_not_initialized=True) -> typing.Generator[nn.Parameter, None, None]:
        """Retrieves xs.
        """

        for pc_layer in self.get_model_pc_layers():
            model_x = pc_layer.get_x()
            if model_x is not None:
                yield model_x
            else:
                if is_warning_x_not_initialized:
                    warnings.warn(
                        (
                            "While you are getting x from all pc layers (calling <pc_trainer.get_model_xs()>), "
                            "some pc layers has not been initialized yet (i.e., has x being None). "
                            "This potentially causes bugs. "
                        ),
                        category=RuntimeWarning
                    )

    def get_num_pc_layers(self) -> int:
        """Computes the total number of :class:`pc_layer.PCLayer contained by the trained model."""

        return sum(1 for _ in self.get_model_pc_layers())

    def get_least_T(self) -> int:
        """Computes the minimum T required based on the number of :class:`pc_layer.PCLayer in the trained model.

        This is assuming that all pc_layers are at different layers (there are no pc_layers in the same layer).

        self.get_num_pc_layers() is to ensure all pc_layers hold informative error.
        +1 is to ensure all weights are updated based on the error in the pc_layers.
        """

        return self.get_num_pc_layers() + 1

    #  METHODS  ########################################################################################################

    def recreate_optimize_x(self) -> None:
        """Recreates the optimizer_x"""

        if self._manual_optimizer_x_fn is None:
            self._optimizer_x = self._optimizer_x_fn(
                self.get_model_xs(),
                **self._optimizer_x_kwargs
            )

        else:
            self._optimizer_x = self._manual_optimizer_x_fn()

    def recreate_optimize_p(self) -> None:
        """Recreates the optimizer_p"""

        if self._manual_optimizer_p_fn is None:
            self._optimizer_p = self._optimizer_p_fn(
                self.get_model_parameters(),
                **self._optimizer_p_kwargs
            )

        else:
            self._optimizer_p = self._manual_optimizer_p_fn()

    def reset_plot_progress(self):
        """Reset plot progress."""

        self._h = 0
        self._plot_progress = {
            "key": [],
            "h": [],
            "t": [],
            "value": [],
        }

    def train_on_batch(
        self,
        inputs: typing.Any,
        loss_fn: typing.Callable = None,
        loss_fn_kwargs: dict = {},
        is_sample_x_at_batch_start: bool = True,
        is_reset_optimizer_x_at_batch_start: bool = True,
        is_reset_optimizer_p_at_batch_start: bool = False,
        is_unwrap_inputs: bool = False,
        is_optimize_inputs: bool = False,
        callback_after_backward: typing.Callable = None,
        callback_after_backward_kwargs: dict = {},
        callback_after_t: typing.Callable = None,
        callback_after_t_kwargs: dict = {},
        is_log_progress: bool = True,
        is_return_results_every_t: bool = True,
        is_checking_after_callback_after_t: bool = True,
        debug: dict = {},
        backward_kwargs: dict = {},
        is_clear_energy_after_use: bool = False,
        is_return_outputs: bool = False,
    ):
        """Train on a batch.

        Args:

            inputs: This will be passed to self.model().

            loss_fn: The function that takes in
                    - the output of self.model
                    - the loss_fn_kwargs as keyword arguments
                and returns a loss.

            loss_fn_kwargs: The keyword arguments passed to loss_fn.

            is_sample_x_at_batch_start: Whether to sample x at the start of the batch.
            is_reset_optimizer_x_at_batch_start: Whether to reset optimizer_x at the start of the batch.
                The default values of the above two arguments are True as we assume for each batch from t=0 to t=T the inference is independent.
                    Specifically, we know that the batch of datapoints is fixed during t=0 to t=T, and will switch to another batch of datapoints from t=T:

                        ------- h=0 ---------, ------- h=1 ---------
                        t=0, t=1, ......, t=T, t=0, t=1, ......, t=T
                         |                    ^ |
                         |        switch the batch of datapoints
                         ^                      ^
                      at_batch_start          at_batch_start

                    The above two arguments controls if sampel x and reset optimizer_x at the start of the batch (at <at_batch_start> in the above digram).

                    If you are doing full batch training (you don't switch between the batch of datapoints and the datapoints are not shuffled either):
                        You may set these two arguments to False after the first batch (h=0 in the above digram).
                        So that x will not be resampled, and the optimizer_x will not be reset.

            is_reset_optimizer_p_at_batch_start: See is_reset_optimizer_x_at_batch_start.

            is_unwrap_inputs: If unwrap inputs to be multiple arguments.

            is_optimize_inputs: If optimize inputs.
                If True, the inputs will be optimized, along with x in your pc_layers.
                    Behind the scene, the inputs will be wrapped into a Parameter, and appended to the optimizer_x.
                    You can access the optimized inputs by pc_trainer.inputs.
                A more elegent way of doing this (also slightly more efficient) would be 
                    put a pc_layer at the beginning of your model 
                    set the energy_fn of this pc_layer to be lambda inputs: 0.0 * inputs['mu'] (i.e., a invalid energy_fn).
                    set is_optimize_inputs to False here 
                    You can access the optimized inputs by the_first_pc_layer_you_added.get_x().
                        by setting the energy_fn to be a valid energy_fn, you add some forces on the inputs that you were optimizing, for example, enforce it to be a onehot vector.
                            I hope this helps you see the flexibility of using pc_layers to achieve interesting things.

            callback_after_backward: Callback functon after backward. It is a good place to do clip gradients. The function will takes in
                - t
                - callback_after_backward_kwargs as keyword arguments

            callback_after_backward_kwargs: This will be passed to callback_after_backward() as keyword arguments.

            callback_after_t: Callback functon after at the end of t. The function will taks in
                - t
                - callback_after_t_kwargs as keyword arguments

            callback_after_t_kwargs: This will be passed to callback_after_t() as keyword arguments.

            is_log_progress: If log progress of training (this will slow down training).

            is_return_results_every_t: If return results of training at every t (this will slow down training).
                If False, only results at the last t will be returned.

            is_checking_after_callback_after_t: If checking the model after callback_after_t() (this will slow down training).

            debug: For passing additional debug arguments.

            backward_kwargs: The keyword arguments passed to overall.backward.

            is_clear_energy_after_use:
                If you have several pc_layers in your model but not all of them are used in a forward pass
                (for example, pc_layer_a is used in the first call of train_on_batch() but not the second one).
                In this case, it is necessary to set this option to True. Because this tells the library to
                clear the energy after use it, so that the energy in pc_layer_a is used only in the first call
                of train_on_batch(), but not in the second one when this pc_layer_a is not even used.

            is_return_outputs: Whether return outputs from forwarding the model.

        Returns:

            A dictionary containing:
                - lists: corresponds to progress during inference, with dimension variable being t
                - single values: corresponds to a single result
        """

        # to speed up the computation, some properties are get here so that can be skipped during t
        is_model_has_pc_layers = self.get_is_model_has_pc_layers()
        model_pc_layers = list(self.get_model_pc_layers())
        model_xs = []

        self.inputs = inputs

        # sanitize model
        assert (self.get_is_model_training() == True), (
            "PCLayer behaves differently in train and eval modes, like Dropout or Batch Normalization. "
            "Thus, call model.eval() before evaluation and model.train() before train. "
            "Make sure your model is in train mode before calling <train_on_batch()>. "
            "It can be done by calling <model.train()>. "
            "Do remember switching your model back to eval mode before evaluating it by calling <model.eval()>. "
        )

        # sanitize args
        if loss_fn is not None:
            assert callable(loss_fn)

        assert isinstance(loss_fn_kwargs, dict)

        assert isinstance(is_sample_x_at_batch_start, bool)
        assert isinstance(is_reset_optimizer_x_at_batch_start, bool)
        assert isinstance(is_reset_optimizer_p_at_batch_start, bool)

        assert isinstance(is_unwrap_inputs, bool)
        if is_unwrap_inputs:
            assert isinstance(inputs, (tuple, list, dict))

        assert isinstance(is_optimize_inputs, bool)
        if is_optimize_inputs:
            assert is_model_has_pc_layers, (
                "<is_optimize_inputs> should only work with models with <PCLayer>. "
            )
            assert (not is_unwrap_inputs)

        if callback_after_backward is not None:
            assert callable(callback_after_backward)

        assert isinstance(callback_after_backward_kwargs, dict)

        if callback_after_t is not None:
            assert callable(callback_after_t)

        assert isinstance(callback_after_t_kwargs, dict)

        assert isinstance(is_log_progress, bool)
        assert isinstance(is_return_results_every_t, bool)

        assert isinstance(debug, dict)

        assert isinstance(is_return_outputs, bool)

        # create t_iterator
        if is_log_progress:
            utils.slow_down_warning(
                "PCTrainer.train_on_batch", "is_log_progress", "False")
            t_iterator = tqdm.trange(self._T)
        else:
            t_iterator = range(self._T)

        if self._is_plot_progress:
            if not is_return_results_every_t:
                warnings.warn(
                    (
                        "Note that plot_progress requires is_return_results_every_t, this has been turned on for you. "
                    ),
                    category=RuntimeWarning
                )
                is_return_results_every_t = True

        if is_return_results_every_t:
            utils.slow_down_warning(
                "PCTrainer.train_on_batch", "is_return_results_every_t", "False"
            )

        # initialize the dict for storing results
        results = {
            "loss": [],
            "energy": [],
            "overall": [],
        }
        if is_return_outputs:
            results["outputs"] = []

        is_dynamic_x_lr = ((self._x_lr_discount < 1.0)
                           or (self._x_lr_amplifier > 1.0))

        if is_dynamic_x_lr:
            overalls = []

        if is_unwrap_inputs:
            if isinstance(self.inputs, dict):
                unwrap_with = "**"
            elif isinstance(self.inputs, (list, tuple)):
                unwrap_with = "*"
            else:
                raise NotImplementedError
        else:
            unwrap_with = ""

        for t in t_iterator:

            # -> inference

            # at_batch_start
            if t == 0:

                if is_model_has_pc_layers:

                    # sample_x
                    if is_sample_x_at_batch_start:
                        for pc_layer in model_pc_layers:
                            pc_layer.set_is_sample_x(True)

                    # optimize_inputs
                    if is_optimize_inputs:
                        # convert inputs to nn.Parameter
                        assert not is_unwrap_inputs, "is_optimize_inputs should not be used with is_unwrap_inputs=True"
                        self.inputs = torch.nn.Parameter(self.inputs, True)

            # forward
            if unwrap_with == "":
                outputs = self._model(self.inputs)
            elif unwrap_with == "*":
                outputs = self._model(*self.inputs)
            elif unwrap_with == "**":
                outputs = self._model(**self.inputs)
            else:
                raise NotImplementedError

            # at_batch_start
            if t == 0:

                if is_model_has_pc_layers:

                    # sample_x
                    if is_sample_x_at_batch_start:
                        # after sample_x, optimizer_x will be reset
                        self.recreate_optimize_x()

                    else:

                        # explicitly asked to reset optimizer_x
                        if is_reset_optimizer_x_at_batch_start:
                            self.recreate_optimize_x()

                    # optimize_inputs
                    if is_optimize_inputs:
                        assert len(self._optimizer_x.param_groups) == 1
                        self._optimizer_x.param_groups[0]["params"].append(
                            self.inputs
                        )

                    # to speed up the computation, some properties are get here so that can be skipped during t
                    model_xs = list(self.get_model_xs())

                    # explicitly asked to reset optimizer_p
                    if is_reset_optimizer_p_at_batch_start:
                        self.recreate_optimize_p()

            if is_return_results_every_t or t == (self._T - 1):
                if is_return_outputs:
                    results["outputs"].append(outputs)

            # loss
            if loss_fn is not None:
                loss = loss_fn(outputs, **loss_fn_kwargs)
                if is_return_results_every_t or t == (self._T - 1):
                    results["loss"].append(loss.item())
            else:
                loss = None

            # energy
            if is_model_has_pc_layers:

                energy = sum(
                    self.get_energies(is_per_datapoint=False)
                )

                if is_clear_energy_after_use:
                    for pc_layer in model_pc_layers:
                        pc_layer.clear_energy()

                if is_return_results_every_t or t == (self._T - 1):
                    results["energy"].append(
                        energy.item()
                    )
            else:
                energy = None

            # loss_x
            if self._loss_x_fn is not None:
                loss_x_layer = []
                for model_x in model_xs:
                    loss_x_layer.append(self._loss_x_fn(model_x))
                if len(loss_x_layer) > 0:
                    loss_x = sum(loss_x_layer).sum()
                else:
                    loss_x = None
            else:
                loss_x = None

            # loss_inputs
            if self._loss_inputs_fn is not None:
                if is_optimize_inputs:
                    loss_inputs = self._loss_inputs_fn(self.inputs)
                else:
                    loss_inputs = None
            else:
                loss_inputs = None

            # overall
            overall = []
            if loss is not None:
                overall.append(loss)
            if energy is not None:
                overall.append(
                    energy * self._energy_coefficient
                )
            if loss_x is not None:
                overall.append(loss_x)
            if loss_inputs is not None:
                overall.append(loss_inputs)
            overall = sum(overall)
            if is_dynamic_x_lr:
                overalls.append(overall)
            if is_return_results_every_t or t == (self._T - 1):
                results["overall"].append(overall.item())

            # early_stop
            early_stop = eval(self._early_stop_condition)

            # _optimizer_x: zero_grad
            if is_model_has_pc_layers:
                if t in self._update_x_at:
                    self._optimizer_x.zero_grad()

            # _optimizer_p: zero_grad
            if (t in self._update_p_at) or (early_stop and self._update_p_at_early_stop):
                self._optimizer_p.zero_grad()

            # backward
            overall.backward(**backward_kwargs)

            # callback_after_backward
            if callback_after_backward is not None:
                callback_after_backward(t, **callback_after_backward_kwargs)

            # optimizer_x: step
            # x_lr_discount
            # x_lr_amplifier
            if is_model_has_pc_layers:

                if t in self._update_x_at:

                    # optimizer_x
                    self._optimizer_x.step()

                    # x_lr_discount
                    # x_lr_amplifier
                    if is_dynamic_x_lr:

                        if len(overalls) >= 2:

                            if not (overalls[-1] < overalls[-2]):
                                # x_lr_discount
                                if self._x_lr_discount < 1.0:
                                    for param_group_i in range(len(self._optimizer_x.param_groups)):
                                        self._optimizer_x.param_groups[param_group_i][
                                            'lr'
                                        ] = self._optimizer_x.param_groups[param_group_i][
                                            'lr'
                                        ] * self._x_lr_discount

                            else:
                                # x_lr_amplifier
                                if self._x_lr_amplifier > 1.0:
                                    for param_group_i in range(len(self._optimizer_x.param_groups)):
                                        self._optimizer_x.param_groups[param_group_i][
                                            'lr'
                                        ] = self._optimizer_x.param_groups[param_group_i][
                                            'lr'
                                        ] * self._x_lr_amplifier

            # optimizer_p: step
            if (t in self._update_p_at) or (early_stop and self._update_p_at_early_stop):
                self._optimizer_p.step()

            # callback_after_t
            if callback_after_t is not None:
                callback_after_t(t, **callback_after_t_kwargs)
                if is_checking_after_callback_after_t:
                    utils.slow_down_warning(
                        "PCTrainer.train_on_batch", "is_checking_after_callback_after_t", "False"
                    )
                    if not (self.get_is_model_training() == True):
                        raise RuntimeError(
                            "If you do <model.eval()> in <callback_after_t()>, you need to put model back to train mode when leaving <callback_after_t()>. "
                        )

            # log_progress
            if is_log_progress:
                log_progress = '|'
                if loss is not None:
                    log_progress += " l: {:.3e} |".format(
                        loss,
                    )
                if energy is not None:
                    log_progress += " e: {:.3e} |".format(
                        energy,
                    )
                if loss_x is not None:
                    log_progress += " x: {:.3e} |".format(
                        loss_x,
                    )
                if loss_inputs is not None:
                    log_progress += " i: {:.3e} |".format(
                        loss_inputs,
                    )
                log_progress += " o: {:.3e} |".format(
                    overall,
                )
                if is_model_has_pc_layers:
                    if self._x_lr_discount < 1.0 or self._x_lr_amplifier > 1.0:
                        x_lrs = []
                        for param_group_i in range(len(self._optimizer_x.param_groups)):
                            x_lrs.append(
                                self._optimizer_x.param_groups[param_group_i][
                                    'lr'
                                ]
                            )
                        log_progress += " x_lrs: {} |".format(
                            x_lrs,
                        )
                t_iterator.set_description(log_progress)

            # plot_progress
            if self._is_plot_progress:

                utils.slow_down_warning("PCTrainer", "plot_progress_at", "[]")

                if (isinstance(self._plot_progress_at, str) and self._plot_progress_at == "all") or (self._h in self._plot_progress_at):

                    for key in ["loss", "energy", "overall"]:
                        result = results[key]
                        if isinstance(result, list) and len(result) > 1:
                            self._plot_progress["key"].append(key)
                            self._plot_progress["h"].append(self._h)
                            self._plot_progress["t"].append(t)
                            self._plot_progress["value"].append(result[-1])

            # early_stop
            if early_stop:
                break

            # <- inference

        # plot_progress
        if self._is_plot_progress:

            utils.slow_down_warning("PCTrainer", "plot_progress_at", "[]")

            if (isinstance(self._plot_progress_at, str) and self._plot_progress_at == "all") or (isinstance(self._plot_progress_at, list) and len(self._plot_progress_at) > 0 and self._h == max(self._plot_progress_at)):

                input(
                    "Is plot progress at {}? (Set plot_progress_at=[] in creation of pc_trainer to disable this. )".format(
                        self._h
                    )
                )

                working_home = os.environ.get('WORKING_HOME')
                if working_home is None:
                    working_home = "~/"
                    warnings.warn(
                        "Please specify your working home by setting the WORKING_HOME environment variable (using absolute path if you are using ray, otherwise relative path like ~/ is fine), defaulting to {}".format(
                            working_home
                        ),
                        category=RuntimeWarning
                    )

                log_dir = os.path.join(
                    working_home, "general-energy-nets", "plot_progress"
                )

                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)

                data = pd.DataFrame(self._plot_progress)

                # debug
                # pd.set_option('display.max_rows', 500)
                # input(data)

                plt.figure()
                sns.relplot(
                    data=data,
                    x="t",
                    y="value",
                    hue="h",
                    palette="rocket_r",
                    col="key",
                    kind='line',
                    facet_kws={
                        "sharey": False,
                        "legend_out": False,
                    },
                ).set(yscale='log')
                plt.savefig(
                    os.path.join(
                        log_dir, "combined-{}.png".format(self._h)
                    )
                )

                plt.figure()
                sns.relplot(
                    data=data,
                    x="t",
                    y="value",
                    hue="h",
                    row="h",
                    palette="rocket_r",
                    col="key",
                    kind='line',
                    facet_kws={
                        "sharey": False,
                        "legend_out": False,
                    },
                ).set(yscale='log')
                plt.savefig(
                    os.path.join(
                        log_dir, "seperated-{}.png".format(self._h)
                    )
                )

                plt.close()

            self._h += 1

        return results

    #  PRIVATE METHODS  ########################################################################################################

    def _preprocess_step_index_list(
        self,
        indices: typing.Union[str, typing.List[int]],
        T: int,
    ) -> typing.List[int]:
        """Preprocesses a specification of step indices that has been provided as an argument.

        Args:
            indices (str or list[int]): The preprocessed indices, which is either a ``str`` specification or an actual
                list of indices.

        Returns:
            list[int]: A list of integer step indices.
        """

        assert isinstance(indices, (str, list))
        assert isinstance(T, int)
        assert T > 0

        if isinstance(indices, str):  # -> indices needs to be converted

            # convert indices
            if indices == "all":
                indices = list(range(T))
            elif indices == "last":
                indices = [T - 1]
            elif indices == "last_half":
                indices = list(range(T // 2, T))
            elif indices == "never":
                indices = []
            else:
                raise NotImplementedError

        else:  # -> indices is a list already

            # ensure the indices are valid
            for t in indices:
                assert isinstance(t, int)
                assert 0 <= t < T

        return indices
