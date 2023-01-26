import os
import sys
import copy
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import seaborn as sns

import utils as u
import analysis_utils as au

from ray.tune import Trainable


logger = u.getLogger(__name__)


class BaseTrainable(Trainable):
    """BaseTrainable.

        This class is abstractive so cannot be directly used via yaml config.
            It should be used as a base class.

        Manage:
            - seed (Tested)
            - device
            - num_iterations (Tested)
                To use this, you need to set in your yaml:
                    stop:
                        is_num_iterations_reached: 1
            - save_checkpoint_code
            - load_checkpoint_code
            - wandb: init a wandb agent
    """

    def setup(self, config):

        u.assert_config_all_valid(self.config)

        exec(self.config.get("before_BaseTrainable_setup_code", "pass"))

        if self.config.get("wandb", None) is not None:
            import wandb
            # create agent
            wandb.init(
                config=config,
                **self.config['wandb'],
            )

        self.reset_device()
        self.reset_seed()

        # depreciate warnings

        # # deterministic
        if self.config.get("deterministic", None) is not None:
            logger.warning(
                "deterministic has been depreciated, it will not take any effect"
            )

        # # train_code
        if (self.config.get("train_code", None) is not None):
            logger.warning(
                "train_code has been depreciated and now called step_code."
            )
        if (self.config.get("train_code", None) is not None) and (self.config.get("step_code", None) is not None):
            logger.error(
                "train_code has been depreciated and now called step_code, don't use both"
            )

        # # stop_code
        if (self.config.get("stop_code", None) is not None):
            logger.warning(
                "stop_code has been depreciated and now called cleanup_code."
            )
        if (self.config.get("stop_code", None) is not None) and (self.config.get("cleanup_code", None) is not None):
            logger.error(
                "stop_code has been depreciated and now called cleanup_code, don't use both"
            )

        exec(self.config.get("after_BaseTrainable_setup_code", "pass"))

        u.assert_config_all_valid(self.config)

    def reset_device(self):

        self.device = eval(self.config.get("device", "torch.device('cpu')"))

    def reset_seed(self):

        self.seed = self.config.get("seed", 5434)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # the following results in error on some platforms, so comment out

        # # see https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking
        # torch.backends.cudnn.benchmark = False

        # # see https://pytorch.org/docs/stable/notes/randomness.html#avoiding-nondeterministic-algorithms
        # torch.use_deterministic_algorithms(True)

        # # see https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-determinism
        # torch.backends.cudnn.deterministic = True

        # Note that the controling of seed is not perfect, see https://pytorch.org/docs/stable/notes/randomness.html#reproducibility

    def reset_config(self, new_config):

        self.config = new_config
        reuse_actors_config = self.config['reuse_actors_config']

        exec(self.config.get("before_BaseTrainable_reset_config_code", "pass"))

        if reuse_actors_config.get('is_reset_seed', True):
            self.reset_seed()

        if reuse_actors_config.get('is_reset_device', True):
            self.reset_device()

        exec(self.config.get("after_BaseTrainable_reset_config_code", "pass"))

        u.assert_config_all_valid(self.config)

        return True

    def manage_num_iterations(self, result_dict):

        if self.config.get("num_iterations", None) is not None:

            if self._iteration >= (self.config['num_iterations'] - 1):
                result_dict['is_num_iterations_reached'] = 1
            else:
                result_dict['is_num_iterations_reached'] = 0

        return result_dict

    def step(
        self,
        # hold results to return
        # key should be string, value should be numbers or str
        result_dict={},
    ):

        if (self.config.get("step_code", None) is not None):
            exec(self.config["step_code"])

        elif (self.config.get("train_code", None) is not None):
            exec(self.config["train_code"])

        else:
            logger.warning("nothing is done in step.")

        result_dict = self.manage_num_iterations(result_dict)

        return result_dict

    def cleanup(self):

        if (self.config.get("cleanup_code", None) is not None):
            exec(self.config["cleanup_code"])

        elif (self.config.get("stop_code", None) is not None):
            exec(self.config["stop_code"])

        else:
            logger.info("nothing is done in cleanup.")

    def save_checkpoint(self, tmp_checkpoint_dir):

        exec(self.config.get("save_checkpoint_code", "pass"))
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):

        exec(self.config.get("load_checkpoint_code", "pass"))
