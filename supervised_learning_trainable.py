from __future__ import print_function

import os
import copy
import pprint
import torch
import numpy as np
import matplotlib.pyplot as plt

import utils
import utils as u
import analysis_utils
import analysis_utils as au

from dataset_learning_trainable import DatasetLearningTrainable

import torch.nn as nn
import torch.nn.functional as F


class SupervisedLearningTrainable(DatasetLearningTrainable):
    """SupervisedLearningTrainable.

        Manage:
            - model_creation_code.
            - predict_code.
            - learn_code.
    """

    def setup(self, config):

        super(SupervisedLearningTrainable, self).setup(config)

        exec(self.config.get("before_SupervisedLearningTrainable_setup_code", "pass"))

        # depreciation warnings

        self.reset_model()

        exec(self.config.get("after_SupervisedLearningTrainable_setup_code", "pass"))

    def reset_model(self):

        exec(self.config.get("model_creation_code", "pass"))

    def reset_config(self, new_config):

        super().reset_config(new_config)
        reuse_actors_config = self.config['reuse_actors_config']

        exec(self.config.get(
            "before_SupervisedLearningTrainable_reset_config_code", "pass"))

        if reuse_actors_config.get('is_reset_model', True):
            self.reset_model()

        exec(self.config.get(
            "after_SupervisedLearningTrainable_reset_config_code", "pass"))

        return True

    def iteration_step(
        self,
        data_pack_key,
        batch_idx,
        batch,
        do_key,
    ):

        # unpack batch
        data, target = batch

        if do_key == 'predict':

            exec(self.config.get("predict_code", "pass"))

        elif do_key == 'learn':

            exec(self.config.get("learn_code", "pass"))

        else:

            raise NotImplementedError
