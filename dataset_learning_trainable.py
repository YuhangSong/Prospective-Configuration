from __future__ import print_function

import os
import ray
import copy
import math
import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms

import utils
import utils as u
import analysis_utils
import analysis_utils as au
import dataset_utils
import dataset_utils as du

from base_trainable import BaseTrainable


logger = u.getLogger(__name__)


class DatasetLearningTrainable(BaseTrainable):
    """DatasetLearningTrainable.

        This class is abstractive so cannot be directly used via yaml config.
            It should be used as a base class.

        Manage:
            - data_packs
                - data_loader (Tested)
                - at_iteration (Tested)
                - do (Tested)
            - code added around iteration
            - log_packs
                - log (Tested)
                - log_fn (Tested)
                - at_iteration (Tested)
                - at_data_pack (Tested)
                - at_batch_idx (Tested)
                - summarize_over_batch_idx_fn (Tested)
            - log_key_holders
            - num_iterations
    """

    def setup(self, config):

        u.assert_config_all_valid(self.config)

        super(DatasetLearningTrainable, self).setup(config)

        exec(self.config.get("before_DatasetLearningTrainable_setup_code", "pass"))

        self.reset_data_packs()
        self.reset_log_packs()
        self.reset_log_key_holders()
        self.reset_after_iteration_fn()

        exec(self.config.get("after_DatasetLearningTrainable_setup_code", "pass"))

        u.assert_config_all_valid(self.config)

    def reset_data_packs(self):

        code_to_run = self.config.get(
            "before_DatasetLearningTrainable_creating_data_packs_code", None
        )
        if code_to_run is None:
            code_to_run = self.config.get(
                "before_DatasetLearningTrainable_setup_code", None
            )
            if code_to_run is not None:
                logger.warning(
                    f"before_DatasetLearningTrainable_creating_data_packs_code is not specified, but before_DatasetLearningTrainable_setup_code is specified. "
                    f"use before_DatasetLearningTrainable_setup_code instead as before_DatasetLearningTrainable_creating_data_packs_code for backward compatibility. "
                    f"Move your code for creating data_packs to before_DatasetLearningTrainable_creating_data_packs_code in the future. "
                )
        if code_to_run is not None:
            exec(code_to_run)

        self.is_data_loader_in_raypg = self.config.get(
            "is_data_loader_in_raypg", False
        )

        # create data_packs if specified
        self.data_packs = copy.deepcopy(self.config.get('data_packs', {}))

        # self.data_packs is a dict
        for data_pack_key, data_pack in self.data_packs.items():
            # each data_pack is a dict, some entries of which should be eval or have default values

            assert isinstance(data_pack["data_loader"], str), (
                f"data_loader should be a string, but got {data_pack['data_loader']}"
            )
            # data_loader should be eval
            data_loader = eval(
                data_pack["data_loader"]
            )

            data_pack["data_loader"] = ray.put(
                data_loader
            ) if self.is_data_loader_in_raypg else data_loader

            # by default, it takes effect at all iterations
            at_iteration = data_pack.get(
                "at_iteration", "'all'"
            )
            assert isinstance(at_iteration, str), (
                f"at_iteration should be a string, but got {at_iteration}"
            )
            data_pack["at_iteration"] = eval(at_iteration)
            if isinstance(data_pack["at_iteration"], list):
                for item in data_pack["at_iteration"]:
                    assert isinstance(item, int), (
                        f'each item in at_iteration should be an integer, but got {item}'
                    )
            elif isinstance(data_pack["at_iteration"], str):
                assert data_pack["at_iteration"] in ["all"], (
                    f"at_iteration should be 'all' if it is a string, but got {data_pack['at_iteration']}"
                )
            else:
                raise NotImplementedError(
                    f"at_iteration should be a list of integers or 'all', but got {data_pack['at_iteration']}"
                )

            # by default, it does the following things in order
            do = data_pack.get(
                "do", "None"
                # other options may be specified by subclasses
                # see implementation of iteration_step() in subclasses
            )
            assert isinstance(do, str), (
                f"do should be a string, but got {do}"
            )
            data_pack["do"] = eval(do)
            if data_pack["do"] is None:
                data_pack["do"] = ['predict', 'learn']
                logger.warning(
                    f"do for datapacks[{data_pack_key}] is not specified, default to {data_pack['do']}, this will be deprecated in the future and raise an error."
                )
            if isinstance(data_pack["do"], list):
                for item in data_pack["do"]:
                    assert isinstance(item, str), (
                        f'each item in do should be a string, but got {item}'
                    )
            else:
                raise NotImplementedError(
                    f"do should be a list of strings, but got {data_pack['do']}"
                )

        exec(
            self.config.get(
                "after_DatasetLearningTrainable_creating_data_packs_code", "pass"
            )
        )

    def reset_log_packs(self):

        # create log_packs if specified
        self.log_packs = copy.deepcopy(self.config.get('log_packs', {}))

        # self.log_packs is a dict
        for _, log_pack in self.log_packs.items():

            if "log_fn" in log_pack.keys():
                assert isinstance(log_pack["log_fn"], str), (
                    f"log_fn should be a string, but got {log_pack['log_fn']}"
                )
                log_pack["log_fn"] = eval(log_pack["log_fn"])
                assert callable(log_pack["log_fn"]), (
                    f"log_fn should be callable, but got {log_pack['log_fn']}"
                )
            elif "log" in log_pack.keys():
                assert isinstance(log_pack["log"], str), (
                    f"log should be a string, but got {log_pack['log']}"
                )
            else:
                raise NotImplementedError
            assert not (
                "log_fn" in log_pack.keys() and "log" in log_pack.keys()
            ), ("log_fn and log cannot be specified at the same time")

            # each log_pack is a dict, some entries of which should be eval or have default values
            # # log only takes effect at specific iterations
            # # by default, it takes effect at all iterations
            at_iteration = log_pack.get(
                "at_iteration", "'all'"
            )
            assert isinstance(at_iteration, str), (
                f"at_iteration should be a string, but got {at_iteration}"
            )
            log_pack["at_iteration"] = eval(at_iteration)
            if isinstance(log_pack["at_iteration"], list):
                for item in log_pack["at_iteration"]:
                    assert isinstance(item, int), (
                        f"at_iteration should be a list of integers, but got {item}"
                    )
            elif isinstance(log_pack["at_iteration"], str):
                assert log_pack["at_iteration"] in ["all"], (
                    f"at_iteration should be 'all' if it is a string, but got {log_pack['at_iteration']}"
                )
            else:
                raise NotImplementedError(
                    f"at_iteration should be a list of integers or 'all', but got {log_pack['at_iteration']}"
                )

            # # log only takes effect for specific data_packs
            # # by default, it takes effect for all data_packs
            at_data_pack = log_pack.get(
                "at_data_pack", str(list(self.data_packs.keys()))
            )
            assert isinstance(at_data_pack, str), (
                f"at_data_pack should be a string, but got {at_data_pack}"
            )
            log_pack["at_data_pack"] = eval(
                at_data_pack
            )
            if isinstance(log_pack["at_data_pack"], list):
                for item in log_pack["at_data_pack"]:
                    assert item in list(self.data_packs.keys()), (
                        f"at_data_pack should be a list of data_pack names in {list(self.data_packs.keys())}, but got {item}"
                    )
            else:
                raise NotImplementedError(
                    f"at_data_pack should be a list of data_pack names, but got {log_pack['at_data_pack']}"
                )

            # # log only takes effect at specific batch_idx
            # # by default, it takes effect at all batch_idx (at_batch_idx="all") and produces a list
            # # such a list is summarized later
            at_batch_idx = log_pack.get(
                "at_batch_idx", "'all'"
            )
            assert isinstance(at_batch_idx, str), (
                f"at_batch_idx should be a string, but got {at_batch_idx}"
            )
            log_pack["at_batch_idx"] = eval(
                at_batch_idx
            )
            if isinstance(log_pack["at_batch_idx"], str):
                assert log_pack["at_batch_idx"] in ["all"], (
                    f"at_batch_idx should be 'all' if it is a string, but get {log_pack['at_batch_idx']}"
                )
            elif isinstance(log_pack["at_batch_idx"], list):
                for item in log_pack["at_batch_idx"]:
                    assert isinstance(item, int), (
                        f"each item in at_batch_idx should be an integer, but get {item}"
                    )
            else:
                raise NotImplementedError(
                    f"at_batch_idx should be 'all' or a list of integers, but get {log_pack['at_batch_idx']}"
                )

            # # summarize the logs over batch_idx
            # # by default it is summarized by mean
            summarize_over_batch_idx_fn = log_pack.get(
                "summarize_over_batch_idx_fn", "lambda x: np.mean(x)"
            )
            assert isinstance(summarize_over_batch_idx_fn, str), (
                f"summarize_over_batch_idx_fn should be a string, but get {summarize_over_batch_idx_fn}"
            )
            log_pack["summarize_over_batch_idx_fn"] = eval(
                summarize_over_batch_idx_fn
            )
            assert callable(log_pack["summarize_over_batch_idx_fn"]), (
                f"summarize_over_batch_idx_fn should be callable, but get {log_pack['summarize_over_batch_idx_fn']}"
            )

        exec(
            self.config.get(
                "after_DatasetLearningTrainable_creating_log_packs_code", "pass"
            )
        )

    def reset_log_key_holders(self):
        # holders for log_pack that is not logger at the very first start
        # this is a problem from ray
        log_key_holders = self.config.get("log_key_holders", "[]")
        assert isinstance(log_key_holders, str), (
            f"log_key_holders should be a string, but get {log_key_holders}"
        )
        self._log_key_holders = eval(
            log_key_holders
        )

    def reset_after_iteration_fn(self):
        after_iteration_fn = self.config.get("after_iteration_fn", "None")
        assert isinstance(after_iteration_fn, str), (
            f"after_iteration_fn should be a string, but get {after_iteration_fn}"
        )
        self.after_iteration_fn = eval(
            after_iteration_fn
        )

    def reset_config(self, new_config):

        super().reset_config(new_config)
        reuse_actors_config = self.config['reuse_actors_config']

        exec(self.config.get("before_DatasetLearningTrainable_reset_config_code", "pass"))

        if reuse_actors_config.get('is_reset_data_packs', True):
            self.reset_data_packs()

        if reuse_actors_config.get('is_reset_log_packs', True):
            self.reset_log_packs()

        if reuse_actors_config.get('is_reset_log_key_holders', True):
            self.reset_log_key_holders()

        if reuse_actors_config.get('is_reset_after_iteration_fn', True):
            self.reset_after_iteration_fn()

        exec(self.config.get("after_DatasetLearningTrainable_reset_config_code", "pass"))

        u.assert_config_all_valid(self.config)

        return True

    def step(self):

        # hold results to return
        # key should be string, value should be numbers or str
        result_dict = {}

        if self.config.get("before_iteration_data_packs_code", None) is not None:
            exec(self.config["before_iteration_data_packs_code"])

        # iterate over self.data_packs
        for data_pack_key, data_pack in self.data_packs.items():

            # only take effect at some specified iterations
            if (data_pack["at_iteration"] == "all") or (self._iteration in data_pack["at_iteration"]):

                if self.config.get("before_iteration_data_loader_code", None) is not None:
                    exec(self.config["before_iteration_data_loader_code"])

                # iterate data_loader
                for batch_idx, batch in enumerate(ray.get(data_pack["data_loader"]) if self.is_data_loader_in_raypg else data_pack["data_loader"]):

                    # get batch and put it to device
                    batch = list(batch)
                    for batch_item_i in range(len(batch)):
                        batch[batch_item_i] = batch[batch_item_i].to(
                            self.device
                        )
                    batch = tuple(batch)

                    if self.config.get("before_iteration_code", None) is not None:
                        exec(self.config["before_iteration_code"])

                    step_iteration_result_dict = {}
                    for do_key in data_pack["do"]:
                        step_iteration_return = self.iteration_step(
                            data_pack_key=data_pack_key,
                            batch_idx=batch_idx,
                            batch=batch,
                            do_key=do_key,
                        )
                        if isinstance(step_iteration_return, dict):
                            step_iteration_result_dict.update(
                                step_iteration_return
                            )

                    if self.after_iteration_fn is not None:
                        self.after_iteration_fn(self, data_pack_key, batch_idx)
                    if self.config.get("after_iteration_code", None) is not None:
                        exec(self.config["after_iteration_code"])

                    ########################################################################
                    ## logging #############################################################
                    ########################################################################

                    for log_pack_key, log_pack in self.log_packs.items():
                        # only take effect at specific iteration
                        if (log_pack["at_iteration"] == "all") or (self._iteration in log_pack["at_iteration"]):
                            # only take effect at specific data_pack
                            if (data_pack_key in log_pack["at_data_pack"]):
                                # only take effect at specific batch_idx
                                if (isinstance(log_pack["at_batch_idx"], str) and log_pack["at_batch_idx"] == "all") or (isinstance(log_pack["at_batch_idx"], list) and batch_idx in log_pack["at_batch_idx"]):
                                    if "log_fn" in log_pack.keys():
                                        result = log_pack["log_fn"](self)
                                    elif "log" in log_pack.keys():
                                        result = eval(log_pack["log"])
                                    else:
                                        raise NotImplementedError(
                                            f"no log_fn or log in log_pack {log_pack_key}"
                                        )
                                    # iterations are represented by ray
                                    # log_pack need to be represented in result_key
                                    # batch_idx will be summarized
                                    result_key = "{}__{}".format(
                                        data_pack_key, log_pack_key
                                    )
                                    if result_key not in result_dict.keys():
                                        result_dict[result_key] = []
                                    result_dict[result_key].append(result)

                    ########################################################################

                    if data_pack.get("num_batches_per_epoch", -1) > 0:
                        if batch_idx == (data_pack["num_batches_per_epoch"]-1):
                            break

                if self.config.get("after_iteration_data_loader_code", None) is not None:
                    exec(self.config["after_iteration_data_loader_code"])

                # result_dict is summarized to reduce batch_idx (on list index)
                for result_key in result_dict.keys():
                    result_dict[result_key] = self.log_packs[result_key.split("__")[1]]["summarize_over_batch_idx_fn"](
                        result_dict[result_key]
                    )

        if self.config.get("after_iteration_data_packs_code", None) is not None:
            exec(self.config["after_iteration_data_packs_code"])

        # Due to ray's problems, metrics that are not logged at the first iteration but
        # only logged in later iterations will not be recorded. This can be solved by
        # ```log_key_holders```.
        if self._iteration == 0:
            for log_key_holder in self._log_key_holders:
                if ':' in log_key_holder:
                    logger.warning(
                        "log_key_holders should not contain ':', because this has been depreciated, use '__' instead. I will replace it for you."
                    )
                    log_key_holder = log_key_holder.replace(
                        ': ', '__'
                    ).replace(
                        ':', '__'
                    )
                result_dict[log_key_holder] = None

        result_dict = self.manage_num_iterations(result_dict)

        # return results
        return result_dict

    def iteration_step(
        self,
        data_pack_key,
        batch_idx,
        batch,
        do_key,
    ):

        raise NotImplementedError

    def cleanup(self):
        super().cleanup()
