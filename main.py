import os
import copy
import yaml
import argparse
import ray

import utils as u

logger = u.getLogger(__name__)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c',
        '--experiment_config',
        required=True,
        type=str,
        help='Format: experiment/config.',
    )

    parser.add_argument(
        "-l",
        "--local-mode",
        default=False,
        action='store_true',
        help="Run in local mode (only one thread is running at a time)."
    )

    args = parser.parse_args()

    # load config
    with open('experiments/{}.yaml'.format(args.experiment_config)) as f:
        config = yaml.safe_load(f)
    # # warning after load config
    logger.warning(
        "Config yaml loaded. You can safely make changes to the yaml now. \n"
    )

    # for historical experiments, there were default imports of trainables, importing here
    # to avoid breaking old experiments
    from base_trainable import BaseTrainable
    from dataset_learning_trainable import DatasetLearningTrainable
    from supervised_learning_trainable import SupervisedLearningTrainable
    from any_energy_trainable import AnyEnergyTrainable
    from hand_coded_rules_trainable import HandCodedRulesTrainable

    # main_import_code
    if config.get('main_import_code', None) is not None:
        exec(config['main_import_code'])
        config.pop('main_import_code')

    # ray_init_kwargs
    if config.get("ray_init_kwargs", None) is not None:
        ray_init_kwargs = copy.deepcopy(config["ray_init_kwargs"])
        config.pop("ray_init_kwargs")

        # depreciated warnings
        if ray_init_kwargs.get('num_cpus', None) is not None:
            if isinstance(ray_init_kwargs.get('num_cpus'), str):
                logger.warning((
                    f"parsing <num_cpus> from str has been depreciated, removing now. "
                ))
                ray_init_kwargs.pop('num_cpus')

    else:
        ray_init_kwargs = {}

    # args.local_mode
    if args.local_mode:
        ray_init_kwargs.update({"local_mode": True})

    # ray.init
    ray.init(**ray_init_kwargs)

    ray_paradigm = config.get("ray_paradigm", None)

    if ray_paradigm is None:

        # depreciated warnings
        ray_paradigm = "run"
        logger.warning(
            "ray_paradigm not specified, defaulting to 'run'. It is recommended to specify this. "
        )

    else:
        config.pop("ray_paradigm")

    local_dir = os.path.join(os.environ.get('RESULTS_DIR'))

    name = args.experiment_config.split('/')[0]

    try:
        exec(
            f"import experiments.{name}.utils as eu"
        )
    except Exception as e:
        logger.warning((
            "the recommended workflow is to use eu.Trainable as your <trainable> or <run_or_experiment>, where eu is imported automatically for your from <your_experiment/utils.py>. "
            f"But `import experiments.{name}.utils as eu` fails with the following error: \n"
            f"{e} "
        ))

    # before_run_code
    before_run_code = None
    if config.get('before_run_code', None) is not None:
        before_run_code = copy.deepcopy(config['before_run_code'])
        config.pop('before_run_code')

    # after_run_code
    after_run_code = None
    if config.get('after_run_code', None) is not None:
        after_run_code = copy.deepcopy(config['after_run_code'])
        config.pop('after_run_code')

    # before_run_code
    if before_run_code is not None:
        exec(before_run_code)

    if ray_paradigm == "run":

        # depreciated warnings
        if config.get('resources_per_trial', None) is not None:
            if config['resources_per_trial'].get('gpu', None) is not None:
                if isinstance(config['resources_per_trial']['gpu'], str):
                    logger.warning((
                        f"parsing <gpu> from str has been depreciated, setting to 0.125 now. "
                    ))
                    config['resources_per_trial']['gpu'] = 0.125

        # config['local_dir']
        config['local_dir'] = local_dir

        # config['run_or_experiment']
        # # depreciation warnings
        if config.get('Trainable', None) is not None:
            logger.warning((
                f"<Trainable> has been deprecated and replaced by <run_or_experiment>. "
            ))
            config['run_or_experiment'] = config['Trainable']
        if config.get('run_or_experiment', 'GenTrainable') == "GeneRecTrainable":
            logger.warning((
                f"<GeneRecTrainable> has been deprecated and replaced by <HandCodedRulesTrainable>. "
            ))
        # # eval run_or_experiment
        run_or_experiment = config.get('run_or_experiment', None)
        if run_or_experiment is None:
            try:
                run_or_experiment = eu.Trainable
            except Exception as e:
                raise RuntimeError(
                    "run_or_experiment not specified, and using eu.Trainable failed with the followig error: \n"
                    f"{e} "
                )
        else:
            assert isinstance(run_or_experiment, str), (
                f"run_or_experiment should be a string, but got {run_or_experiment} "
            )
            if 'eu.' not in run_or_experiment:
                logger.warning(
                    "the recommended workflow is to use eu.Trainable as your <run_or_experiment>, where eu is imported automatically for your from <your_experiment/utils.py>. "
                    f"But you are using a custom <run_or_experiment>: {run_or_experiment}"
                )
            run_or_experiment = eval(
                run_or_experiment
            )
        config['run_or_experiment'] = run_or_experiment

        # config['name']
        # # name is the experiment from experiment_config
        config['name'] = name

        # config['stop']
        if config.get('stop', None) is not None:
            if isinstance(config['stop'], str):
                config['stop'] = eval(config['stop'])

        # config['scheduler']
        if config.get('scheduler', None) is not None:
            config['scheduler'] = eval(
                config['scheduler']
            )

        # config['global_checkpoint_period']
        if config.get('global_checkpoint_period', None) is not None:
            # # global_checkpoint_period counld be "np.inf", so eval it
            config['global_checkpoint_period'] = eval(
                config['global_checkpoint_period']
            )

        # config
        if config.get('config', None) is None:
            logger.warning((
                f"All config passing into the trainable should be moved under a entry named config. "
            ))

        # callbacks
        if len(config.get('callbacks', [])) > 0:
            for i in range(len(config['callbacks'])):
                config['callbacks'][i] = eval(
                    config['callbacks'][i]
                )

        analysis = ray.tune.run(
            **config,
        )

    elif ray_paradigm == "fit":

        is_resume = config['is_resume']
        config.pop('is_resume')
        restore_kwargs = config['restore_kwargs']
        config.pop('restore_kwargs')

        if not is_resume:

            trainable = config.get('trainable', None)
            if trainable is None:
                try:
                    trainable = eu.Trainable
                except Exception as e:
                    raise RuntimeError(
                        "trainable not specified, and using eu.Trainable failed with the followig error: \n"
                        f"{e} "
                    )
            else:
                assert isinstance(trainable, str), (
                    f"trainable should be a string, but got {type(trainable)}."
                )
                if 'eu.' not in trainable:
                    logger.warning(
                        "the recommended workflow is to use eu.Trainable as your <trainable>, where eu is imported automatically for your from <your_experiment/utils.py>. "
                        f"But you are using a custom <trainable>: {trainable}"
                    )
                trainable = eval(trainable)
            config['trainable'] = trainable
            config['tune_config'] = eval(config['tune_config'])
            config['run_config'] = eval(config['run_config'])

            tuner = ray.tune.Tuner(
                **config,
            )

        else:

            tuner = ray.tune.Tuner.restore(
                path=os.path.join(os.environ.get('RESULTS_DIR'), name),
                **restore_kwargs,
            )

        results = tuner.fit()

    else:
        raise ValueError(f"ray_paradigm {ray_paradigm} not supported.")

    # after_run_code
    if after_run_code is not None:
        exec(after_run_code)
