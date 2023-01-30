import ray
import torch
import pprint
import copy
import os
import yaml
import numpy as np
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import utils as u
from utils import concatenate_dicts, fig_to_pil, report_via_email
import analysis_utils as au
import pandas as pd
import tqdm
import pickle
import argparse
import time

# historically, we import these functions from analysis_utils for direct usage in exec commands, but it is better to use au.<the function>
from analysis_utils import df2tb
from analysis_utils import nature_pre, nature_post, nature_catplot, nature_catplot_sharey, nature_relplot, nature_relplot_curve, add_metric_per_group, select_rows_per_group

matplotlib.use("Agg")

logger = u.getLogger(__name__)

if __name__ == "__main__":

    starting_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p",
        "--setup",
        default=[
            "sns.set_theme('talk',font_scale=1.2)",
        ],
        type=str,
        nargs="*",
        help="Setup commands that are called at the beginning of the program.")

    parser.add_argument(
        "-l",
        "--log-dir",
        default=None,
        type=str,
        help="Load results of trails from this directory.")

    parser.add_argument(
        "-m",
        "--metrics",
        default=[],
        type=str,
        nargs="*",
        help="How to reduce the dataframe of the trail to a scalar. Example: df['loss'][64].")

    parser.add_argument(
        "-f",
        "--config-file",
        default=[],
        type=str,
        nargs="*",
        help="Use configs from this file, determining what data is loaded.")

    parser.add_argument(
        "-g",
        "--group-by",
        default=[],
        type=str,
        nargs="*",
        help="Group visualization by this list of columns.")

    parser.add_argument(
        "--group-by-eval",
        default="[]",
        type=str,
        help="Group visualization by this list of columns (produced by `eval()`).")

    parser.add_argument(
        "-s",
        "--to-vis-str",
        default="./to_vis_str.json",
        type=str,
        help="Use dict from this file to map the column names to new strs for visualization.")

    parser.add_argument(
        "-r",
        "--preprocess",
        default=[],
        type=str,
        nargs="*",
        help="Exec code for preprocessing the dataframe before the dataframe is grouped. Example: df=new_col(df,'config_renamed',lambda row: {'old_name_0':'new_name_0','old_name_1':'new_name_1'}[row['config']])")

    parser.add_argument(
        "-v",
        "--visualize",
        default=[],
        type=str,
        nargs="*",
        help="How to visualize your analysis. Example: sns.barplot(data=df,x=r'lr',y=r'loss')")

    parser.add_argument(
        "-t",
        "--title",
        default="default-title",
        type=str,
        help=(
            "Title of the generated visualization."
        ))

    parser.add_argument(
        "--fig-formats",
        default=["png", "pdf"],
        type=str,
        nargs="*",
        help="Formats of the generated figure.")

    parser.add_argument(
        "-k",
        "--savefig-kwargs",
        default="{'bbox_inches':'tight'}",
        type=str,
        help="The savefig_kwargs that is passed to <analysis_utils.save_fig()>.")

    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        type=str,
        help="Output results of analysis to this dir. Default to be the folder of the first config_file.")

    parser.add_argument(
        "-d",
        "--load-dataframe",
        default=False,
        action="store_true",
        help=(
            "If load generated dataframe with the same title instead of generating from arguments -f and -l."
            "This is particularly useful when playing with visualization commands, as you do not want to waste time on loading data."
        ))

    parser.add_argument(
        "-z",
        "--visdom",
        default=False,
        action="store_true",
        help=(
            "If send analysis to visdom."
        ))

    parser.add_argument(
        "-c",
        "--comet",
        default=False,
        action="store_true",
        help=(
            "If send analysis to comet (@ https://www.comet.com/site/). See wandb for more documentations. "
        ))

    parser.add_argument(
        "-w",
        "--wandb",
        default=False,
        action="store_true",
        help=(
            "If send analysis to wandb (Weight & Bias @ https://wandb.ai/home). "
            "This will log fig and doc to wandb, with configuration specified by <-g><--group-by> or <--group-by-eval>. "
            "From wandb, it is easier to group the fig and doc. "
        ))

    args = parser.parse_args()

    # generate or process some args
    args.output_dir = os.path.dirname(args.config_file[0])

    experiment_name = args.output_dir.split('/')[-1]

    # uid is a unique identifier for runing this script (set as  the commit hash of the current git repo)
    uid = u.get_commit_hash()

    # print title
    print(" TITLE: {}; UID: {} ".format(
        args.title, uid
    ).center(
        u.WINDOW_NUM_COLUMNS, "*"
    ))

    if args.visdom:
        import visdom
        # create visdom server
        viz = visdom.Visdom()
        # Environments are automatically hierarchically organized by the first _.
        # Thus remove the first _ from the experiment name and title, and join them by _.
        viz_env = f"{os.path.basename(args.output_dir).replace('_','-')}_{args.title.replace('_','-')}"
        # clear the viz_env
        viz.delete_env(env=viz_env)

    # excute setup commands
    for p in args.setup:
        exec(p)

    analysis_df_file = os.path.join(
        args.output_dir,
        "{}.pkl".format(args.title)
    )

    if not args.load_dataframe:

        # get analysis_df
        analysis_df = au.get_analysis_df(
            au.Analysis(args.log_dir),
            metrics=args.metrics,
        )

        # load config_dicts
        config_dicts = []
        for config_file in args.config_file:
            config_dict = yaml.safe_load(
                open(config_file)
            )
            if (config_dict.get("config", None) is None) and (config_dict.get("param_space", None) is None):
                warnings.warn(
                    (
                        f"All config passing into the trainable should be moved under a entry named config (ray_paradigm: run) or param_space (ray_paradigm: fit). No abeying this is supported so far but will be deprecated soon."
                    ),
                    category=DeprecationWarning,
                )
            elif (config_dict.get("config", None) is not None) and (config_dict.get("param_space", None) is not None):
                warnings.warn(
                    (
                        f"config and param_space cannot be used at the same time."
                    ),
                    category=DeprecationWarning,
                )
            elif config_dict.get("config", None) is not None:
                config_dict = config_dict["config"]
            elif config_dict.get("param_space", None) is not None:
                config_dict = config_dict["param_space"]
            else:
                raise NotImplementedError
            config_dicts.append(
                config_dict
            )
        logger.warning(
            "Config yaml loaded. You can safely make changes to the yaml now."
        )

        # one_row_per_config
        analysis_df = analysis_df.one_row_per_config()

        # purge analysis_df
        analysis_df = au.purge_analysis_df_with_config_dicts(
            analysis_df=analysis_df,
            config_dicts=config_dicts,
            rename_dict=au.load_dict(path=args.to_vis_str),
        )

        # auto_phrase_config_for_analysis_df
        analysis_df = au.auto_phrase_config_for_analysis_df(
            analysis_df=analysis_df,
        )

        with open(analysis_df_file, 'wb') as f:
            pickle.dump(analysis_df, f)

    else:

        logger.warning(
            "You are loading a previously generated dataframe according to <-t>/<--title>. Your arguments of <-l>/<--log-dir> <-m>/<--metrics> <-f>/<--config-file> and <-s>/<--to-vis-str> will not take effect."
        )

        with open(analysis_df_file, 'rb') as f:
            analysis_df = pickle.load(f)

    config_columns = analysis_df.config_columns
    metric_columns = analysis_df.metric_columns

    df = analysis_df.dataframe
    for r in args.preprocess:
        exec(r)
    df_ungrouped = df

    # split df_ungrouped into groups
    if (len(args.group_by) > 0) or (len(eval(args.group_by_eval)) > 0):
        assert not (
            (len(args.group_by) > 0) and (len(eval(args.group_by_eval)) > 0)
        ), (
            f"You can only specify one of <-g>/<--group-by> and <--group-by-exec>. "
            f"Instead, you have args.group_by={args.group_by} and args.group_by_eval={args.group_by_eval}. "
        )
        if (len(args.group_by) > 0):
            group_by = args.group_by
        elif len(eval(args.group_by_eval)) > 0:
            group_by = eval(args.group_by_eval)
        else:
            raise NotImplementedError
        # split df_ungrouped into groups if args.group_by or args.group_by_eval is specified
        group_iterator = df_ungrouped.groupby(group_by)
    else:
        group_by = []
        group_iterator = [("", df_ungrouped)]

    # warn if the number of groups is larger than
    if len(group_iterator) > 20:

        logger.warning(
            f"You are producing a large number ({len(group_iterator)}) of groups of visualization."
        )

    # setup output main doc
    mdoc = open(
        os.path.join(
            args.output_dir,
            "{}.md".format(args.title)
        ),
        "w",
    )
    ms = ""
    figs_dict = {}

    group_iterator = tqdm.tqdm(group_iterator, leave=False)

    for id, df in group_iterator:

        if not isinstance(id, tuple):
            # when there is just one col passing into df.groupby()
            # it results single element instead of a tuple
            # make it consistent
            id = (id,)

        # get key word id
        kw_id = {}
        for i, g in enumerate(group_by):
            kw_id[g] = id[i]

        # kw_id_with_uid is kw_id with a unique id
        # it is used to seperate multiple run of analysis
        # for now, it is used with comet_ml and wandb
        kw_id_with_uid = concatenate_dicts(
            [kw_id, {'uid': uid}]
        )

        if args.comet:
            from comet_ml import Experiment
            # create agent
            comet_experiment = Experiment(
                project_name=f"{experiment_name}--{args.title}",
            )
            # log config (group_by)
            comet_experiment.log_parameters(kw_id_with_uid)

        if args.wandb:
            import wandb
            # create agent
            wandb.init(
                # Allow multiple `wandb.init()` calls in the same process.
                reinit=True,
                project=f"{experiment_name}--{args.title}",
                # log config (group_by)
                config=kw_id_with_uid,
            )

        def dict2str(d):
            str_ = ""
            for k, v in d.items():
                str_ += f"{k}={v}; "
            return str_
        kw_id_str = dict2str(kw_id)

        # print key word id
        group_iterator.set_description(
            "GROUP: {}".format(
                kw_id_str
            )
        )

        # id is a tuple, e.g., (0.1, 0.5), so format a friendly str from it
        id_str = au.format_friendly_string(str(id))
        if len(id_str) > 200:
            id_str = au.format_friendly_string(str(id), is_abbr=True)
        # e.g., 0_1_0_5

        group_title = "{}-{}".format(
            args.title, id_str,
        )

        # setup output figure
        fig = plt.figure()
        # setup output doc
        doc = open(
            os.path.join(
                args.output_dir,
                "{}.md".format(group_title)
            ),
            "w",
        )
        s = ""

        # excute visualize commands
        for v in args.visualize:
            exec(v)

        if args.visdom:
            def get_viz_kwargs(prefix):
                return {
                    'env': viz_env,
                    'win': f"{prefix}: {kw_id_str}",
                    'opts': {
                        "title": f"{prefix}: {kw_id_str}",
                        # resizable option is buggy in current version of visdom
                        # 'resizable': True,
                    },
                }

        # feed figure to comet
        if args.comet:
            # BUG -> DEPRECIATED: comet_ml does not support add image panels (can only be viewed in each experiment)
            comet_experiment.log_image(
                image_data=fig_to_pil(plt.gcf()),
                name="fig",
            )

        # feed fig to wandb
        if args.wandb:
            wandb.log({"fig": wandb.Image(plt.gcf())})

        # save figure
        paths = au.save_fig(
            logdir=args.output_dir,
            title=group_title,
            formats=args.fig_formats,
            savefig_kwargs=eval(args.savefig_kwargs),
        )

        # feed figure to visdom
        if args.visdom:
            viz.matplot(
                plt, **get_viz_kwargs('figure'),
            )

        # fill in figs_dict
        figs_dict[
            kw_id_str
        ] = [
            "![](./{}.png)".format(
                group_title
            )
        ]

        # save s
        doc.write("\n")
        doc.write(s)
        doc.write("\n")

        # feed s to visdom
        if args.visdom:
            if len(s) > 0:
                viz.text(
                    s, **get_viz_kwargs('text'),
                )

        # feed doc to wandb
        if args.wandb:
            if len(s) > 0:
                wandb.log({"doc": s})

        # close figure and doc
        plt.close()
        doc.close()

    # log figs_dict to mdoc
    ms += "# Figures"
    ms += "\n\n"
    ms += au.df2tb(
        pd.DataFrame(figs_dict)
    )

    # save ms
    mdoc.write("\n")
    mdoc.write(ms)
    mdoc.write("\n")

    # If the analysis takes more than 60 seconds, send an email to the user when it is done.
    if (time.time()-starting_time) > 60.0:
        report_via_email(
            subject=f"Analysis [{args.title}] on [{args.log_dir.split('/')[-1 if args.log_dir[-1]!='/' else -2]}] done"
        )
