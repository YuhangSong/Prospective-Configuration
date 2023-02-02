import math
import glob
import logging
import os
import re
import smtplib
import ssl
import subprocess
import time
import copy
import collections
import json

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

import numpy as np
import pandas as pd
import requests

from email.mime.text import MIMEText
from PIL import Image

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_uniform_
from torch.utils.data import DataLoader, TensorDataset

from torchvision import transforms

matplotlib.use('agg')

'''
    logger
'''


def getLogger(logger_name):

    logging.basicConfig(
        level=logging.WARNING,
        format='\n# %(levelname)s (%(name)s): %(message)s\n',
    )

    logger = logging.getLogger(logger_name)

    return logger


logger = getLogger(__name__)

'''
    For job management.
        - split_config.py
        - submit_as_jobs.py
'''


# def args_as_python_command_str(args, except_args=[]):

#     return dict_as_python_command_str(vars(args), except_args)


# def dict_as_python_command_str(dict_, except_args=[]):

#     assert isinstance(dict_, dict)
#     assert isinstance(except_args, list)

#     python_command_str = ""
#     for key, value in dict_.items():
#         if key not in except_args:

#             if isinstance(value, list):
#                 value = " ".join(str(value_item) for value_item in value)
#             else:
#                 value = "{}".format(value)

#             python_command_str += "--{} {} ".format(key, value)

#     # remove the addition space at the end
#     python_command_str = python_command_str[:-1]

#     return python_command_str


# def format_python_command_str(base_python_command, args=None, except_args=[], additional_args_dict=None):

#     assert isinstance(base_python_command, str)
#     assert base_python_command[:6] == "python"
#     assert base_python_command.split(' ')[1][-3:] == ".py"

#     command = base_python_command

#     if args is not None:
#         command += " "
#         command += args_as_python_command_str(args, except_args)

#     if additional_args_dict is not None:
#         command += " "
#         command += dict_as_python_command_str(additional_args_dict)

#     return command


# def format_job_name(any_str):
#     """Format any string to a job name.
#         Following:
#             - Job names must consist of lower case alphanumeric characters or "-" and start with an alphabetic character (e.g. "my-name",  or "abc-123")
#             - Max length of 53
#     """
#     name = "j-" + re.sub("[^0-9a-z]+", "-", any_str.lower())
#     if len(name) > 20:
#         name = name.split("-")[0] + "-" + "".join(
#             [word[0] for word in name.split("-")[1:-1]]
#         ) + "-" + name.split("-")[-1]
#     return name


# def add_split_suffix(experiment_config, split):

#     return "{}-split-{}".format(
#         experiment_config,
#         split,
#     )


'''
    For general purposes.
'''


def init_yuhang(w, C=0.01, A=1.0):

    assert isinstance(w, torch.Tensor)
    assert w.dim() == 2

    assert isinstance(C, float)
    assert 1 > C > 0

    fan_in = w.size(1)
    fan_out = w.size(0)

    sign = torch.normal(
        mean=torch.zeros_like(w),
        std=torch.ones_like(w),
    ).sign()

    w.uniform_(
        A/fan_in*C,
        A/fan_in,
    )

    w.mul_(sign)


def init_xavier_uniform_reciprocal_friendly_(w, gain=1, C=0.01, is_test=False):
    """Initialize w using Xavier uniform reciprocal friendly.

    Args:
        w (torch.Tensor): Weights.
        gain (float): Gain.
        C (float): Minimul absolute value / maximual absolute value.
        is_test (bool): Whether it is test mode.
    """
    assert isinstance(w, torch.Tensor)

    fan_in, fan_out = _calculate_fan_in_and_fan_out(w)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    # Calculate uniform bounds from standard deviation
    a = math.sqrt(3.0) * std

    sign = torch.normal(
        mean=torch.zeros_like(w),
        std=torch.ones_like(w),
    ).sign()

    if is_test:
        sign[0].fill_(1.0)
        sign[1].fill_(1.0)
        sign[2].fill_(-1.0)
        sign[3].fill_(-1.0)

    w.uniform_(
        a*C,
        a,
    )

    if is_test:
        w[0].fill_(a*C)
        w[1].fill_(a)
        w[2].fill_(a*C)
        w[3].fill_(a)

    w.mul_(sign)

    return a*a*C


def get_error_undershot_only(target, prediction):
    """Get error, but only for undershot cases."""
    error = target-prediction
    undershot_sign = (
        target-target.mean(dim=1, keepdim=True)
    ).sign()
    error_sign = error.sign()
    sign_match = (undershot_sign*error_sign).clamp(min=0)
    return error*sign_match


def dict_values(d):
    """Yield all values in a nested dict.
    """
    for v in d.values():
        if isinstance(v, dict):
            yield from dict_values(v)
        else:
            yield v


def get_commit_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")


def hardcode_w_update(w, x_pre, e_post, lr, acf, acf_at, weight_decay=0.0):
    """
    Hardcode w update.
    """

    assert isinstance(w, torch.Tensor), (
        f'Expected w to be a torch.Tensor, but got {type(w)}')
    assert isinstance(x_pre, torch.Tensor), (
        f'Expected x_pre to be a torch.Tensor, but got {type(x_pre)}')
    assert isinstance(e_post, torch.Tensor), (
        f'Expected e_post to be a torch.Tensor, but got {type(e_post)}')

    assert w.dim() == 2, (
        f'Expected w to be a 2D tensor, but got {w.dim()}D')
    assert x_pre.size(1) == w.size(0), (
        f'Expected x_pre.size(1) to be {w.size(0)}, but got {x_pre.size(1)}')
    assert e_post.size(1) == w.size(1), (
        f'Expected e_post.size(1) to be {w.size(1)}, but got {e_post.size(1)}')

    assert isinstance(lr, float), (
        f'Expected lr to be a float, but got {type(lr)}')
    assert 0.0 < lr, (
        f'Expected lr to be positive, but got {lr}')
    assert isinstance(acf, nn.Module), (
        f'Expected acf to be a nn.Module, but got {type(acf)}')
    assert isinstance(acf_at, str), (
        f'Expected acf_at to be a str, but got {type(acf)}')

    assert isinstance(weight_decay, float), (
        f'Expected weight_decay to be a float, but got {type(weight_decay)}')
    assert 0.0 <= weight_decay, (
        f'Expected weight_decay to be non-negative, but got {weight_decay}')

    if acf_at == 'pre':

        # y = acf(x_pre) * w
        # loss = 0.5 * (y - target)**2
        # e_post = y - target
        # d loss / d w = d loss / d y * d y / d w
        #              = e_post * acf(x_pre)
        # so, w -= acf(x_pre) * e_post * lr

        bmm_result = torch.bmm(
            acf(
                x_pre
            ).unsqueeze(2),
            (
                (
                    e_post
                )
            ).unsqueeze(1)
        )

    elif acf_at == 'post':

        # y = acf(x_pre * w)
        # loss = 0.5 * (y - target)**2
        # e_post = y - target
        # d loss / d w = d loss / d y * d y / d w
        #              = e_post * acf'(x_pre*w) * x_pre
        # so, w -= x_pre * (e_post * acf'(x_pre*w)) * lr

        bmm_result = torch.bmm(
            (
                x_pre
            ).unsqueeze(2),
            (
                (
                    e_post
                )*grad(acf)(
                    torch.matmul(x_pre, w)
                )
            ).unsqueeze(1)
        )

    elif acf_at == 'both':

        # y = acf(x_pre) * acf(x_pre * w)
        # loss = 0.5 * (y - target)**2
        # e_post = y - target
        # d loss / d w = d loss / d y * d y / d w
        #              = e_post * acf'(acf(x_pre*w)) * acf(x_pre)
        # so, w -= acf(x_pre) * (e_post * acf'(acf(x_pre)*w)) * lr

        bmm_result = torch.bmm(
            (
                acf(x_pre)
            ).unsqueeze(2),
            (
                (
                    e_post
                )*grad(acf)(
                    torch.matmul(acf(x_pre), w)
                )
            ).unsqueeze(1)
        )

    else:
        raise NotImplementedError

    d_p = bmm_result.sum(dim=0)

    if weight_decay != 0:
        d_p = d_p.add(w, alpha=weight_decay)

    w.add_(d_p, alpha=-lr)

    return bmm_result


def get_dicts_keys_overlap(dict1, dict2):
    """
    Get the overlapping keys of two dicts.
    """
    assert isinstance(dict1, dict)
    assert isinstance(dict2, dict)
    return get_lists_overlap(list(dict1.keys()), list(dict2.keys()))


def get_lists_overlap(list1, list2):
    """
    Returns the overlap of two lists.
    :param list1: first input list
    :param list2: second input list
    :return: list containing elements that are present in both input lists.
    """
    # Asserting the inputs
    assert isinstance(
        list1, list
    ), f'Expected list1 to be a list, but got {type(list1)}'
    assert isinstance(
        list2, list
    ), f'Expected list2 to be a list, but got {type(list2)}'

    return list(set(list1) & set(list2))


def is_dicts_keys_overlap(dict1, dict2):
    """
    Check if two dicts have any overlapping keys.
    """
    assert isinstance(dict1, dict)
    assert isinstance(dict2, dict)
    return is_lists_overlap(list(dict1.keys()), list(dict2.keys()))


def is_lists_overlap(list1, list2):
    """
    Check if two lists have any overlapping elements.
    :param list1: list : first list
    :param list2: list : second list
    :return: bool : True if the lists have overlapping elements, False otherwise.
    """

    # assert that the inputs are lists
    assert isinstance(list1, list)
    assert isinstance(list2, list)

    # check if there is any intersection between set1 and set2
    set1 = set(list1)
    set2 = set(list2)

    if set1.intersection(set2):
        return True
    else:
        return False


def reflect_scale(scale):

    scheduler_kwargs = {}
    data_loader_kwargs = {}

    scale = scale.split('/')
    data_loader_kwargs['dataset_name'] = str(scale[0])
    data_loader_kwargs['data_image_size'] = int(scale[1])
    data_loader_kwargs['data_is_gray'] = eval(scale[2])
    data_loader_kwargs['data_is_flatten'] = eval(scale[3])
    data_loader_kwargs['target_min'] = float(scale[4])
    data_loader_kwargs['target_max'] = float(scale[5])
    data_loader_kwargs['partial_targets_num'] = int(scale[6])
    data_loader_kwargs['partial_num'] = int(scale[7])
    data_loader_kwargs['batch_size'] = int(scale[8])

    scheduler_kwargs['max_t'] = int(scale[9])
    scheduler_kwargs['grace_period'] = int(scale[10])
    scheduler_kwargs['metric'] = str(scale[11])

    return data_loader_kwargs, scheduler_kwargs


def get_file_extensions_in_dir(dir_path, ignore_empty_extension=False):
    """
    Returns a list of unique file extensions in the given directory and its subdirectories.

    Parameters:
    - dir_path (str): The path to the directory to search.
    - ignore_empty_extension (bool): Whether to ignore files without an extension.

    Returns:
    - list: A list of file extensions as strings.
    """

    assert os.path.isdir(dir_path), (
        f"The given path {dir_path} is not a directory."
    )
    assert isinstance(ignore_empty_extension, bool), (
        f"The given ignore_empty_extension {ignore_empty_extension} is not a boolean."
    )

    extensions = []

    # Walk through all files and directories in the directory
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            # Get the file extension
            extension = os.path.splitext(file)[1]
            if ignore_empty_extension and extension == "":
                continue
            # Add the extension to the list if it's not already there
            if extension not in extensions:
                extensions.append(extension)

    return extensions


def get_file_extension(file):
    """Returns the file extension of a file.
    """
    _, file_extension = os.path.splitext(file)
    return file_extension


def print_df_as_dict(df):
    print(json.dumps(df.to_dict(orient='list'), sort_keys=True, indent=4))


def hashify(x):
    """Make x hashable if it is not hashable.
    """
    if not isinstance(x, collections.Hashable):
        x = str(x)
    return x


def assert_frame_equal(df1, df2, ignore_order=True, **kwargs):
    """Assert that two dataframes are equal.

    This function is a wrapper around pandas.util.testing.assert_frame_equal() that
    allows for ignoring the order of the rows and cols when comparing the dataframes.

    Parameters
    ----------
    df1 : pandas.DataFrame
        The first dataframe.
    df2 : pandas.DataFrame
        The second dataframe.
    ignore_order : bool, optional
        Whether to ignore the order of the rows and cols when comparing the dataframes.
    **kwargs
        Keyword arguments to pass to pandas.util.testing.assert_frame_equal().
    """

    assert isinstance(df1, pd.DataFrame)
    assert isinstance(df2, pd.DataFrame)
    assert isinstance(ignore_order, bool)

    df1 = df1.copy()
    df2 = df2.copy()

    def prep_df(df):
        df = df.applymap(hashify)
        df = df.reindex(
            sorted(df.columns),
            axis=1,
        )
        df = df.sort_values(
            list(df.columns),
            axis='index',
        )
        df = df.reset_index(drop=True)
        return df

    try:
        pd.util.testing.assert_frame_equal(
            prep_df(
                df1
            ) if ignore_order else df1.reset_index(drop=True),
            prep_df(
                df2
            ) if ignore_order else df2.reset_index(drop=True),
            check_dtype=False,
            **kwargs
        )
    except AssertionError as e:
        raise AssertionError(str(e))


def concatenate_dicts(dict_list):
    """
    Concatenates a list of dictionaries into a single dictionary.

    Parameters:
    dict_list (list of dicts): The list of dictionaries to concatenate.

    Returns:
    dict: The concatenated dictionary.
    """
    # Assert that the input is a list of dictionaries
    assert all(
        isinstance(d, dict) for d in dict_list
    ), "Input must be a list of dictionaries"

    # Create a new empty dictionary to store the concatenation
    concatenated_dict = {}

    # Iterate through the list of dictionaries
    for dictionary in dict_list:
        # Update the concatenated dictionary with the key-value pairs from the current dictionary
        concatenated_dict.update(dictionary)

    # Return the concatenated dictionary
    return concatenated_dict


def fig_to_pil(fig):
    """
    Convert a Matplotlib figure to a PIL image.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to convert.

    Returns
    -------
    PIL.Image
        A PIL image representation of the figure.
    """

    # Assert that the input is a Matplotlib figure
    assert isinstance(
        fig, matplotlib.figure.Figure), "Input must be a Matplotlib figure"

    # Draw the figure to a canvas
    canvas = FigureCanvasAgg(fig)

    # Get the canvas's pixel buffer and convert it to a PIL image
    buf, size = canvas.print_to_buffer()
    return Image.frombytes('RGB', size, buf)


def fig_to_numpy(fig):
    """Convert a Matplotlib figure to a NumPy array.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to be converted.

    Returns
    -------
    buf : numpy.ndarray
        The NumPy array representation of the figure.
    """

    # Assert that the input is a Matplotlib figure
    assert isinstance(
        fig, matplotlib.figure.Figure), "Input must be a Matplotlib figure"

    # draw the renderer
    fig.canvas.draw()

    # Get the RGB buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (h, w, 3)

    return buf.transpose((2, 0, 1))


def remove_items(list_, list_remove, in_place=False):
    """
    Remove items in list_remove from list_.

    Parameters:
    list_ (list): The list from which to remove items.
    list_remove (list): The list of items to remove.

    Returns:
    list: The modified list with items removed.
    """

    assert isinstance(list_, list), "list_ must be a list"
    assert isinstance(list_remove, list), "list_remove must be a list"

    if not in_place:
        list_ = copy.deepcopy(list_)  # make a deep copy of the list
    else:
        list_ = list_

    for item in list_remove:
        if item in list_:
            list_.remove(item)
        else:
            logger.warning(
                "Item {} not in list_ and cannot be removed.".format(item)
            )

    return list_


def clean_files_with_extension(path, extensions):
    """
    It removes all files with a given extension from a given directory

    :param path: The path to the directory you want to clean
    :param extensions: a list of file extensions to delete
    """
    os.chdir(path)
    for extension in extensions:
        for file_ in glob.glob(f"*.{extension}"):
            os.remove(file_)


try:
    WINDOW_NUM_ROWS, WINDOW_NUM_COLUMNS = os.popen(
        'stty size', 'r'
    ).read().split()
except Exception as e:
    WINDOW_NUM_ROWS, WINDOW_NUM_COLUMNS = ['23', '101']

WINDOW_NUM_ROWS = int(WINDOW_NUM_ROWS)
WINDOW_NUM_COLUMNS = int(WINDOW_NUM_COLUMNS)


# def get_working_home():
#     working_home = os.environ.get('WORKING_HOME')
#     if working_home is None:
#         working_home = "/Users/yuhangsong/"
#         print("Please specify your working home by setting the WORKING_HOME environment variable, defaulting to {}".format(working_home))
#     return working_home


def countdown(t, msg=""):

    while t:
        mins, secs = divmod(t, 60)
        timer = "{}{:02d}:{:02d}".format(msg, mins, secs)
        print(timer, end="\r")
        time.sleep(1)
        t -= 1

    print("\n")


def torch_max_onehot(x, dim=1):
    """Convert a torch.Tensor to a onehot version by max (the maximum value gets the one)."""
    onehot = (
        x - x.max(dim=dim, keepdim=True)[0]
    ).sign() + 1.0
    return onehot


def get_classification_error(prediction, target, is_binary=False, binary_threshold=0.5):
    """Get the classification error from prediction and target.
    """
    if is_binary:
        assert target.size(1) == 1
        target_binary = (
            (target-binary_threshold).sign() + 1.0
        )/2.0
        assert prediction.size(1) == 1
        prediction_binary = (
            (prediction-binary_threshold).sign() + 1.0
        )/2.0
        classification_error_total = (
            prediction_binary - target_binary
        ).abs().sum()
    else:
        assert target.size(1) > 1
        target_onehot = torch_max_onehot(target, dim=1)
        assert prediction.size(1) > 1
        prediction_onehot = torch_max_onehot(prediction, dim=1)
        classification_error_total = (
            prediction_onehot - target_onehot
        ).abs().sum(
            dim=1, keepdim=False
        ).sign().sum()

    classification_error = classification_error_total / target.size(0)
    return classification_error


def tensordataset_data_loader(input_, target_, noise_std=0.0, **data_loader_kwargs):
    """Short-cutting fn that creates DataLoader for a TensorDataset."""

    return DataLoader(
        TensorDataset(
            torch_normal(
                torch.FloatTensor(
                    input_,
                ),
                noise_std,
            ),
            torch_normal(
                torch.FloatTensor(
                    target_,
                ),
                noise_std,
            ),
        ),
        **data_loader_kwargs,
    )


transforms_flatten = transforms.Lambda(
    lambda x: x.view(-1)
)


def torch_normal(mean, std):
    """An modification to torch.normal that suppports std of zero."""
    assert isinstance(std, (float, int))
    if std == 0.0:
        return mean
    else:
        return torch.normal(mean, std)


def np_idx2onehot(idx, size, unlabelled_fill_fn=lambda size: (1.0 / size), target_min=0.0, target_max=1.0):
    """Apply on numpy/int idx, convert idx to onehot.
        The idx of -1 is considered to be unlabelled data, thus filled with unlabelled_fill_fn.
    """
    if idx >= 0:
        onehot = torch.nn.functional.one_hot(
            torch.LongTensor([idx]), size).squeeze(0)
        onehot = (1-onehot)*target_min + onehot*target_max
    elif idx == -1:
        onehot = torch.Tensor(size)
        onehot.fill_(unlabelled_fill_fn(size))
    else:
        raise NotImplementedError
    return onehot


# def param_delta_fn(key, param_model_i, param_model_j):
#     """Function that computed delta between two params.
#     """
#     return (
#         param_model_i - param_model_j
#     ).norm(2).item()


def grad(f):
    """Return a function that is the grad of the function f.
    """
    def result(x):
        # make leaf variables out of the inputs
        inputs = x.detach().requires_grad_(True)
        outputs = f(inputs)
        grad_ = torch.autograd.grad(
            outputs, inputs, grad_outputs=torch.ones_like(outputs)
        )[0]
        return grad_
    return result


def sigmoid_inverse(x, eps=0.03):
    """Inverse of torch.sigmoid.
    """
    x = x.clamp(0.0 + eps, 1.0 - eps)
    return torch.log(x / (1 - x))


def hardtanh_inverse(x, min_val=-1.0, max_val=1.0, eps=0.03):
    """Inverse of F.hardtanh.
    """
    x = x.clamp(min_val + eps, max_val - eps)
    return x


def tanh_inverse(x, eps=0.03):
    """Inverse of F.tanh.
    """
    x = x.clamp(-1.0 + eps, 1.0 - eps)
    return torch.atanh(x)


def identity(x):
    """Identity
    """
    return x


def identity_inverse(x):
    """Inverse of identity.
    """
    return x


# def get_machines():
#     """
#     It reads the file `machines.csv` and returns a dataframe
#     :return: A dataframe of the machines.csv file
#     """
#     return pd.read_csv(
#         os.path.join(
#             os.path.dirname(__file__),
#             'machines.csv',
#         )
#     )


# def get_machine_id():
#     """
#     It gets the machine id by matching the ip address in the output of `ifconfig` with the ip addresses
#     in the `machines` table
#     :return: The machine_id of the machine that is running the code.
#     """

#     # get the output by running ifconfig
#     ifconfig = subprocess.Popen(
#         "ifconfig",
#         shell=True,
#         stdout=subprocess.PIPE,
#     ).stdout.read().decode("utf-8")

#     # get matched_machine_ids
#     matched_machine_ids = []
#     for _, row in get_machines().iterrows():
#         if row['ip'] != 'none':
#             if row['ip'] in ifconfig:
#                 matched_machine_ids.append(row['id'])

#     # return
#     if len(matched_machine_ids) > 1:
#         raise Exception("There are multiple machine_ip matches the ifconfig.")
#     elif len(matched_machine_ids) < 1:
#         return "Unknown"
#     else:
#         return matched_machine_ids[0]


def report_via_email(subject='None', text='None', from_='mailgun', to_='yuhang.song@some.ox.ac.uk'):

    subject = f'{subject}'

    print('# INFO: reporting via email \nSubject: {}\nText: {}\n'.format(
        subject,
        text,
    ))

    if from_ in ['mailgun']:

        mailgun_email(
            from_={
                'mailgun': "Mailgun Sandbox <postmaster@sandboxfdb501a0d17c4a86a5334e3f7cbdc2ae.mailgun.org>",
            }[from_],
            to_=to_,
            subject=subject,
            text=text,
        )

    elif from_ in ['163', 'gmail']:

        smtp_email(
            from_={
                '163': {
                    'address': 'yuhangsong_machine@163.com',
                    'password': 'RJUNSTDKWDMYWRLB',
                    'smtp_server': 'smtp.163.com',
                },
                'gmail': {
                    'address': 'yuhangsong.machine@gmail.com',
                    'password': 'Pinmoh-gizjeb-4dymsy',
                    'smtp_server': 'smtp.gmail.com',
                },
            }[from_],
            to_=to_,
            subject=subject,
            text=text,
        )

    else:

        raise NotImplementedError


def mailgun_email(
        from_,
        to_,
        subject,
        text,
):
    """Send an email via mailgun.
    """

    try:
        requests.post(
            "https://api.mailgun.net/v3/sandboxfdb501a0d17c4a86a5334e3f7cbdc2ae.mailgun.org/messages",
            auth=("api", "fdab819d3b4dd8afad74aba2d90acd95-1b8ced53-13c59969"),
            data={
                "from": from_,
                "to": f"Yuhang Song <{to_}>",
                "subject": subject,
                "text": text
            },
        )
        print('# INFO: email sent')
    except Exception as e:
        print('# INFO: sending email unsuccessful: \n{}'.format(e))


def smtp_email(
        from_,
        to_='yuhangsong2017@gmail.com',
        subject='Experiment Report',
        text='',
):
    """Send an email via smtp.
    """

    '''build message'''
    msg = MIMEText(text)
    msg['Subject'] = subject
    msg['From'] = from_['address']
    msg['To'] = to_

    print('# INFO: sending email \n{}\n'.format(
        msg,
    ))

    '''builder server and send'''
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(from_['smtp_server'], 465, context=context) as server:
            server.login(from_['address'], from_['password'])
            server.sendmail(from_['address'], [to_], msg.as_string())
            server.quit()
        print('# INFO: email sent')
    except Exception as e:
        print('# INFO: sending email unsuccessful: \n{}'.format(e))


def print_msg(msg, label='WARNING'):

    print()
    print('# {}: {}'.format(
        label, msg,
    ))
    print()


class SCO(nn.Module):
    """Shift, Scale and Offset a torch.nn.Module.
    """

    def __init__(self, module, s=0.0, c=1.0, o=0.0):
        """
        :param module: A torch.nn.Module.
        :param s: shift
        :param c: scale
        :param o: offset
        """
        super(SCO, self).__init__()
        assert isinstance(module, nn.Module)
        self.module = module
        assert isinstance(s, float)
        self.shift = s
        assert isinstance(o, float)
        self.offset = o
        assert isinstance(c, float)
        self.scale = c

    def forward(self, x):
        return self.module(
            x + self.shift
        ) * self.scale + self.offset


# class ControlledSigmoid(nn.Sigmoid):
#     """Controlled Sigmoid to have a continuous shift to tanh.
#     """

#     def __init__(self, l=1.0, o=0.0, c=1.0):
#         """
#         The function __init__() is a constructor that initializes the class ControlledSigmoid with the parameters
#         l, o, and c

#         :param l: slope
#         :param o: offset
#         :param c: scale
#         """
#         super(ControlledSigmoid, self).__init__()
#         assert isinstance(l, float)
#         self.slope = l
#         assert isinstance(o, float)
#         self.offset = o
#         assert isinstance(c, float)
#         self.scale = c

#     def forward(self, x):
#         return (
#             super(ControlledSigmoid, self).forward(
#                 x * self.slope
#             ) + self.offset
#         ) * self.scale


# class ControlledReLU(nn.ReLU):
#     """Controlled ReLU.
#     """

#     def __init__(self, s=0.0, o=0.0, c=1.0):
#         """
#         The function __init__() is a constructor that initializes the class ControlledReLU with the parameters
#         s, and o

#         :param s: shift
#         :param o: offset
#         :param c: scale
#         """
#         super(ControlledReLU, self).__init__()
#         assert isinstance(s, float)
#         self.shift = s
#         assert isinstance(o, float)
#         self.offset = o
#         assert isinstance(c, float)
#         self.scale = c

#     def forward(self, x):
#         return (
#             super(ControlledReLU, self).forward(
#                 x + self.shift
#             ) + self.offset
#         ) * self.scale


# class ControlledHardtanh(nn.Hardtanh):
#     """Controlled Hardtanh.
#     """

#     def __init__(self, s=0.0, o=0.0, c=1.0):
#         """
#         The function __init__() is a constructor that initializes the class ControlledHardtanh with the parameters
#         s, and o

#         :param s: shift
#         :param o: offset
#         """
#         super(ControlledHardtanh, self).__init__()
#         assert isinstance(s, float)
#         self.shift = s
#         assert isinstance(o, float)
#         self.offset = o
#         assert isinstance(c, float)
#         self.scale = c

#     def forward(self, x):
#         return (
#             super(ControlledHardtanh, self).forward(
#                 x + self.shift
#             ) + self.offset
#         ) * self.scale


# class ISoftplus(nn.Module):
#     r"""Applies the ISoftplus function (inverse of Softplus):math:`\text{ISoftplus}(x) = \frac{1}{\beta} *
#     \log(\exp(\beta * x) - 1)` element-wise.

#     For numerical stability the implementation reverts to the linear function
#     when :math:`input \times \beta > threshold`.

#     Args:
#         beta: the :math:`\beta` value for the ISoftplus/Softplus formulation. Default: 1
#         threshold: values above this revert to a linear function. Default: 20

#     Shape:
#         - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
#         - Output: :math:`(*)`, same shape as the input.

#     .. image:: ../scripts/activation_images/ISoftplus.png

#     Examples::

#         >>> m = nn.ISoftplus()
#         >>> input = torch.randn(2)
#         >>> output = m(input)
#     """
#     __constants__ = ['beta', 'threshold']
#     beta: int
#     threshold: int

#     def __init__(self, beta: int = 1, threshold: int = 20) -> None:
#         super(ISoftplus, self).__init__()
#         self.beta = beta
#         self.threshold = threshold
#         self.zero_point = math.log(2)/beta

#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         mask = (input-self.threshold).sign().clamp(0, 1)
#         return (self.beta*((input-self.zero_point)*(1-mask)+self.zero_point)).exp().add(-1).log().div(self.beta) + mask*input

#     def extra_repr(self) -> str:
#         return 'beta={}, threshold={}'.format(self.beta, self.threshold)


# class ControlledSoftplus(nn.Softplus):
#     """Controlled Softplus.
#     """

#     def __init__(self, s=0.0, o=0.0, c=1.0, beta=1, threshold=20):
#         """
#         The function __init__() is a constructor that initializes the class ControlledSoftplus with the parameters
#         s, and o

#         :param s: shift
#         :param o: offset
#         :param c: scale
#         """
#         super(ControlledSoftplus, self).__init__(
#             beta=beta,
#             threshold=threshold,
#         )
#         assert isinstance(s, float)
#         self.shift = s
#         assert isinstance(o, float)
#         self.offset = o
#         assert isinstance(c, float)
#         self.scale = c

#     def forward(self, x):
#         return super(ControlledSoftplus, self).forward(
#             x + self.shift
#         ) * self.scale + self.offset


def flatten_dict(d, parent_key='', sep=': ', level_limit=np.inf, level_count=0):
    """
    Flattens a dictionary by merging the keys. Nested dictionaries are recursively
    flattened until level_limit is reached.

    Parameters
    ----------
    d: dict
        The dictionary to flatten
    parent_key: str
        The parent key of the current dictionary. Used as a prefix for the keys
        of the current dictionary when merging keys.
    sep: str
        The separator to use when merging keys
    level_limit: int
        The maximum number of times the function should recursively flatten nested
        dictionaries.
    level_count: int
        The current level of recursion. This should not be set by the user.

    Returns
    -------
    dict
        A flattened version of the input dictionary.
    """
    items = []
    level_count += 1
    for k, v in d.items():
        new_key = parent_key + sep + str(k) if parent_key else str(k)
        if isinstance(v, collections.MutableMapping) and (level_count < level_limit):
            items.extend(flatten_dict(v, new_key,
                                      sep=sep,
                                      level_limit=level_limit,
                                      level_count=level_count,
                                      ).items())
        else:
            items.append((new_key, v))
    return dict(items)


def assert_config_all_valid(config):
    """Ray save Trainable.config as experiment_state*.json, which means config cannot containing any memory intensive objects.
    On the other hand, we want to be able to identify and compare config, so the value of config should be of "regular" types.
    """
    for key, value in flatten_dict(config).items():
        assert isinstance(value, (str, int, float, bool)), (
            f"Invalid value for {key}: {value} of type {type(value)}"
        )


def prepare_path(path):
    """Check if path exists, if not, create one.
    """
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(
                os.path.dirname(
                    path
                )
            )
        except OSError as exc:
            # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def map_list_by_dict(l, d):
    """
    Maps the values of a list using a dictionary. If a value in the list is a key
    in the dictionary, it is replaced by the corresponding value in the dictionary.

    Parameters
    ----------
    l: list
        The list to map
    d: dict
        The dictionary to use for mapping

    Returns
    -------
    list
        The mapped list
    """
    for k, v in d.items():
        if k in l:
            l[l.index(k)] = v
    return l


def inquire_confirm(msg, default=True):
    # return prompt(
    #     [
    #         {
    #             'type': 'confirm',
    #             'message': msg + ' Confirm?',
    #             'name': 'confirm',
    #             'default': True,
    #         },
    #     ],
    #     style=custom_style_2,
    # )['confirm']
    answer = input(
        f"[Confirm]: {msg} {'[Y/n]' if default else '[y/N]'}"
    )
    if len(answer) == 0:
        answer = default
    else:
        if answer.lower() in ['y', 'yes']:
            answer = True
        elif answer.lower() in ['n', 'no']:
            answer = False
        else:
            raise ValueError(
                f"Invalid answer: {answer}"
            )
    return answer


def inquire_input(msg, default=None):
    answer = input(
        f"[Input]: {msg} {'[Default: <' + default + '>]' + ' ' if default is not None else ''}"
    )
    if len(answer) == 0:
        answer = default
    return answer


def prepare_dir(dir):
    """Check if dir exists, if not, create one.
    """
    if dir[-1] != '/':
        dir += '/'
    if not os.path.exists(dir):
        try:
            os.makedirs(
                dir
            )
        except OSError as exc:
            # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def list_intersection(lst1, lst2):
    """Get the intersection of two lists.
    """
    assert isinstance(lst1, list)
    assert isinstance(lst2, list)
    return list(set(lst1) & set(lst2))


"""The following is depreciated code"""


def tensordataset_data_loader_fn(input_, target_, noise_std, batch_size):
    """Short-cutting fn that creates DataLoader for a TensorDataset."""

    logger.warning(
        "tensordataset_data_loader_fn is deprecated, use tensordataset_data_loader instead.")

    return DataLoader(
        TensorDataset(
            torch_normal(
                torch.FloatTensor(
                    input_,
                ),
                noise_std,
            ),
            torch_normal(
                torch.FloatTensor(
                    target_,
                ),
                noise_std,
            ),
        ),
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True,
        shuffle=False,
    )


def tensordataset_data_loader_fn_with_f_inverse(tr, input_, target_, noise_std, batch_size):
    """Short-cutting fn that creates DataLoader for a TensorDataset. Note that it is with an f_inverse."""

    logger.warning(
        "tensordataset_data_loader_fn_with_f_inverse is deprecated, use tensordataset_data_loader instead.")

    return DataLoader(
        TensorDataset(
            tr.f_inverse(
                torch_normal(
                    torch.FloatTensor(
                        input_,
                    ),
                    noise_std,
                ),
            ),
            torch_normal(
                torch.FloatTensor(
                    target_,
                ),
                noise_std,
            ),
        ),
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True,
        shuffle=False,
    )

# onehot_zero = 0.03
# onehot_one = 0.97
# onehot_half = (onehot_zero + onehot_one) / 2.0


# class PartialSigmoid(torch.nn.Module):
#     def __init__(
#         self,
#         min_index: int = 0,
#         max_index: int = -1,
#     ):
#         super(PartialSigmoid, self).__init__()
#         self.min_index = min_index
#         self.max_index = max_index

#     def forward(self, inputs: torch.Tensor) -> torch.Tensor:
#         inputs_sigmoid = torch.sigmoid(
#             inputs[:, self.min_index:self.max_index]
#         )
#         return torch.cat(
#             [
#                 inputs[:, :self.min_index],
#                 inputs_sigmoid,
#                 inputs[:, self.max_index:],
#             ],
#             dim=1,
#         )


# def is_perfect_square(number):
#     root = math.sqrt(number)
#     if int(root + 0.5) ** 2 == number:
#         return True
#     else:
#         return False


# def copy_dict_by_keys(dict_, keys):
#     return dict(
#         (key, dict_[key]) for key in keys
#     )


# def energy_reduction(x, has_batch_dim, energy_reduction_mode):
#     if has_batch_dim:
#         if 'batch' in energy_reduction_mode:
#             x = x.mean(
#                 dim=0, keepdim=True
#             )
#         if 'neuro' in energy_reduction_mode:
#             x = x.view(
#                 x.size()[0], -1
#             ).mean(
#                 dim=1, keepdim=True
#             )
#     else:
#         if 'neuro' in energy_reduction_mode:
#             x = x.view(
#                 1, -1
#             ).mean(
#                 dim=1, keepdim=True
#             )
#     x = x.sum()
#     return x


# def torch_onehot2idx(onehot):
#     assert onehot.dim() == 2, "Shape needs to be [batch, onehot]"
#     idx = onehot.nonzero()[:, 1:2]
#     return idx


# def torch_clamp_onehot(x):
#     return x.clamp(onehot_zero, onehot_one)


# def torch_unclamp_onehot(x):
#     return (x - onehot_half).sign().clamp(0.0, 1.0)


# def np_clamp_onehot(x):
#     return np.clip(x, onehot_zero, onehot_one)


# def get_classification_accuracy(output, target, topk=1):
#     """Computes the precision@k for the specified values of k"""

#     _, pred = output.topk(topk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))

#     correct_k = correct[:topk].view(-1).float().sum(0)

#     return correct_k


# def add_to_result_dict(key, value, result_dict):
#     if key not in result_dict.keys():
#         result_dict[key] = float(value)
#     else:
#         result_dict[key] += float(value)
#     return result_dict


# def put_to_result_dict(key, value, result_dict):
#     if key in result_dict.keys():
#         print('# WARNING: put this key in result_dict for multiple times, overriding..')
#     result_dict[key] = float(value)
#     return result_dict


# def plot_output(output, data):
#     tmp = [output[0].view(28, 28), output[1].view(28, 28)]
#     if data[0, 1] > 0.0:
#         tmp = list(reversed(tmp))
#     draw_heatmap(
#         'output',
#         torch.cat(
#             tmp,
#             dim=1,
#         ).cpu().numpy(),
#     )
#     plt.draw()
#     plt.pause(0.001)


# def torch_astype_float16(x):
#     return x.half()


# def torch_astype_float32(x):
#     return x.float()


# def torch_astype_float64(x):
#     return x.double()


# def count_dict(d, reduce='max'):
#     """Count the number of levels of a dict.
#     """
#     if reduce == 'max':
#         return max(count_dict(v, reduce) if isinstance(v, dict) else 0 for v in d.values()) + 1
#     elif reduce == 'min':
#         return min(count_dict(v, reduce) if isinstance(v, dict) else 0 for v in d.values()) + 1
#     else:
#         raise NotImplementedError


# def flatten(x):
#     return x.view(-1)


# def try_check_output(cmd):
#     try:
#         subprocess.check_output(
#             cmd, shell=True
#         )
#     except subprocess.CalledProcessError as e:
#         print()
#         print_msg(f'error when {cmd}:')
#         print(e.output)
