# Inferring Neural Activity Before Plasticity: A Foundation for Learning Beyond Backpropagation

This repository contains the code for the paper [Inferring Neural Activity Before Plasticity: A Foundation for Learning Beyond Backpropagation](https://www.biorxiv.org/content/10.1101/2022.05.17.492325v1).
By _Yuhang Song*_, _Beren Millidge_, _Tommaso Salvatori_, _Thomas Lukasiewicz_, _Zhenghua Xu_, _Rafal Bogacz*_.
**doi**: https://doi.org/10.1101/2022.05.17.492325.
Under 2nd round peer-review at _Nature Neuroscience_.

![](./interfere.png)

## Table of Contents

- [Inferring Neural Activity Before Plasticity: A Foundation for Learning Beyond Backpropagation](#inferring-neural-activity-before-plasticity-a-foundation-for-learning-beyond-backpropagation)
  - [Table of Contents](#table-of-contents)
  - [Setting up environment](#setting-up-environment)
    - [With docker](#with-docker)
    - [Without docker](#without-docker)
      - [Install dependencies](#install-dependencies)
      - [Set up environment variables](#set-up-environment-variables)
      - [Download the datasets](#download-the-datasets)
  - [Structure of the code](#structure-of-the-code)
  - [Run the code to reproduce experiments and figures](#run-the-code-to-reproduce-experiments-and-figures)
  - [Other notes](#other-notes)
    - [Warning and error messages](#warning-and-error-messages)
    - [Reproducibility](#reproducibility)
  - [Citation](#citation)
  - [Contact](#contact)

## Setting up environment

### With docker

A [`Dockerfile`](./Dockerfile) is available in the main directory above, and a pre-built docker image from this `Dockerfile` is available on docker hub here: [dryuhangsong/general-energy-nets:1.0](https://hub.docker.com/r/dryuhangsong/general-energy-nets).

For those who are not familiar with docker, it provides a image of a light-weighted virtual machine, which is widedly used as a frozen version of a environment ready for running the code with all dependencies installed.
Our code is run and tested on major Linux distributions and Darwin/Mac systems, but not on Windows, in case users are on platforms that we haven't tested our code, such as Windows, and having difficulty setting up environment, one can use the docker image which will start a virtual machine with frozen environment ready to run our code.
If you are not familar with docker but want to learn and use docker, [get started with docker](https://docs.docker.com/get-started/).

Alternatively, you can set up the environment by yourself (without docker), see [Without docker](#without-docker).

### Without docker

This section setup the environment without docker.

Note that exact reproducibility is not guaranteed without docker.
But you should not get obvious divergent results from what is reported in the paper, becase the code run on pretty high-level APIs of the packages, and the high-level APIs are supposed to be stable across different versions of the packages.
If you do obverve divergent results, please try to install package with version matching the docker image, and also **let us know** so that we can make a note for other users.

Also in this guide, we don't fix the package versions or python versions, because different python versions and packages versions might be supported and not supported on different systems.

You can use [conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html#managing-python) or other tools to manage the python environment.
You then need to walk through the following a few steps to get your environment ready.

#### Install dependencies

You need to install the following packages:

```bash
pip install ray[all] torch torchvision torchaudio seaborn tqdm visdom tabulate
```

#### Set up environment variables

You need to set up the following environment variables:

- `DATA_DIR`: The directory where you want to store the datasets.
- `RESULTS_DIR`: The directory where you want to store the results.
- `CODE_DIR`: The directory where you clone and store the code.

Of course, make sure that the directories exist.
Exactly how to set up environment variables depends on your system (a quick search on internet should give you the answer).

#### Download the datasets

```bash
python -c "from torchvision import datasets; import os; [eval(f'datasets.{dataset}')(os.environ.get('DATA_DIR'),download=True) for dataset in ['MNIST']]"
python -c "from torchvision import datasets; import os; [eval(f'datasets.{dataset}')(os.environ.get('DATA_DIR'),download=True) for dataset in ['FashionMNIST']]"
python -c "from torchvision import datasets; import os; [eval(f'datasets.{dataset}')(os.environ.get('DATA_DIR'),download=True) for dataset in ['CIFAR10']]"
```

## Structure of the code

The code is organized as follows:

- files in `./` are the main files for running the code. Following gives a brief description of each file.
  - `main.py`: Calling this file with configuration yaml file will will launch experiments (the backbone of the code is ray, which will launch multiple processes to run the code in parallel)
  - `utils.py`: Utility functions
  - `analysis_v1.py`: Functions for analyzing the results. Calling this file with configuration file will load the results and plot the figures.
  - `analysis_utils.py`: Utility functions for analysis.
  - `*_trainable.py`: Various trainable classes that is shared across different experiments.
  - `data_utils.py`: Utility functions for dataset.
  - `fit_data.py`: Functions for fitting data from biological experiments.
- `experiments`: This folder contains all the experiments. Specifically, each subfolder contains
  - the `README.md` file that describes the experiment, document the comments to run to reproduce the experiment and reproduce the figure in the paper.
  - the configuration file(s) (`.yaml`) for the experiment.
  - `.py` files that are specific to the experiment.
  - `.png` and `.pdf` files that are the figures generated by the experiment.
- `tests`: This folder contains unit test for all layers of the code. They are coded in pytest style.

## Run the code to reproduce experiments and figures

The simply look into each subfolders in `experiments` and follow the instructions in the `README.md` file, where the resulted figures are also documented.

By reading each `README.md` file in a markdown editor (such as viewing it on github), the resulted figures are attached inline, so it is obvious which corresponding figure it produces in the paper.

Before looking at each experiment folder, we explain the shared underlying logic of the code.
In each experiment folder, the `README.md` documents two kinds of commands to run:

- `python main.py -c <config_file.yaml>`: This command will launch the experiment. The experiment will be launched in parallel with multiple processes (with ray as backend), and the results will be saved to `$RESULTS_DIR` in your environment variable.
  - You will see command `ray job submit --runtime-env runtime_envs/runtime_env_without_ip.yaml --address $pssr -- ` before `python main.py -c <config_file.yaml>`, it is to submit the job to ray cluster to run instead of run locally. 
    - If you want to run locally, you can simply remove this command, and run `python main.py -c <config_file.yaml>` on your local machine.
    - If you want to run on a ray cluster, you will need to get yourself educated about [ray cluster](https://docs.ray.io/en/latest/cluster/getting-started.html). Then you need to set up a ray cluster and set the environment variable `$pssr` to the address of the ray cluster.
- `python analysis_v1.py -c <config_file.yaml>`: This command will load the results from `$RESULTS_DIR` and plot the figures. The figures will be saved to the experiment folder.
  - This command does not limit to produce figures though, it loads the results as a pandas dataframe and do anything with it, depending on the exec commands you passed to it. For example, it is sometimes used to produced tables as well.

## Other notes

There are several other things to note.

### Warning and error messages

You may see some warning messages when running the code:

- depreciation warning messages from the code base are very safe, it is there so that to remind me what functionalities/logic has been depreciated, but the code is fully backward compatible.
- depreciation warning messages from dependencies are normally safe, as the dependencies of the code base are well maintained libaries like PyTorch, Numpy, Seaborn and etc, depreciation warning messages is almost guaranteed to have backward compatibility.

You should not see any error messages if you are using the docker image.
You may see error messages if you are using your own environment, but they are usually easy to fix by compring the `Dockerfile` with your procedure of setting up the environment.
Open an issue if you have any problem in dealing with error messages and we will help out as we can.

### Reproducibility

Reproducibility should be guaranteed by the docker image and how we control the randomness of Pytorch, Numpy and Random packages.
Please open an issue if you find any reproducibility issue.

## Citation

We kindly ask you to cite our paper if you use our code in your research.

```bib
@article {Song2022.05.17.492325,
    author = {Song, Yuhang and Millidge, Beren and Salvatori, Tommaso and Lukasiewicz, Thomas and Xu, Zhenghua and Bogacz, Rafal},
    title = {Inferring Neural Activity Before Plasticity: A Foundation for Learning Beyond Backpropagation},
    elocation-id = {2022.05.17.492325},
    year = {2022},
    doi = {10.1101/2022.05.17.492325},
    publisher = {Cold Spring Harbor Laboratory},
    abstract = {For both humans and machines, the essence of learning is to pinpoint which components in its information processing pipeline are responsible for an error in its output {\textemdash} a challenge that is known as credit assignment1. How the brain solves credit assignment is a key question in neuroscience, and also of significant importance for artificial intelligence. Many recent studies1{\textendash}12 presuppose that it is solved by backpropagation13{\textendash}16, which is also the foundation of modern machine learning17{\textendash}22. However, it has been questioned whether it is possible for the brain to implement backpropagation23, 24, and learning in the brain may actually be more efficient and effective than backpropagation25. Here, we set out a fundamentally different principle on credit assignment, called prospective configuration. In prospective configuration, the network first infers the pattern of neural activity that should result from learning, and then the synaptic weights are modified to consolidate the change in neural activity. We demonstrate that this distinct mechanism, in contrast to backpropagation, (1) underlies learning in a well-established family of models of cortical circuits, (2) enables learning that is more efficient and effective in many contexts faced by biological organisms, and (3) reproduces surprising patterns of neural activity and behaviour observed in diverse human and animal learning experiments. Our findings establish a new foundation for learning beyond backpropagation, for both understanding biological learning and building artificial intelligence.Competing Interest StatementThe authors have declared no competing interest.},
    URL = {https://www.biorxiv.org/content/early/2022/05/18/2022.05.17.492325},
    eprint = {https://www.biorxiv.org/content/early/2022/05/18/2022.05.17.492325.full.pdf},
    journal = {bioRxiv}
}
```

## Contact

For questionts about the code, please open an issue in this repository.

For questions about the paper, please [contact Yuhang Song and Rafal Bogacz](mailto:yuhang.song@bndu.ox.ac.uk;rafal.bogacz@ndcn.ox.ac.uk).
