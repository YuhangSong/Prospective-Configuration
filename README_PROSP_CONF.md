- [Inferring Neural Activity Before Plasticity: A Foundation for Learning Beyond Backpropagation](#inferring-neural-activity-before-plasticity-a-foundation-for-learning-beyond-backpropagation)
  - [Setting up environment](#setting-up-environment)
    - [Structure of the code](#structure-of-the-code)

# Inferring Neural Activity Before Plasticity: A Foundation for Learning Beyond Backpropagation

## Setting up environment

A `Dockerfile` and a pre-built docker image from this `Dockerfile` is available on docker hub [here](yuhangsongchina/general-energy-nets:1.0).

For those who are not familiar with docker, it provides a image of a light-weighted virtual machine, which is widedly used as a frozen version of a environment ready for running the code with all dependencies installed.
Our code is run and tested on major Linux distributions and Darwin/Mac systems, but not on Windows, in case users are on platforms that we haven't tested our code, such as Windows, and having difficulty setting up environment, one can use the docker image which will start a virtual machine with frozen environment ready to run our code.

For those who don't want to use docker, they can refer to comments in `Dockerfile` to set up the environment.
As can be seen in the `Dockerfile`, all packages are pretty standard and can be installed with `pip` or `conda`.

If you are not familar with docker but want to learn and use docker, [get started with docker](https://docs.docker.com/get-started/).

If you prefer to use conda, [get started with conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html#managing-python).

We recommend using docker.

### Structure of the code

The code is organized as follows:
