# # create clean conda environment
# # # NOTE: python=3.8.10 does not work for pyspice
# # # NOTE: conda specification in runtime_env.yaml is not working, it says "This may be because needed library dependencies are not installed in the worker environment"
# RUN conda activate base && conda remove -n ray --all -y && conda create -n ray python=3.7.13 -y

FROM rayproject/ray-ml:16d9fe-py37-gpu

# # init conda
# RUN conda init bash && source /root/.bashrc

# # activate conda environment in bashrc (if not already there)
# RUN if ! grep -q "conda activate ray" ~/.bashrc; then echo "conda activate ray" >> ~/.bashrc; fi

RUN rm -rf /home/ray/anaconda3/lib/python3.7/site-packages/numpy

# pyspice
RUN conda install -c conda-forge ngspice-exe -y && conda install -c conda-forge ngspice -y && conda install -c conda-forge ngspice-lib -y && conda install -c conda-forge pyspice -y

# torch
# # NOTE: higher version of pytorch does not compatible with memtorch
RUN conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 -c pytorch -c conda-forge
# # # for MacOS
# # # # RUN conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 -c pytorch

# memtorch
RUN pip install memtorch-cpu scikit-learn

# others
RUN pip install seaborn tqdm visdom plotly tabulate

# ray
# # NOTE: remove [all] will cause no dashboard, at least need [default]
# # NOTE: if you have the following error: No module named 'ray.air.integrations'
# # # adding `pip uninstall ray && pip install -U` will fix it
# # NOTE: upgrade ray version from 2.x to 3.x so that scheduler can be resumed
RUN pip uninstall ray -y && pip install -U "ray[all] @ https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp37-cp37m-manylinux2014_x86_64.whl" 
# # # for MacOS
# # # # RUN pip install "ray[all] @ https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp37-cp37m-macosx_10_15_intel.whl"

# RUN pip install wandb
# # does not work with h-params view, so move to comet_ml
RUN pip install comet_ml
# RUN pip install optuna

RUN pip install --upgrade tensorflow-probability

# # amazon-efs
# RUN sudo kill -9 `sudo lsof /var/lib/dpkg/lock-frontend | awk '{print $2}' | tail -n 1`; \
#     sudo pkill -9 apt-get; \
#     sudo pkill -9 dpkg; \
#     sudo dpkg --configure -a; \
#     sudo apt-get -y install binutils; \
#     cd $HOME; \
#     git clone https://github.com/aws/efs-utils; \
#     cd $HOME/efs-utils; \
#     ./build-deb.sh; \
#     sudo apt-get -y install ./build/amazon-efs-utils*deb; \
#     cd $HOME; \
#     mkdir efs;

# RUN sudo apt-get install vim -y

# download datasets
ENV DATA_DIR=/home/ray/data/
RUN mkdir $DATA_DIR
RUN python -c "from torchvision import datasets; import os; [eval(f'datasets.{dataset}')(os.environ.get('DATA_DIR'),download=True) for dataset in ['MNIST']]"
RUN python -c "from torchvision import datasets; import os; [eval(f'datasets.{dataset}')(os.environ.get('DATA_DIR'),download=True) for dataset in ['FashionMNIST']]"
RUN python -c "from torchvision import datasets; import os; [eval(f'datasets.{dataset}')(os.environ.get('DATA_DIR'),download=True) for dataset in ['CIFAR10']]"

ENV RESULTS_DIR=/home/ray/general-energy-nets-results/
RUN mkdir $RESULTS_DIR