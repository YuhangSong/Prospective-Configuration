# build
# docker build -t dryuhangsong/prospective-configuration:latest .
# push
# docker push dryuhangsong/prospective-configuration:latest

FROM rayproject/ray-ml:2.4.0-py38-gpu

# python packages
RUN pip install seaborn==0.12.1 tqdm==4.63.0 visdom==0.2.3 plotly==5.11.0 tabulate==0.9.0 torch==1.12.1+cu116 torchvision==0.13.1+cu116 -f https://download.pytorch.org/whl/cu116/torch_stable.html

# RUN pip install --upgrade tensorflow-probability

ENV DATA_DIR=/home/ray/data/
RUN mkdir $DATA_DIR
RUN python -c "from torchvision import datasets; import os; [eval(f'datasets.{dataset}')(os.environ.get('DATA_DIR'),download=True) for dataset in ['FashionMNIST']]"
RUN python -c "from torchvision import datasets; import os; [eval(f'datasets.{dataset}')(os.environ.get('DATA_DIR'),download=True) for dataset in ['CIFAR10']]"