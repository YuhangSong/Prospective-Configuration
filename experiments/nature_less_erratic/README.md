<!-- TOC -->

- [traj-learning-rate](#traj-learning-rate)
  - [two error bars (nature)](#two-error-bars-nature)
  - [traj-learning-rate-apgr](#traj-learning-rate-apgr)
    - [two error bars (nature)](#two-error-bars-nature-1)

<!-- /TOC -->

```bash

```

<!-- # random data all

random data, all measurements are studied

```bash
ray job submit --runtime-env runtime_envs/runtime_env_without_ip.yaml --address $PSSR -- python main.py -c nature_less_erratic/rand-data-all
```

```bash
running
python analysis_v1.py \
-t "rand-data-all" \
-l "../general-energy-nets-results/nature_less_erratic/" \
-m "df['measure_result'].iloc[-1]" \
-f "./experiments/nature_less_erratic/rand-data-all.yaml" \
-g 'measure' \
-v \
"df=nature_pre(df)" \
"g=nature_relplot(data=df,x='pc_learning_rate',y='measure_result',hue='Rule',row='iteration_end',col='num_datapoints',sharey=False).set(xscale='log',yscale='log')" \
"nature_post(g,is_grid=True)"
```

[doc](./rand-data-all.md)

# angle based

## real dataset

### base

with some search

```bash
ray job submit --runtime-env runtime_envs/runtime_env_without_ip.yaml --address $PSSR -- python main.py -c nature_less_erratic/base
```

```bash
python analysis_v1.py \
-t "base" \
-l "../general-energy-nets-results/nature_less_erratic/" \
-m "df['consistency'].iloc[-1]" \
-f "./experiments/nature_less_erratic/base.yaml" \
-v \
"df=nature_pre(df)" \
"g=nature_relplot(data=df,x='pc_learning_rate',y='consistency',hue='Rule',col='type').set(xscale='log')" \
"nature_post(g,is_grid=True)"
```

![](./base-.png)

### base-nature

focus on one config

weird, try random data, quit, not applied in nature

```bash
ray job submit --runtime-env runtime_envs/runtime_env_without_ip.yaml --address $PSSR -- python main.py -c nature_less_erratic/base-nature
```

```bash
python analysis_v1.py \
-t "base-nature" \
-l "../general-energy-nets-results/nature_less_erratic/" \
-m "df['consistency'].iloc[-1]" \
-f "./experiments/nature_less_erratic/base-nature.yaml" \
-v \
"df=nature_pre(df)" \
"g=nature_relplot(data=df,x='pc_learning_rate',y='consistency',hue='Rule', col='Inference rate')" \
"nature_post(g,is_grid=True)"
```

![](./base-nature-.png)

### base-apgr

same as the above but ap and gr

```bash
ray job submit --runtime-env runtime_envs/runtime_env_without_ip.yaml --address $PSSR -- python main.py -c nature_less_erratic/base-apgr
```

```bash
python analysis_v1.py \
-t "base-apgr" \
-l "../general-energy-nets-results/nature_less_erratic/" \
-m "df['consistency'].iloc[-1]" \
-f "./experiments/nature_less_erratic/base-apgr.yaml" \
-v \
"g=nature_relplot(data=df,x='pc_learning_rate',y='consistency',hue='Rule', col='Inference rate',sharey=False).set(xscale='log')" \
"nature_post(g,is_grid=True)"
```

![](./base-apgr-.png)

# length based

## random data

randomly generated a batch of data

### base-path-rand-data

```bash
ray job submit --runtime-env runtime_envs/runtime_env_without_ip.yaml --address $PSSR -- python main.py -c nature_less_erratic/base-path-rand-data -m T1
```

```bash
python analysis_v1.py \
-t "base-path-rand-data" \
-l "../general-energy-nets-results/nature_less_erratic/" \
-m "df['trace_rate'].iloc[-1]" \
-f "./experiments/nature_less_erratic/base-path-rand-data.yaml" \
-v \
"df=nature_pre(df)" \
"g=nature_relplot(data=df,x='pc_learning_rate',y='trace_rate',hue='Rule',row='iteration_end',col='num_datapoints').set(xscale='log')" \
"nature_post(g,is_grid=True)"
```

![](./base-path-rand-data-.png)

### length of final vector as the x axis

length of final vector as the x axis

```bash
python analysis_v1.py \
-t "base-path-rand-data-x-length" \
-l "../general-energy-nets-results/nature_less_erratic/" \
-m "df['traj_length'].iloc[-1]" "df['final_length'].iloc[-1]" \
-f "./experiments/nature_less_erratic/base-path-rand-data.yaml" \
-g "pc_learning_rate" \
-v \
"df=nature_pre(df)" \
"g=sns.displot(kind='kde',data=df,x='final_length',y='traj_length',hue='Rule',facet_kws={'sharey':False,'sharex':False})"
```

[md](./base-path-rand-data-x-length.md)

```bash
python analysis_v1.py \
-t "base-path-rand-data-x-length-focus" \
-l "../general-energy-nets-results/nature_less_erratic/" \
-m "df['traj_length'].iloc[-1]" "df['final_length'].iloc[-1]" \
-f "./experiments/nature_less_erratic/base-path-rand-data.yaml" \
-v \
"df=nature_pre(df)" \
"df=filter_dataframe_by_dict(df,{'pc_learning_rate':0.1})" \
"g=sns.jointplot(data=df,x='final_length',y='traj_length',hue='Rule',kind='kde')"
```

![](./base-path-rand-data-x-length-focus-.png)

```bash
ray job submit --runtime-env runtime_envs/runtime_env_without_ip.yaml --address $PSSR -- python main.py -c nature_less_erratic/base-path-rand-data -m T1
```

```bash
python analysis_v1.py \
-t "base-path-rand-data-x-length-focus-new" \
-l "../general-energy-nets-results/nature_less_erratic/" \
-m "df['traj_length'].iloc[-1]" "df['final_length'].iloc[-1]" \
-f "./experiments/nature_less_erratic/base-path-rand-data.yaml" \
-v \
"df=nature_pre(df)" \
"g=sns.relplot(data=df,x='final_length',y='traj_length',hue='Rule',size='pc_learning_rate',col='seed')"
```

![](./base-path-rand-data-x-length-focus-new-.png)

## base-path-rand-one-data

randomly generated one single data

```bash
ray job submit --runtime-env runtime_envs/runtime_env_without_ip.yaml --address $PSSR -- python main.py -c nature_less_erratic/base-path-rand-one-data
```

```bash
python analysis_v1.py \
-t "base-path-rand-one-data" \
-l "../general-energy-nets-results/nature_less_erratic/" \
-m "df['trace_rate'].iloc[-1]" \
-f "./experiments/nature_less_erratic/base-path-rand-one-data.yaml" \
-v \
"df=nature_pre(df)" \
"g=nature_relplot(data=df,x='pc_learning_rate',y='trace_rate',hue='Rule',row='iteration_end',col='num_datapoints',sharey=False).set(xscale='log',yscale='log')" \
"nature_post(g,is_grid=True)"
```

![](./base-path-rand-one-data-.png) -->

# traj-learning-rate

try to put traj length and learning rate into the same plot

```bash
ray job submit --runtime-env runtime_envs/runtime_env_without_ip.yaml --address $PSSR -- python main.py -c nature_less_erratic/traj-learning-rate-pc
ray job submit --runtime-env runtime_envs/runtime_env_without_ip.yaml --address $PSSR -- python main.py -c nature_less_erratic/traj-learning-rate-bp
```

<!-- ```bash
python analysis_v1.py \
-t "traj-learning-rate" \
-l "../general-energy-nets-results/nature_less_erratic/" \
-m "df['traj_length'].iloc[-1]" "df['final_length'].iloc[-1]" \
-f "./experiments/nature_less_erratic/traj-learning-rate-pc.yaml" "./experiments/nature_less_erratic/traj-learning-rate-bp.yaml" \
-v \
"df=nature_pre(df)" \
"g=sns.relplot(data=df,x='final_length',y='traj_length',hue='Rule',size='pc_learning_rate',style='Rule',alpha=0.75,col='seed',col_wrap=3)"
```

![](./traj-learning-rate-.png) -->

## two error bars (nature)

```bash
python analysis_v1.py \
-t "traj-learning-rate-two-ebars" \
-l "../general-energy-nets-results/nature_less_erratic/" \
-m "df['traj_length'].iloc[-1]" "df['final_length'].iloc[-1]" \
-f "./experiments/nature_less_erratic/traj-learning-rate-pc.yaml" "./experiments/nature_less_erratic/traj-learning-rate-bp.yaml" \
-v \
"df=nature_pre(df)" \
"from experiments.nature_less_erratic.utils import plot" \
"plot(df)"
```

![](./traj-learning-rate-two-ebars-.png)

## traj-learning-rate-apgr

```bash
ray job submit --runtime-env runtime_envs/runtime_env_without_ip.yaml --address $PSSR -- python main.py -c nature_less_erratic/traj-learning-rate-ap
ray job submit --runtime-env runtime_envs/runtime_env_without_ip.yaml --address $PSSR -- python main.py -c nature_less_erratic/traj-learning-rate-gr
```

### two error bars (nature)

```bash
python analysis_v1.py \
-t "traj-learning-rate-apgr-two-ebars" \
-l "../general-energy-nets-results/nature_less_erratic/" \
-m "df['traj_length'].iloc[-1]" "df['final_length'].iloc[-1]" \
-f "./experiments/nature_less_erratic/traj-learning-rate-ap.yaml" "./experiments/nature_less_erratic/traj-learning-rate-gr.yaml" \
-v \
"from experiments.nature_less_erratic.utils import plot" \
"plot(df,'learning_rate')"
```

![](./traj-learning-rate-apgr-two-ebars-.png)
