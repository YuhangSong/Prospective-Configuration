<!-- TOC -->

- [base-plot](#base-plot)
  - [Fit plot to data](#fit-plot-to-data)
  - [Apply fit data and plot](#apply-fit-data-and-plot)
- [base-plot-fit-plot-reason](#base-plot-fit-plot-reason)
- [base-reason-simple](#base-reason-simple)
- [base-reason-simple-inference](#base-reason-simple-inference)

<!-- /TOC -->

<!-- # base

Initial investigation: search different configurations and get possible configurations that can produce similar pattern in the data from biological experiments.

```bash
ray job submit --runtime-env runtime_envs/runtime_env_without_ip.yaml --address $PSSR -- python main.py -c nature_reversal_task/base
```

```bash
python analysis_v1.py \
-t "base" \
-l "../general-energy-nets-results/nature_reversal_task/" \
-m "compress_plot('log','training_iteration')" \
-f "./experiments/nature_reversal_task/base.yaml" \
-g "init_std" \
-v \
"import experiments.nature_reversal_task.utils as u" \
"df=u.prepare_df(df)" \
"u.plot(df,row='pc_learning_rate',sharey=False)"
```

[doc](./base.md)

Successful. -->

# base-plot

Plot the best fit pattern from the above base.

```bash
ray job submit --runtime-env runtime_envs/runtime_env_without_ip.yaml --address $PSSR -- python main.py -c nature_reversal_task/base-plot-bp
ray job submit --runtime-env runtime_envs/runtime_env_without_ip.yaml --address $PSSR -- python main.py -c nature_reversal_task/base-plot-pc
```

```bash
python analysis_v1.py \
-t "base-plot" \
-l "../general-energy-nets-results/nature_reversal_task/" \
-m "compress_plot('log','training_iteration')" \
-f "./experiments/nature_reversal_task/base-plot-pc.yaml" "./experiments/nature_reversal_task/base-plot-bp.yaml" \
-v \
"import experiments.nature_reversal_task.utils as u" \
"df=u.prepare_df(df)" \
"u.plot(df, sharey=False)"
```

![](./base-plot-.png)

## Fit plot to data

Fit the above plot to the data in biological experiments.

Data in biological experiments is defined in ```.utils.py```.

```bash
python analysis_v1.py \
-t "base-plot-fit" \
-l "../general-energy-nets-results/nature_reversal_task/" \
-m "compress_plot('log','training_iteration')" \
-f "./experiments/nature_reversal_task/base-plot-pc.yaml" "./experiments/nature_reversal_task/base-plot-bp.yaml" \
-g "PC" \
-v \
"import experiments.nature_reversal_task.utils as u" \
"df=u.prepare_df(df)" \
"df=u.reduce_mean(df)" \
"df=u.fit_data(df, kw_id)" \
"print()" \
"print(kw_id)" \
"print(df2tb(df.head(3)))"
```

Make sure the ```fit_data_w``` and ```fit_data_b``` are within the range of the search defined in ```.utils.py```.

## Apply fit data and plot

```bash
python analysis_v1.py \
-t "base-plot-fit-plot" \
-l "../general-energy-nets-results/nature_reversal_task/" \
-m "compress_plot('log','training_iteration')" \
-f "./experiments/nature_reversal_task/base-plot-pc.yaml" "./experiments/nature_reversal_task/base-plot-bp.yaml" \
-v \
"import experiments.nature_reversal_task.utils as u" \
"df=u.prepare_df(df)" \
"df=u.apply_fit_data(df)" \
"df=u.add_data(df)" \
"u.plot(df, sharey=True)"
```

![](./base-plot-fit-plot-.png)

# base-plot-fit-plot-reason

Show the reason of such learnign effects.

```bash
ray job submit --runtime-env runtime_envs/runtime_env_without_ip.yaml --address $PSSR -- python main.py -c nature_reversal_task/base-plot-reason-bp
ray job submit --runtime-env runtime_envs/runtime_env_without_ip.yaml --address $PSSR -- python main.py -c nature_reversal_task/base-plot-reason-pc
```

```bash
python analysis_v1.py \
-t "base-plot-fit-plot-reason" \
-l "../general-energy-nets-results/nature_reversal_task/" \
-m "compress_plot('weight','training_iteration')" \
-f "./experiments/nature_reversal_task/base-plot-reason-pc.yaml" "./experiments/nature_reversal_task/base-plot-reason-bp.yaml" \
-v \
"import experiments.nature_reversal_task.utils as u" \
"df=u.prepare_df_reason(df)" \
"u.plot_reason(df, sharey=True)"
```

![](./base-plot-fit-plot-reason-.png)

<!-- ```bash
python analysis_v1.py \
-t "base-plot-fit-plot-reason-seed" \
-l "../general-energy-nets-results/nature_reversal_task/" \
-m "compress_plot('weight','training_iteration')" \
-f "./experiments/nature_reversal_task/base-plot-reason-pc.yaml" "./experiments/nature_reversal_task/base-plot-reason-bp.yaml" \
-v \
"import experiments.nature_reversal_task.utils as u" \
"df=u.prepare_df_reason(df)" \
"u.plot_reason_seed(df,y='weight')"
```

![](./base-plot-fit-plot-reason-seed-.png) -->

<!-- ```bash
python analysis_v1.py \
-t "base-plot-fit-plot-reason-seed-id" \
-l "../general-energy-nets-results/nature_reversal_task/" \
-m "compress_plot('correct_id','training_iteration')" \
-f "./experiments/nature_reversal_task/base-plot-reason-pc.yaml" "./experiments/nature_reversal_task/base-plot-reason-bp.yaml" \
-v \
"import experiments.nature_reversal_task.utils as u" \
"df=u.prepare_df_reason(df,y='correct_id')" \
"u.plot_reason_seed(df,y='correct_id')"
```

![](./base-plot-fit-plot-reason-seed-id-.png) -->

# base-reason-simple

Show the reason of such learnign effects with a simple example.

```bash
ray job submit --runtime-env runtime_envs/runtime_env_without_ip.yaml --address $PSSR -- python main.py -c nature_reversal_task/base-reason-simple
```

```bash
python analysis_v1.py \
-t "base-reason-simple" \
-l "../general-energy-nets-results/nature_reversal_task/" \
-m "compress_plot('weight','training_iteration')" \
-f "./experiments/nature_reversal_task/base-reason-simple.yaml" \
-v \
"import experiments.nature_reversal_task.utils as u" \
"df=u.prepare_df_reason(df)" \
"u.plot_reason_simple(df, sharey=True)"
```

![](./base-reason-simple-.png)

# base-reason-simple-inference

Show the reason of such learnign effects with a simple example, the inference of the hidden neuron.

```bash
ray job submit --runtime-env runtime_envs/runtime_env_without_ip.yaml --address $PSSR -- python main.py -c nature_reversal_task/base-reason-simple-inference
```

```bash
python analysis_v1.py \
-t "base-reason-simple-inference" \
-l "../general-energy-nets-results/nature_reversal_task/" \
-m "eval(df['value-along-t'].iloc[-1])" \
-f "./experiments/nature_reversal_task/base-reason-simple-inference.yaml" \
-v \
"import experiments.nature_reversal_task.utils as u" \
"df=u.prepare_df_reason_inference(df)" \
"u.plot_reason_simple_inference(df)"
```

![](./base-reason-simple-inference-.png)
