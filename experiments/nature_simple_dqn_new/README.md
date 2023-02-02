- [base-focus](#base-focus)
  - [curve (nature)](#curve-nature)

<!-- # base

```bash
ray job submit --runtime-env runtime_envs/runtime_env_without_ip.yaml --address $pssr --  python main.py -c nature_simple_dqn_new/base
```

## mean

```bash
python analysis_v1.py \
-t "base-mean" \
-l "$RESULTS_DIR/nature_simple_dqn_new/" \
-m "df['Episode Reward'].mean()" \
-f "./experiments/nature_simple_dqn_new/base.yaml" \
-g "Optimizer for inference" "Optimizer for learn" "is_q_target" "bias" "Inference rate" "is_norm_obs" "is_norm_rew" "batch_size" "buffer_limit" "gamma" "num_learn_epochs_per_eposide" "interval_update_target_q" "is_detach_target" "MainT" "pc_layer_at" "acf" \
-v \
"df=nature_pre(df)" \
"g=nature_relplot(data=df,x='pc_learning_rate',y='Mean of episode reward',hue='Rule',style='Rule',col='Env',sharey=False).set(xscale='log')" \
"nature_post(g)"
```

[doc](./base-mean.md)

## max

```bash
python analysis_v1.py \
-t "base-max" \
-l "$RESULTS_DIR/nature_simple_dqn_new/" \
-m "df['Episode Reward'].max()" \
-f "./experiments/nature_simple_dqn_new/base.yaml" \
-g "Optimizer for inference" "Optimizer for learn" "is_q_target" "bias" "Inference rate" "is_norm_obs" "is_norm_rew" "batch_size" "buffer_limit" "gamma" "num_learn_epochs_per_eposide" "interval_update_target_q" "is_detach_target" "MainT" "pc_layer_at" "acf" \
-v \
"df=nature_pre(df)" \
"g=nature_relplot(data=df,x='pc_learning_rate',y='Max of episode reward',hue='Rule',style='Rule',col='Env',sharey=False).set(xscale='log')" \
"nature_post(g)"
```

[doc](./base-max.md)

## curve

```bash
python analysis_v1.py \
-t "base-curve" \
-l "$RESULTS_DIR/nature_simple_dqn_new/" \
-m "compress_plot('Episode Reward','training_iteration')" "df['Episode Reward'].mean()" \
-f "./experiments/nature_simple_dqn_new/base.yaml" \
-g "Optimizer for inference" "Optimizer for learn" "is_q_target" "bias" "Inference rate" "is_norm_obs" "is_norm_rew" "batch_size" "buffer_limit" "gamma" "num_learn_epochs_per_eposide" "interval_update_target_q" "is_detach_target" "MainT" "pc_layer_at" "acf" \
-v \
"import experiments.nature_simple_dqn_new.utils as u" \
"u.plot(df)"
```

[doc](./base-curve.md) -->

# base-focus

<!-- 'Optimizer for inference': 'SGD', 'Optimizer for learn': 'SGD', 'is_q_target': False, 'bias': True, 'Inference rate': 0.05, 'is_norm_obs': True, 'is_norm_rew': False, 'batch_size': 60, 'buffer_limit': 50000, 'gamma': 0.98, 'num_learn_epochs_per_eposide': 10, 'interval_update_target_q': 20, 'is_detach_target': True, 'MainT': 32, 'pc_layer_at': 'before_acf', 'acf': 'Sigmoid' -->

```bash
ray job submit --runtime-env runtime_envs/runtime_env_without_ip.yaml --address $pssr --  python main.py -c nature_simple_dqn_new/base-focus
```

<!-- ## mean

```bash
python analysis_v1.py \
-t "base-focus-mean" \
-l "$RESULTS_DIR/nature_simple_dqn_new/" \
-m "df['Episode Reward'].mean()" \
-f "./experiments/nature_simple_dqn_new/base-focus.yaml" \
-g "Optimizer for inference" "Optimizer for learn" "is_q_target" "bias" "Inference rate" "is_norm_obs" "is_norm_rew" "batch_size" "buffer_limit" "gamma" "num_learn_epochs_per_eposide" "interval_update_target_q" "is_detach_target" "MainT" "pc_layer_at" "acf" \
-v \
"df=nature_pre(df)" \
"g=nature_relplot(data=df,x='pc_learning_rate',y='Mean of episode reward',hue='Rule',style='Rule',col='Env',sharey=False).set(xscale='log')" \
"nature_post(g)"
```

[doc](./base-focus-mean.md)

## max

```bash
python analysis_v1.py \
-t "base-focus-max" \
-l "$RESULTS_DIR/nature_simple_dqn_new/" \
-m "df['Episode Reward'].max()" \
-f "./experiments/nature_simple_dqn_new/base-focus.yaml" \
-g "Optimizer for inference" "Optimizer for learn" "is_q_target" "bias" "Inference rate" "is_norm_obs" "is_norm_rew" "batch_size" "buffer_limit" "gamma" "num_learn_epochs_per_eposide" "interval_update_target_q" "is_detach_target" "MainT" "pc_layer_at" "acf" \
-v \
"df=nature_pre(df)" \
"g=nature_relplot(data=df,x='pc_learning_rate',y='Max of episode reward',hue='Rule',style='Rule',col='Env',sharey=False).set(xscale='log')" \
"nature_post(g)"
```

[doc](./base-focus-max.md) -->

## curve (nature)

```bash
python analysis_v1.py \
-t "base-focus-curve" \
-l "$RESULTS_DIR/nature_simple_dqn_new/" \
-m "compress_plot('Episode Reward','training_iteration')" "df['Episode Reward'].mean()" \
-f "./experiments/nature_simple_dqn_new/base-focus.yaml" \
-g "Optimizer for inference" "Optimizer for learn" "is_q_target" "bias" "Inference rate" "is_norm_obs" "is_norm_rew" "batch_size" "buffer_limit" "gamma" "num_learn_epochs_per_eposide" "interval_update_target_q" "is_detach_target" "MainT" "pc_layer_at" "acf" \
-v \
"import experiments.nature_simple_dqn_new.utils as u" \
"u.plot(df)"
```

[doc](./base-focus-curve.md)

![](./base-focus-curve-SGD_SGD_False_True_0_05_True_False_60_50000_0_98_10_20_True_32_before_acf_Sigmoid.png)

<!-- # base-ffocus

'Inference rate': 0.05, 'MainT': 32

```bash
ray job submit --runtime-env runtime_envs/runtime_env_without_ip.yaml --address $pssr --  python main.py -c nature_simple_dqn_new/base-ffocus -m T0
``` -->

<!-- ## mean

```bash
python analysis_v1.py \
-t "base-ffocus-mean" \
-l "$RESULTS_DIR/nature_simple_dqn_new/" \
-m "df['Episode Reward'].mean()" \
-f "./experiments/nature_simple_dqn_new/base-ffocus.yaml" \
-g "Optimizer for inference" "Optimizer for learn" "is_q_target" "bias" "Inference rate" "is_norm_obs" "is_norm_rew" "batch_size" "buffer_limit" "gamma" "num_learn_epochs_per_eposide" "interval_update_target_q" "is_detach_target" "MainT" "pc_layer_at" "acf" \
-v \
"df=nature_pre(df)" \
"g=nature_relplot(data=df,x='pc_learning_rate',y='Mean of episode reward',hue='Rule',style='Rule',col='Env',sharey=False).set(xscale='log')" \
"nature_post(g)"
```

[doc](./base-ffocus-mean.md)

## max

```bash
python analysis_v1.py \
-t "base-ffocus-max" \
-l "$RESULTS_DIR/nature_simple_dqn_new/" \
-m "df['Episode Reward'].max()" \
-f "./experiments/nature_simple_dqn_new/base-ffocus.yaml" \
-g "Optimizer for inference" "Optimizer for learn" "is_q_target" "bias" "Inference rate" "is_norm_obs" "is_norm_rew" "batch_size" "buffer_limit" "gamma" "num_learn_epochs_per_eposide" "interval_update_target_q" "is_detach_target" "MainT" "pc_layer_at" "acf" \
-v \
"df=nature_pre(df)" \
"g=nature_relplot(data=df,x='pc_learning_rate',y='Max of episode reward',hue='Rule',style='Rule',col='Env',sharey=False).set(xscale='log')" \
"nature_post(g)"
```

[doc](./base-ffocus-max.md) -->

<!-- ## curve

```bash
python analysis_v1.py \
-t "base-ffocus-curve" \
-l "$RESULTS_DIR/nature_simple_dqn_new/" \
-m "compress_plot('Episode Reward','training_iteration')" "df['Episode Reward'].mean()" \
-f "./experiments/nature_simple_dqn_new/base-ffocus.yaml" \
-g "Optimizer for inference" "Optimizer for learn" "is_q_target" "bias" "Inference rate" "is_norm_obs" "is_norm_rew" "batch_size" "buffer_limit" "gamma" "num_learn_epochs_per_eposide" "interval_update_target_q" "is_detach_target" "MainT" "pc_layer_at" "acf" \
-v \
"import experiments.nature_simple_dqn_new.utils as u" \
"u.plot(df)"
```

[doc](./base-ffocus-curve.md)

## curve-loss

```bash
python analysis_v1.py \
-t "base-ffocus-curve-loss" \
-l "$RESULTS_DIR/nature_simple_dqn_new/" \
-m "compress_plot('Episode Loss','training_iteration')" "df['Episode Reward'].mean()" \
-f "./experiments/nature_simple_dqn_new/base-ffocus.yaml" \
-g "Optimizer for inference" "Optimizer for learn" "is_q_target" "bias" "Inference rate" "is_norm_obs" "is_norm_rew" "batch_size" "buffer_limit" "gamma" "num_learn_epochs_per_eposide" "interval_update_target_q" "is_detach_target" "MainT" "pc_layer_at" "acf" \
-v \
"import experiments.nature_simple_dqn_new.utils as u" \
"u.plot(df,plot="Loss")"
```

[doc](./base-ffocus-curve-loss.md) -->
