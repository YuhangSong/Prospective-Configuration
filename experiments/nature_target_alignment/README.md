-   [target alignment for deeper nets](#target-alignment-for-deeper-nets)
    -   [base-depth (nature)](#base-depth-nature)
    -   [base-depth-acf (nature)](#base-depth-acf-nature)
    -   [base-depth-init (nature)](#base-depth-init-nature)
    -   [base-depth-orth-init (nature)](#base-depth-orth-init-nature)

# base-112

Simulating the problem in figure 1 about interference.

```bash

python main.py -c nature_target_alignment/base-112
```

## plot-112

```bash
python analysis_v1.py \
-t "plot-112" \
-l "$RESULTS_DIR/nature_target_alignment/" \
-m "compress_plot('prediction','training_iteration')" \
-f "./experiments/nature_target_alignment/base-112.yaml" \
--fig-name fig3-a \
--source-include-columns "Rule" "training_iteration" "x_0" "x_1" \
-v \
"import experiments.nature_target_alignment.utils as u" \
"df=u.plot_112(df)"
```

![](./plot-112-.png)

# base-112-lr

Simulating the problem in figure 1 about interference, with search on learning rate.

```bash
python main.py -c nature_target_alignment/base-112-lr
```

```bash
python analysis_v1.py \
-t "plot-112-lr" \
-l "$RESULTS_DIR/nature_target_alignment/" \
-m "compress_plot('prediction','training_iteration')" \
-f "./experiments/nature_target_alignment/base-112-lr.yaml" \
--fig-name fig3-d \
--source-exclude-columns acf batch_size hidden_size num_iterations num_layers seed \
-v \
"import experiments.nature_target_alignment.utils as u" \
"df=u.plot_112_lr(df)"
```

![](./plot-112-lr-.png)

## base-depth-width-linear-angle (nature)

```bash
python main.py -c nature_target_alignment/base-depth-width-linear-angle
```

### plot (nature)

```bash
python analysis_v1.py \
-t "plot-depth-width-linear-angle_alignment" \
-l "$RESULTS_DIR/nature_target_alignment/" \
-m "df['train__target_alignment'].iloc[-1]" \
-f "./experiments/nature_target_alignment/base-depth-width-linear-angle.yaml" \
--fig-name fig3-e \
--source-exclude-columns pc_learning_rate acf batch_size gain_lg hidden_size init_fn input_target_std 'log_packs: prediction_std: at_data_pack' 'log_packs: prediction_std: log' 'log_packs: target_alignment: at_data_pack' 'log_packs: target_alignment: log' 'log_packs: target_alignment_angle: at_data_pack' 'log_packs: target_alignment_angle: log' num_iterations \
-v \
"import experiments.nature_target_alignment.utils as u" \
"u.plot_depth_width_linear_angle_alignment(df)"
```

![](./plot-depth-width-linear-angle_alignment-.png)

# target alignment for deeper nets

## base-depth (nature)

Target alignment of bp, pc and tp (target propropagation)

```bash
python main.py -c nature_target_alignment/base-depth
python main.py -c nature_target_alignment/base-depth-tp
```

```bash
python analysis_v1.py \
-t "base-depth" \
-l "$RESULTS_DIR/nature_target_alignment/" \
-m "df['train__target_alignment'].iloc[-1]" \
-f "./experiments/nature_target_alignment/base-depth.yaml" "./experiments/nature_target_alignment/base-depth-tp.yaml" \
-v \
"import experiments.nature_target_alignment.utils as u" \
"u.base_depth(df)"
```

![](./base-depth-.png)

## base-depth-acf (nature)

```bash
python main.py -c nature_target_alignment/base-depth-acf
```

```bash
python analysis_v1.py \
-t "base-depth-acf" \
-l "$RESULTS_DIR/nature_target_alignment/" \
-m "df['train__target_alignment'].iloc[-1]" \
-f "./experiments/nature_target_alignment/base-depth-acf.yaml" \
-v \
"import experiments.nature_target_alignment.utils as u" \
"u.base_depth_acf(df)"
```

![](./base-depth-acf-.png)

Notes:

-   sigmoid give high nature_target_alignment for both bp and pc, due to that it igonres input, so formally run tanh and identity to presenting in the paper
-   what is the role of batch_size?

## base-depth-init (nature)

As suggested by the reviewer, use orthogonal init as in Exact solutions to the nonlinear dynamics of learning in deep linear neural networks - Saxe, A. et al. (2013).

```bash
python main.py -c nature_target_alignment/base-depth-init
```

```bash
python analysis_v1.py \
-t "base-depth-init" \
-l "$RESULTS_DIR/nature_target_alignment/" \
-m "df['train__target_alignment'].iloc[-1]" \
-f "./experiments/nature_target_alignment/base-depth-init.yaml" \
-v \
"import experiments.nature_target_alignment.utils as u" \
"u.base_depth_init(df)"
```

![](./base-depth-init-.png)

## base-depth-orth-init (nature)

Look at if orth init have difference with different learning rates.

```bash
python main.py -c nature_target_alignment/base-depth-orth-init
```

```bash
python analysis_v1.py \
-t "base-depth-orth-init" \
-l "$RESULTS_DIR/nature_target_alignment/" \
-m "df['train__target_alignment'].iloc[-1]" \
-f "./experiments/nature_target_alignment/base-depth-orth-init.yaml" \
-v \
"import experiments.nature_target_alignment.utils as u" \
"u.base_depth_orth_init(df)"
```

![](./base-depth-orth-init-.png)
