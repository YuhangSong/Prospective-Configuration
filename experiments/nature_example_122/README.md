- [Base](#base)
- [With small learning rate](#with-small-learning-rate)


<!-- # Energy machine

```bash
python main.py -c nature_example_122/machine -l
``` -->

# Base

<!-- ```bash
python main.py -c nature_example_122/levelmap
```

```bash
python analysis_v1.py \
-t "plot-levelmap" \
-l "$RESULTS_DIR/nature_example_122/" \
-m "df['w_2'].iloc[-1]" \
-f "./experiments/nature_example_122/levelmap.yaml" \
-v \
"import experiments.nature_example_122.utils as u" \
"u.plot_levelmap(df)"
```

![](./plot-levelmap-.png) -->

```bash
python main.py -c nature_example_122/traj-bp
python main.py -c nature_example_122/traj-pc
```

```bash
python analysis_v1.py \
-t "plot-traj" \
-l "$RESULTS_DIR/nature_example_122/" \
-m "df['train__w_1'].iloc[-1]" "df['train__w_2'].iloc[-1]" \
-f "./experiments/nature_example_122/traj-bp.yaml" "./experiments/nature_example_122/traj-pc.yaml" \
-v \
"import experiments.nature_example_122.utils as u" \
"u.plot_traj(df)"
```

![](./plot-traj-.png)

<!-- ```bash
python main.py -c nature_example_122/levelmap-output
```

```bash
python analysis_v1.py \
-t "plot-levelmap-output" \
-l "$RESULTS_DIR/nature_example_122/" \
-m "df['x_2'].iloc[-1]" \
-f "./experiments/nature_example_122/levelmap-output.yaml" \
-v \
"import experiments.nature_example_122.utils as u" \
"u.plot_levelmap(df,id='x')"
```

![](./plot-levelmap-output-.png) -->

```bash
python main.py -c nature_example_122/traj-output-bp
python main.py -c nature_example_122/traj-output-pc
```

```bash
python analysis_v1.py \
-t "plot-traj-output" \
-l "$RESULTS_DIR/nature_example_122/" \
-m "df['train__x_1'].iloc[-1]" "df['train__x_2'].iloc[-1]" \
-f "./experiments/nature_example_122/traj-output-bp.yaml" "./experiments/nature_example_122/traj-output-pc.yaml" \
-v \
"import experiments.nature_example_122.utils as u" \
"u.plot_traj(df,id='x')"
```


![](./plot-traj-output-.png)

<!-- ```bash

python main.py -c nature_example_122/traj-output-learn_last


python analysis_v1.py \
-t "plot-traj-output-learn_last" \
-l "$RESULTS_DIR/nature_example_122/" \
-m "df['train__x_1'].iloc[-1]" "df['train__x_2'].iloc[-1]" \
-f "./experiments/nature_example_122/traj-output-learn_last.yaml" \
-v \
"import experiments.nature_example_122.utils as u" \
"u.plot_traj(df,id='x')"
    
```

![](./plot-traj-output-learn_last-.png) -->

# With small learning rate

```bash
python main.py -c nature_example_122/traj-small-lr-bp
python main.py -c nature_example_122/traj-small-lr-pc
```

```bash
python analysis_v1.py \
-t "plot-traj-small-lr" \
-l "$RESULTS_DIR/nature_example_122/" \
-m "df['train__w_1'].iloc[-1]" "df['train__w_2'].iloc[-1]" \
-f "./experiments/nature_example_122/traj-small-lr-bp.yaml" "./experiments/nature_example_122/traj-small-lr-pc.yaml" \
-v \
"import experiments.nature_example_122.utils as u" \
"u.plot_traj(df,is_style_rule=False)"
```

![](./plot-traj-small-lr-.png)

```bash
python main.py -c nature_example_122/traj-small-lr-output-bp
python main.py -c nature_example_122/traj-small-lr-output-pc
```

```bash
python analysis_v1.py \
-t "plot-traj-small-lr-output" \
-l "$RESULTS_DIR/nature_example_122/" \
-m "df['train__x_1'].iloc[-1]" "df['train__x_2'].iloc[-1]" \
-f "./experiments/nature_example_122/traj-small-lr-output-bp.yaml" "./experiments/nature_example_122/traj-small-lr-output-pc.yaml" \
-v \
"import experiments.nature_example_122.utils as u" \
"u.plot_traj(df,id='x',is_style_rule=False)"
```


![](./plot-traj-small-lr-output-.png)