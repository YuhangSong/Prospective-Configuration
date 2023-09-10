# traj-learning-rate

Put traj length and learning rate into the same plot

```bash
python main.py -c nature_less_erratic/traj-learning-rate-pc
python main.py -c nature_less_erratic/traj-learning-rate-bp
```

## two error bars (nature)

```bash
python analysis_v1.py \
-t "traj-learning-rate-two-ebars" \
-l "$RESULTS_DIR/nature_less_erratic/" \
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
python main.py -c nature_less_erratic/traj-learning-rate-ap
python main.py -c nature_less_erratic/traj-learning-rate-gr
```

### two error bars (nature)

```bash
python analysis_v1.py \
-t "traj-learning-rate-apgr-two-ebars" \
-l "$RESULTS_DIR/nature_less_erratic/" \
-m "df['traj_length'].iloc[-1]" "df['final_length'].iloc[-1]" \
-f "./experiments/nature_less_erratic/traj-learning-rate-ap.yaml" "./experiments/nature_less_erratic/traj-learning-rate-gr.yaml" \
-v \
"from experiments.nature_less_erratic.utils import plot" \
"plot(df,'learning_rate')"
```

![](./traj-learning-rate-apgr-two-ebars-.png)
