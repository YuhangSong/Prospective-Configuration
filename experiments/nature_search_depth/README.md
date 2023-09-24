# base-focus

```bash
python main.py -c nature_search_depth/base-focus
```

## mean

```bash
python analysis_v1.py \
-t "mean-focus" \
-l "$RESULTS_DIR/nature_search_depth/" \
-m "df['test__classification_error'].mean()" \
-f "./experiments/nature_search_depth/base-focus.yaml" \
-g "init_fn" \
--fig-name fig3-g \
--source-include-columns pc_learning_rate Rule "Mean of test__classification_error" seed \
--source-columns-rename '{"pc_learning_rate": "learning rate"}' \
--source-df-filter '{"num_layers": 15}' \
-v \
"import experiments.nature_search_depth.utils as u" \
"df=u.plot_mean(df)"
```

![](./mean-focus-torch_nn_init_xavier_normal.png)

## mean-select_lr

```bash
python analysis_v1.py \
-t "mean-focus-select_lr" \
-l "$RESULTS_DIR/nature_search_depth/" \
-m "df['test__classification_error'].mean()" \
-f "./experiments/nature_search_depth/base-focus.yaml" \
--fig-name fig3-h \
--source-include-columns num_layers Rule "Mean of test__classification_error" seed \
-g "init_fn" \
-v \
"import experiments.nature_search_depth.utils as u" \
"df=u.plot_mean_select_lr(df, config_columns)"
```

![](./mean-focus-select_lr-torch_nn_init_xavier_normal.png)

## curve-select_lr

```bash
python analysis_v1.py \
-t "curve-focus-select_lr" \
-l "$RESULTS_DIR/nature_search_depth/" \
-m "compress_plot('test__classification_error','training_iteration')" "df['test__classification_error'].mean()" \
-f "./experiments/nature_search_depth/base-focus.yaml" \
-g "init_fn" \
--fig-name fig3-f \
--source-include-columns test__classification_error num_layers Rule training_iteration seed \
-v \
"import experiments.nature_search_depth.utils as u" \
"df=u.plot_curve(df, config_columns)"
```

![](./curve-focus-select_lr-torch_nn_init_xavier_normal.png)
