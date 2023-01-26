<!-- TOC -->

- [base](#base)

<!-- /TOC -->

```bash
e218fde351853aaa9a133a1b5a721656f4bd8c22
```

# base

```bash
ray job submit --runtime-env runtime_envs/runtime_env_without_ip.yaml --address $PSSR -- python main.py -c nature_pc_pi_layered/base
```

```bash
python analysis_v1.py \
-t "base" \
-l "../general-energy-nets-results/nature_pc_pi_layered/" \
-m "df['train__prospective_index'].mean()" \
-f "./experiments/nature_pc_pi_layered/base.yaml" \
-v \
"ax=sns.stripplot(data=df,x='l',y='Prospective index',hue='update_p_at',size=20,marker='D',edgecolor='gray',alpha=0.25,palette=np.concatenate((sns.color_palette()[6:7], sns.color_palette()[0:1])))"
```

![](./base-.png)
