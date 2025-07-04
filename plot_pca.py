import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------------------------------------
# 1)  Put *your* PCA coordinates into a tidy dataframe
# -----------------------------------------------------------
# Replace these toy numbers with your real data ↓↓↓
df = pd.DataFrame({
    'pc1': [-1150, -1100, -1080,           # meta-trained
            -1180, -1125, -1060,           # single-task
             600, 4500, 5200, 5700, 6400], # vanilla
    'pc2': [   30,    20,    25,           # meta-trained
             5040, 4980, 4950,             # single-task
               50, -1200, -1500, -1800, 5050],  # vanilla
    'label': ['Meta-Trained']*3 +
             ['Single-Task']*3 +
             ['Vanilla']*5
})

# -----------------------------------------------------------
# 2)  Create the “Initial Weights”  ★ points
#     (here:  +400 PC1   +150 PC2  from each vanilla point)
# -----------------------------------------------------------
offset = dict(pc1 = +400, pc2 = +150)
df_init = (df.query("label == 'Vanilla'")
             .assign(label='Initial Weights',
                     pc1=lambda d: d.pc1 + offset['pc1'],
                     pc2=lambda d: d.pc2 + offset['pc2']))

plot_df = pd.concat([df, df_init], ignore_index=True)

# -----------------------------------------------------------
# 3)  Plot on one clean set of axes
# -----------------------------------------------------------
markers = {'Meta-Trained':'^', 'Single-Task':'s', 'Vanilla':'o',
           'Initial Weights':'*'}
colors  = {'Meta-Trained':'tab:blue', 'Single-Task':'tab:green',
           'Vanilla':'tab:red',     'Initial Weights':'orange'}

fig, ax = plt.subplots(figsize=(9,7))
for lbl, grp in plot_df.groupby('label'):
    ax.scatter(grp.pc1, grp.pc2,
               marker=markers[lbl], s=130,
               color=colors[lbl], edgecolor='k' if lbl!='Initial Weights' else 'none',
               label=lbl)

ax.set_xlabel('Principal Component 1 (54.5 %)')
ax.set_ylabel('Principal Component 2 (16.2 %)')
ax.grid(alpha=.3)
ax.legend(title='Training Type', loc='center right')
plt.tight_layout()
plt.savefig('pca_no_inset.png', dpi=300)
