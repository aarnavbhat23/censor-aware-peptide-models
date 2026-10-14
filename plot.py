import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

frame=pd.read_csv('results/per-paper.csv');pivot=frame.pivot(index='paper',columns='model',values='observation_nll')
fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
delta=pivot.censored_likelihood-pivot.endpoint_gaussian
ax.barh(range(len(delta)),delta,color=['#216e91' if v<0 else '#ad661b' for v in delta])
ax.set_yticks(range(len(delta)),delta.index,fontsize=8);ax.axvline(0,color='#444',lw=1)
ax.set_xlabel('Censor-aware minus endpoint NLL (negative is better)');ax.spines[['top','right']].set_visible(False)
fig.savefig('results/source-transfer.png',dpi=180);fig.savefig('results/source-transfer.svg')
