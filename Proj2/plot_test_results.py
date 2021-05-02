import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("results/test_output.dat", delim_whitespace=True,
                    names=['x', 'y', 'pred', 'truth'], usecols=(0,1,2,3),
                    dtype={'x':'float','y':'float','pred':'bool','truth':'bool'})

fig, ax = plt.subplots(1, 2, figsize=(10,5))

ax[0].scatter(data.loc[data['pred'],'x'], data.loc[data['pred'],'y'], c = 'r')
ax[0].scatter(data.loc[~data['pred'],'x'], data.loc[~data['pred'],'y'], c = 'b')
ax[0].set_title("Prediction")
ax[1].scatter(data.loc[data['truth'],'x'], data.loc[data['truth'],'y'], c = 'r')
ax[1].scatter(data.loc[~data['truth'],'x'], data.loc[~data['truth'],'y'], c = 'b')
ax[1].set_title("Truth")
fig.savefig("results/test_output.png", dpi=150, bbox_inches='tight')
