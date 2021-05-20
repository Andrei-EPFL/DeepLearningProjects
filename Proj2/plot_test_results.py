import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv("results/test_output.dat", delim_whitespace=True,
                    names=['x', 'y', 'pred', 'truth'], usecols=(0,1,2,3),
                    dtype={'x':'float','y':'float','pred':'bool','truth':'bool'})

datatorch = pd.read_csv("results/pytorch_test_output.dat", delim_whitespace=True,
                    names=['x', 'y', 'pred', 'truth'], usecols=(0,1,2,3),
                    dtype={'x':'float','y':'float','pred':'bool','truth':'bool'})


theta = np.linspace( 0 , 2 * np.pi , 150 )
radius = np.sqrt(1. / (2 * np.pi))
a = radius * np.cos( theta )
b = radius * np.sin( theta )

# fig, ax = plt.subplots(1, 2, figsize=(10,5))
fig, ax = plt.subplots(1, 1, figsize=(5,5))

range_t_ndl = np.logical_and(datatorch['pred'], ~data['pred'])
range_t_dl = np.logical_and(datatorch['pred'], data['pred'])
range_nt_dl = np.logical_and(~datatorch['pred'], data['pred'])
range_nt_ndl = np.logical_and(~datatorch['pred'], ~data['pred'])


ax.scatter(datatorch.loc[range_t_ndl,'x'], datatorch.loc[range_t_ndl,'y'], c = 'orange', s=10, label="t 1 & d 0")
ax.scatter(datatorch.loc[range_nt_dl,'x'], datatorch.loc[range_nt_dl,'y'], c = 'lime', s=10, label="t 0 & d 1")
ax.scatter(datatorch.loc[range_nt_ndl,'x'], datatorch.loc[range_nt_ndl,'y'], c = 'dodgerblue', s=10, label="t 0 & d 0")
ax.scatter(datatorch.loc[range_t_dl,'x'], datatorch.loc[range_t_dl,'y'], c = 'red', s=10, label="t 1 & d 1")
ax.plot(a + 0.5, b + 0.5, color="grey")
ax.legend(loc="upper center", bbox_to_anchor=(0.23, 0.59, 0.5, 0.5), ncol=4)
# ax[0].scatter(datatorch.loc[datatorch['pred'],'x'], datatorch.loc[datatorch['pred'],'y'], c = 'orange')

# ax[0].scatter(data.loc[data['pred'],'x'], data.loc[data['pred'],'y'], c = 'r')
#ax[0].scatter(data.loc[~data['pred'],'x'], data.loc[~data['pred'],'y'], c = 'b')

#ax[0].scatter(datatorch.loc[~datatorch['pred'],'x'], datatorch.loc[~datatorch['pred'],'y'], c = 'grey')

# ax[0].set_title("Prediction")
# ax[1].scatter(data.loc[data['truth'],'x'], data.loc[data['truth'],'y'], c = 'r')
# ax[1].scatter(data.loc[~data['truth'],'x'], data.loc[~data['truth'],'y'], c = 'b')
# ax[1].set_title("Truth")
# ax[0].scatter(0.3829, 0.9593, color="green")
# ax[0].scatter(0.4000, 0.6014, color="k")

fig.savefig("results/test_output.png", dpi=150, bbox_inches='tight')
plt.show()