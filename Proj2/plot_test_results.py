import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_(ax, data_pt, data_dl):
    theta = np.linspace( 0 , 2 * np.pi , 150 )
    radius = np.sqrt(1. / (2 * np.pi))
    a = radius * np.cos( theta )
    b = radius * np.sin( theta )

    range_t_ndl = np.logical_and(data_pt['pred'], ~data_dl['pred'])
    range_t_dl = np.logical_and(data_pt['pred'], data_dl['pred'])
    range_nt_dl = np.logical_and(~data_pt['pred'], data_dl['pred'])
    range_nt_ndl = np.logical_and(~data_pt['pred'], ~data_dl['pred'])

    print(len(data_pt.loc[range_t_ndl,'x']))
    print(len(data_pt.loc[range_nt_dl,'x']))
    ax.scatter(data_pt.loc[range_t_ndl,'x'], data_pt.loc[range_t_ndl,'y'], c = 'orange', s=10, label="t 1 & d 0")
    ax.scatter(data_pt.loc[range_nt_dl,'x'], data_pt.loc[range_nt_dl,'y'], c = 'lime', s=10, label="t 0 & d 1")
    ax.scatter(data_pt.loc[range_nt_ndl,'x'], data_pt.loc[range_nt_ndl,'y'], c = 'dodgerblue', s=10, label="t 0 & d 0")
    ax.scatter(data_pt.loc[range_t_dl,'x'], data_pt.loc[range_t_dl,'y'], c = 'red', s=10, label="t 1 & d 1")
    ax.plot(a + 0.5, b + 0.5, color="grey")



data_dl32 = pd.read_csv("results/float32_dl_test_output_S42.dat", delim_whitespace=True,
                    names=['x', 'y', 'pred', 'truth'], usecols=(0,1,2,3),
                    dtype={'x':'float','y':'float','pred':'bool','truth':'bool'})

data_pt32 = pd.read_csv("results/float32_pt_test_output_S42.dat", delim_whitespace=True,
                    names=['x', 'y', 'pred', 'truth'], usecols=(0,1,2,3),
                    dtype={'x':'float','y':'float','pred':'bool','truth':'bool'})


data_dl64 = pd.read_csv("results/float64_dl_test_output_S42.dat", delim_whitespace=True,
                    names=['x', 'y', 'pred', 'truth'], usecols=(0,1,2,3),
                    dtype={'x':'float','y':'float','pred':'bool','truth':'bool'})

data_pt64 = pd.read_csv("results/float64_pt_test_output_S42.dat", delim_whitespace=True,
                    names=['x', 'y', 'pred', 'truth'], usecols=(0,1,2,3),
                    dtype={'x':'float','y':'float','pred':'bool','truth':'bool'})



fig, ax = plt.subplots(2, 1, figsize=(5,10))

plot_(ax[0],data_pt32, data_dl32)
plot_(ax[1],data_pt64, data_dl64)
ax[0].legend(loc="upper center", bbox_to_anchor=(0.23, 0.59, 0.5, 0.5), ncol=4)

#fig, ax = plt.subplots(1, 1, figsize=(5,5))

# ax[0].scatter(data_pt.loc[data_pt['pred'],'x'], data_pt.loc[data_pt['pred'],'y'], c = 'orange')

# ax[0].scatter(data.loc[data_dl['pred'],'x'], data.loc[data_dl['pred'],'y'], c = 'r')
#ax[0].scatter(data.loc[~data_dl['pred'],'x'], data.loc[~data_dl['pred'],'y'], c = 'b')

#ax[0].scatter(data_pt.loc[~data_pt['pred'],'x'], data_pt.loc[~data_pt['pred'],'y'], c = 'grey')

# ax[0].set_title("Prediction")
# ax[1].scatter(data.loc[data_dl['truth'],'x'], data.loc[data_dl['truth'],'y'], c = 'r')
# ax[1].scatter(data.loc[~data_dl['truth'],'x'], data.loc[~data_dl['truth'],'y'], c = 'b')
# ax[1].set_title("Truth")
# ax[0].scatter(0.3829, 0.9593, color="green")
# ax[0].scatter(0.4000, 0.6014, color="k")

fig.savefig("results/test_output.png", dpi=150, bbox_inches='tight')
plt.show()