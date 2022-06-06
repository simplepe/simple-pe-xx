import matplotlib.pyplot as plt
import h5py
import corner
import numpy as np

f = h5py.File('final_samples.hdf5')

corner.corner(np.array([f['mchirp'],f['eta'],f['chi_eff'],f['chi_p'],f['theta']]).T,
              labels=["chirp mass", "symmetric mass ratio", "effective spin", "chi_p", "theta_jn"],
                       quantiles=[0.05, 0.5, 0.95],
                       show_titles=True, title_kwargs={"fontsize": 12})
plt.savefig('corner_mc_eta_chieff_chip_theta.png')