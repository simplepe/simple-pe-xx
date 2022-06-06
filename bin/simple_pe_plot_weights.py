import matplotlib.pyplot as plt
import h5py
import corner
import numpy as np

# import the weighted samples, plot the weights
f = h5py.File('weighted_samples.hdf5')

plt.figure(figsize=(12,8))
plt.scatter(f['eta'], np.degrees(f['theta']), c=f['p_33'])
plt.xlabel('eta')
plt.ylabel('theta (degrees)')
plt.colorbar(label='p(33)')
plt.grid()
plt.savefig('p33_vs_eta_theta.png')

plt.figure(figsize=(12,8))
plt.scatter(f['eta'], np.degrees(f['theta']), c=f['p_2pol'])
plt.xlabel('eta')
plt.ylabel('theta (degrees)')
plt.colorbar(label='p(2pol)')
plt.grid()
plt.savefig('p_2pol_vs_eta_theta.png')


plt.figure(figsize=(12,8))
plt.scatter(f['eta'], np.degrees(f['theta']), c=f['p_33'][:] * f['p_2pol'][:])
plt.xlabel('eta')
plt.ylabel('theta (degrees)')
plt.colorbar(label='p(33, 2pol)')
plt.grid()
plt.savefig('p_33_2pol_vs_eta_theta.png')

plt.figure(figsize=(12,8))
plt.scatter(f['chi_p'], np.degrees(f['theta']), c=f['p_prec'])
plt.xlabel('chi_p')
plt.ylabel('theta (degrees)')
plt.colorbar(label='p(prec)')
plt.grid()
plt.savefig('p_prec_vs_chi_p_theta.png')

plt.figure(figsize=(12,8))
plt.scatter(f['chi_p'], np.degrees(f['theta']), c=f['weights'])
plt.xlabel('chi_p')
plt.ylabel('theta (degrees)')
plt.colorbar(label='p(33, 2_pol, prec)')
plt.grid()
plt.savefig('p_33_2pol_prec_vs_chi_p_theta.png')