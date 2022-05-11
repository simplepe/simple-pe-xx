# get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pylab as plt
from simple_pe.detectors import network
from simple_pe.localization import event
import cartopy.crs as ccrs
from matplotlib.path import Path


# Set the plotting parameters:
plt.rcParams.update({
    "lines.markersize": 6,
    "lines.markeredgewidth": 1.5,
    "lines.linewidth": 1.0,
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 20,
})


def calc_network_response(events):
    """
    Calculate the network alignment factor for a 2D list of events sorted by the 2d ra and dec grids.
    """
    response = np.zeros_like(events, dtype=float)
    for i, ev in np.ndenumerate(events):
        response[i] = np.linalg.norm(ev.get_f())
    response /= np.linalg.norm(ev.get_data('sigma'))
    return response


def calc_alignment_factor(events):
    """
    Calculate the network alignment factor for a 2D list of events sorted by the 2d ra and dec grids.
    """
    alignment = np.zeros_like(events, dtype=float)
    for i, ev in np.ndenumerate(events):
        alignment[i] = ev.get_f()[1] / ev.get_f()[0]
    return alignment


net_state = "o3"
found_thresh = 0.0
net_thresh = 1.0
loc_thresh = 0.0
duty_cycle = 1.0
a_max = 200.
a_step = 1.0
a_window = 5.0
w = np.hanning(2 * a_window / a_step)

savefig = True
savetxt = True

## Run a set of fake events, fixed distance, face on
################################################################
dra, ddec = np.pi / 100.0, np.pi / 100.0
[ra, dec] = np.mgrid[-np.pi:np.pi + dra * 0.9:dra, -np.pi / 2:np.pi / 2 + ddec * 0.9:dra]

# set distance 
Dco = 40.
psi = 0.
cosi = 1.

params = {'distance': Dco,
          'polarization': psi,
          'coa-phase': 0.,
          'inclination': 0.,
          'RAdeg': 0.,
          'DEdeg': 0.,
          'gps': 999995381,
          'mass1': 1.35,
          'mass2': 1.35}

n = network.Network()
n.set_configuration(net_state, found_thresh, loc_thresh, duty_cycle)
all_events = np.zeros_like(ra, dtype=object)
for i, r in np.ndenumerate(ra):
    params['RAdeg'] = np.degrees(r)
    params['DEdeg'] = np.degrees(dec[i])
    x = event.Event(params=params)
    x.add_network(n)
    all_events[i] = x

f_response = calc_network_response(all_events)
sky_coverage = np.sum((f_response > (np.amax(f_response) / np.sqrt(2))) * np.cos(dec)) / np.sum(np.cos(dec))
fname = "%s_sky_coverage.txt" % net_state
if savetxt:
    f = open(fname, "w")
    f.write('%s: best sensitivity = %.2f, sky coverage (Schutz FOM 2) = %.2f\n' % (
        net_state, f_response.max(), sky_coverage))

# ## Calculate alignment factors

alphas = calc_alignment_factor(all_events)
pol_sky = np.sum((alphas > np.amax(alphas) / np.sqrt(2)) * np.cos(dec)) / np.sum(np.cos(dec))
pol_source = np.sum((alphas > np.amax(alphas) / np.sqrt(2)) * np.cos(dec) * f_response ** 3) / np.sum(
    np.cos(dec) * f_response ** 3)
if savetxt:
    f.write('%s: mean alpha = %.2f, fract of sky with alpha > alpha_max/sqrt(2) = %.2f\n'
            % (net_state, np.sum(alphas * np.cos(dec)) / np.sum(np.cos(dec)), pol_sky))
    f.write('%s: mean (source weighted) alpha: %.2f\n'
            % (net_state, np.sum(alphas * np.cos(dec) * f_response ** 3) \
               / np.sum(np.cos(dec) * f_response ** 3)))
    f.close()

'''For plotting the L shaped interferometer marker have to manually draw them as in this cell:
'''

verts = [
    (0., 1.),  # left, top
    (0., 0.),  # left, bottom
    (1., 0.),  # right, bottom
]
codes = [Path.MOVETO,
         Path.LINETO,
         Path.LINETO,
         ]
path = Path(verts, codes)
data_crs = ccrs.PlateCarree()

v = np.linspace(0.0, 1.0, 41)
plt.figure(figsize=(10, 10))

ax = plt.axes(projection=ccrs.Mollweide())
ax.coastlines()
for ifo in n.ifos:
    ax.plot(np.degrees(getattr(n, ifo).longitude), np.degrees(getattr(n, ifo).latitude), marker=path, markersize=20,
            markeredgewidth=4, markerfacecolor='None', markeredgecolor='m', transform=data_crs)

cf = ax.contourf(np.degrees(ra), np.degrees(dec), f_response, v, cmap=plt.cm.viridis, transform=data_crs)
plt.colorbar(cf, fraction=0.046, pad=0.04)
plt.savefig("%s_sky_sens.png" % net_state, dpi=200, bbox_inches='tight')

plt.figure(figsize=(10, 10))

ax = plt.axes(projection=ccrs.Mollweide())
ax.coastlines()
for ifo in n.ifos:
    ax.plot(np.degrees(getattr(n, ifo).longitude), np.degrees(getattr(n, ifo).latitude), marker=path, markersize=20,
            markeredgewidth=4, markerfacecolor='None', markeredgecolor='m', transform=data_crs)

cf = ax.contourf(np.degrees(ra), np.degrees(dec), f_response ** 3, v, cmap=plt.cm.viridis, transform=ccrs.PlateCarree())
plt.colorbar(cf, fraction=0.046, pad=0.04)

plt.savefig(".%s_sky_rate.png" % net_state, dpi=200, bbox_inches='tight')

plt.figure(figsize=(10, 10))

ax = plt.axes(projection=ccrs.Mollweide())
ax.coastlines()
for ifo in n.ifos:
    ax.plot(np.degrees(getattr(n, ifo).longitude), np.degrees(getattr(n, ifo).latitude), marker=path, markersize=20,
            markeredgewidth=4, markerfacecolor='None', markeredgecolor='m', transform=data_crs)

cf = ax.contourf(np.degrees(ra), np.degrees(dec), alphas, v, cmap=plt.cm.viridis, transform=ccrs.PlateCarree())
plt.colorbar(cf, fraction=0.046, pad=0.04)

plt.savefig("%s_2nd_pol.png" % net_state, dpi=200, bbox_inches='tight')
