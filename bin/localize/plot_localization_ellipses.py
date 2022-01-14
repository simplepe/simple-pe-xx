get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pylab as plt

from simple_pe.localization import localize
from simple_pe.detectors import detectors
import lal

import cartopy.crs as ccrs


'''For plotting the L shaped interferometer marker have to manually draw them as in this cell:
'''
from matplotlib.path import Path

verts = [
    (0., 1.), # left, top
    (0., 0.), # left, bottom
    (1., 0.), # right, bottom
    ]
codes = [Path.MOVETO,
         Path.LINETO,
         Path.LINETO,
         ]
path = Path(verts, codes)

net_state = "design"
method = 'time'
found_thresh = 5.0
net_thresh = 12.0
loc_thresh = 4.0
duty_cycle = 1.0

BNS_min_chirp_mass = 1.4 * 2**(-1./5)


savefig = True
savetxt = False


# set up trials
dra, ddec = np.pi/8.0, np.pi/16.0
[ra,dec] = np.mgrid[-np.pi+dra:np.pi:dra,-np.pi/2 + ddec:np.pi/2:ddec]

ra = ra.flatten()
dec = dec.flatten()
Dco = 200.
psi = 0.
cosi = 1.

gps_time = 999995380
gps = lal.LIGOTimeGPS(gps_time,0)
gmst = lal.GreenwichMeanSiderealTime(gps)

ntrials = len(ra)

params = {}
params['distance'] = Dco
params['gps'] = gps
params['coa-phase'] = 0.
params['polarization'] = psi
params['inclination'] = np.arccos(cosi)
params['mass1'] = 1.4
params['mass2'] = 1.4
params['RAdeg'] = 0.
params['DEdeg'] = 0.

num_found, num_loc, all_lists = {}, {}, {}
Ms={}

n = localize.Network()
n.set_configuration(net_state, found_thresh, loc_thresh, duty_cycle)

nf, nl = 0, 0
all_list = []
for trial,r in enumerate(ra):
    params['RAdeg'] = np.degrees(r)
    params['DEdeg'] = np.degrees(dec[trial])
    x = localize.Event(n, params=params)
    x.add_network(n)
    all_list.append(x)
    if x.detected:
        nf += 1
        if x.localized >= 3:
            nl += 1
            x.localize_all()

# store in network dictionaries
all_list
num_found = nf
num_loc= nl

#check all were localized
print('%s events were not localized by %s' % (ntrials - num_loc, net_state) )


patches=[]
detect=[]
for ev in all_list:
    detect.append(ev.detected)
    try: patches.append(ev.patches[method])
    except: KeyError
patches = np.array(patches)
print("%s: Fraction localized to one patch = %.3f" %
      (net_state, (1. * sum(patches == 1)/len(patches))) )


# Get the localization matrix for each point

plt.figure(figsize=[20,20])
ax = plt.axes(projection=ccrs.Mollweide())
data_crs = ccrs.PlateCarree()
ax.coastlines()

for i, ev in enumerate(all_list):
    if ev.detected and ev.localized >= 3:
        phi, theta = ev.localization['time'].make_ellipse()
        ax.plot(np.degrees(phi), np.degrees(theta), 'g', transform=data_crs)
    else:
        ra = (ev.ra - gmst) % (2*np.pi)
        ax.plot(np.degrees(ev.ra - gmst), np.degrees(ev.dec) ,'rx', markersize=6, markeredgewidth=1,
                transform=data_crs)

for ifo in ev.ifos:
    i = ev.__getattribute__(ifo)
    phi, theta = detectors.phitheta(i.location/np.linalg.norm(i.location))
    ax.plot(np.degrees(phi), np.degrees(theta), marker=path, markersize=25,
            markerfacecolor='w', markeredgecolor='k', markeredgewidth=4,
            transform=data_crs)

plt.title(net_state, fontsize=24)
if savefig:
   plt.savefig('%s_sky_ellipses.png' % net_state)



plt.figure(figsize=[20,20])
ax = plt.axes(projection=ccrs.Mollweide())
data_crs = ccrs.PlateCarree()
ax.coastlines()

for i, ev in enumerate(all_list):
    if not ev.detected:
        ra = (ev.ra - gmst) % (2*np.pi)
        ax.plot(np.degrees(ev.ra - gmst), np.degrees(ev.dec) ,'#ff0325', marker = 'x',
                markersize=6, markeredgewidth=1, transform=data_crs)
    elif ev.localized < 3:
        ra = (ev.ra - gmst) % (2*np.pi)
        ax.plot(np.degrees(ev.ra - gmst), np.degrees(ev.dec) ,'#0346ff', marker = 'x',
                markersize=6, markeredgewidth=1, transform=data_crs)
    else:
        phi, theta = ev.localization['time'].make_ellipse()
        ax.plot(np.degrees(phi), np.degrees(theta), 'g', transform=data_crs)

for ifo in ev.ifos:
    i = ev.__getattribute__(ifo)
    phi, theta = detectors.phitheta(i.location/np.linalg.norm(i.location))
    ax.plot(np.degrees(phi), np.degrees(theta), marker=path, markersize=25,
            markerfacecolor='w', markeredgecolor='k', markeredgewidth=4, transform=data_crs)

plt.title(net_state, fontsize = 24)
if savefig:
   plt.savefig('%s_sky_ellipses_found.png' % net_state)

