#! /usr/bin/env python
import numpy as np
import pylab as plt
from argparse import ArgumentParser
import os

from simple_pe import detectors
from simple_pe import localization
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

'''For plotting the L shaped interferometer marker have to m
anually draw them as in this cell:
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


def calc_network_response(events):
    """
    Calculate the network alignment factor for a 2D list
    of events sorted by the 2d ra and dec grids.
    """
    response = np.zeros_like(events, dtype=float)
    for i, ev in np.ndenumerate(events):
        response[i] = np.linalg.norm(ev.get_f())
    response /= np.linalg.norm(ev.get_data('sigma'))
    return response


def calc_alignment_factor(events):
    """
    Calculate the network alignment factor for a 2D list of
    events sorted by the 2d ra and dec grids.
    """
    alignment = np.zeros_like(events, dtype=float)
    for i, ev in np.ndenumerate(events):
        alignment[i] = ev.get_f()[1] / ev.get_f()[0]
    return alignment


def command_line():
    """Define the command line arguments for `simple_pe_localization_ellipses`
    """
    parser = ArgumentParser()

    parser.add_argument(
        "--net-state",
        help="the network state (one of those defined in detectors.py)",
        default="design",
        type=str,
    )
    parser.add_argument(
        "--npoints",
        help="Number of points in right-ascension and declination to use",
        default=100,
        type=int,
    )
    parser.add_argument(
        "--outdir",
        help="Directory to store the output",
        default="./",
    )

    return parser


def main(args=None):
    """Main interface for `simple_pe_analysis`
    """
    parser = command_line()
    opts, _ = parser.parse_known_args(args=args)

    if not os.path.isdir(opts.outdir):
        os.mkdir(opts.outdir)

    found_thresh = 0.0
    net_thresh = 1.0
    loc_thresh = 0.0
    duty_cycle = 1.0

    # Run a set of fake events, fixed distance, face on
    dra, ddec = 2 * np.pi / opts.npoints, np.pi / opts.npoints
    [ra, dec] = np.mgrid[-np.pi:np.pi + dra * 0.9:dra,
                -np.pi / 2:np.pi / 2 + ddec * 0.9:dra]

    # These parameters aren't used other than to get the network response
    params = {'distance': 40.,
              'polarization': 0.,
              'coa-phase': 0.,
              'inclination': 0.,
              'RAdeg': 0.,
              'DEdeg': 0.,
              'gps': 999995381,
              'mass1': 1.35,
              'mass2': 1.35}

    n = detectors.Network(threshold=net_thresh)
    n.set_configuration(opts.net_state, found_thresh, loc_thresh, duty_cycle)

    all_events = np.zeros_like(ra, dtype=object)
    for i, r in np.ndenumerate(ra):
        params['RAdeg'] = np.degrees(r)
        params['DEdeg'] = np.degrees(dec[i])
        ev = localization.Event.from_params(params=params)
        ev.add_network(n)
        all_events[i] = ev

    f_response = calc_network_response(all_events)
    sky_coverage = np.sum((f_response > (np.amax(f_response) / np.sqrt(2)))
                          * np.cos(dec)) / np.sum(np.cos(dec))

    fname = "%s_sky_coverage.txt" % opts.net_state
    f = open(fname, "w")
    f.write('%s: best sensitivity = %.2f, sky coverage (Schutz FOM 2) = %.2f\n'
            % (opts.net_state, f_response.max(), sky_coverage))

    alphas = calc_alignment_factor(all_events)
    pol_sky = np.sum((alphas > np.amax(alphas) / np.sqrt(2)) * np.cos(dec)) \
              / np.sum(np.cos(dec))
    pol_source = np.sum((alphas > np.amax(alphas) / np.sqrt(2)) * np.cos(dec)
                        * f_response ** 3) / np.sum(np.cos(dec) * f_response
                                                    ** 3)
    f.write("%s: mean alpha = %.2f,"
            "fract of sky with alpha > alpha_max/sqrt(2) = %.2f\n"
            % (opts.net_state, np.sum(alphas * np.cos(dec)) /
               np.sum(np.cos(dec)), pol_sky))
    f.write('%s: mean (source weighted) alpha: %.2f\n'
            % (opts.net_state, np.sum(alphas * np.cos(dec) * f_response ** 3) \
               / np.sum(np.cos(dec) * f_response ** 3)))
    f.close()

    v = np.linspace(0.0, 1.0, 41)
    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=ccrs.Mollweide())
    ax.coastlines()
    for ifo in n.ifos:
        ax.plot(np.degrees(getattr(n, ifo).longitude),
                np.degrees(getattr(n, ifo).latitude),
                marker=path, markersize=20,
                markeredgewidth=4, markerfacecolor='None',
                markeredgecolor='m', transform=data_crs)

    cf = ax.contourf(np.degrees(ra), np.degrees(dec), f_response, v,
                     cmap=plt.cm.viridis, transform=data_crs)
    plt.colorbar(cf, fraction=0.046, pad=0.04)
    plt.savefig("%s/%s_sky_sens.png" % (opts.outdir, opts.net_state),
                dpi=200,
                bbox_inches='tight')

    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=ccrs.Mollweide())
    ax.coastlines()
    for ifo in n.ifos:
        ax.plot(np.degrees(getattr(n, ifo).longitude),
                np.degrees(getattr(n, ifo).latitude),
                marker=path, markersize=20,
                markeredgewidth=4, markerfacecolor='None',
                markeredgecolor='m', transform=data_crs)

    cf = ax.contourf(np.degrees(ra), np.degrees(dec),
                     f_response ** 3, v, cmap=plt.cm.viridis,
                     transform=ccrs.PlateCarree())
    plt.colorbar(cf, fraction=0.046, pad=0.04)
    plt.savefig("%s/%s_sky_rate.png" % (opts.outdir, opts.net_state),
                dpi=200,
                bbox_inches='tight')

    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=ccrs.Mollweide())
    ax.coastlines()
    for ifo in n.ifos:
        ax.plot(np.degrees(getattr(n, ifo).longitude),
                np.degrees(getattr(n, ifo).latitude),
                marker=path, markersize=20,
                markeredgewidth=4, markerfacecolor='None',
                markeredgecolor='m', transform=data_crs)

    cf = ax.contourf(np.degrees(ra), np.degrees(dec), alphas,
                     v, cmap=plt.cm.viridis, transform=ccrs.PlateCarree())
    plt.colorbar(cf, fraction=0.046, pad=0.04)

    plt.savefig("%s/%s_2nd_pol.png" % (opts.outdir, opts.net_state),
                dpi=200,
                bbox_inches='tight')


if __name__ == "__main__":
    main()
