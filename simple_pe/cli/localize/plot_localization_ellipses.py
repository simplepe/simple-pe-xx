#! /usr/bin/env python

import numpy as np
import pylab as plt
import os
from argparse import ArgumentParser

from simple_pe import localization
from simple_pe import detectors
import cartopy.crs as ccrs

'''
For plotting the L shaped interferometer marker have to manually draw them 
as in this cell:
'''
from matplotlib.path import Path

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


def command_line():
    """Define the command line arguments for `simple_pe_localization_ellipses`
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--seed",
        help="random seed to set for reproducibility",
        default=123456789,
        type=int
    )
    parser.add_argument(
        "--net-state",
        help="the network state (one of those defined in detectors.py)",
        default="design",
        type=str,
    )
    parser.add_argument(
        "--method",
        help=(
            "localization method."
            "One of ('time', 'coh', 'left', 'right', 'marg')"
        ),
        default="time",
        type=str,
    )
    parser.add_argument(
        "--found-thresh",
        help="Threshold at which signal is observed in single detector",
        default=5.0,
        type=float,
    )
    parser.add_argument(
        "--net-thresh",
        help="Network threshold for detection",
        default=12.0,
        type=float,
    )
    parser.add_argument(
        "--loc-thresh",
        help="Threshold for a detector to contribute to "
             "localization",
        default=4.0,
        type=float,
    )
    parser.add_argument(
        "--duty-cycle",
        help="Duty cycle for each detector (assumed equal and independent "
             "for all detectors)",
        default=1.,
        type=float,
    )
    parser.add_argument(
        "--npoints",
        help="Number of points in right-ascension and declination to use",
        default=16,
        type=int,
    )
    parser.add_argument(
        "--distance",
        help="Distance at which to simulate signals",
        default=200.,
        type=float,
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
    np.random.seed(opts.seed)

    if not os.path.isdir(opts.outdir):
        os.mkdir(opts.outdir)

    # set up trials
    dra, ddec = 2 * np.pi / opts.npoints, np.pi / opts.npoints
    [ra, dec] = np.mgrid[-np.pi + dra:np.pi:dra,
                -np.pi / 2 + ddec:np.pi / 2:ddec]
    ra = ra.flatten()
    dec = dec.flatten()

    params = {'distance': opts.distance,
              'gps': float(999995380),
              'coa-phase': 0.,
              'polarization': 0.,
              'inclination': 0.,
              'mass1': 1.4,
              'mass2': 1.4,
              'RAdeg': 0.,
              'DEdeg': 0.}

    n = detectors.Network()
    n.set_configuration(opts.net_state,
                        opts.found_thresh,
                        opts.loc_thresh,
                        opts.duty_cycle)

    nf, nl = 0, 0
    all_list = []
    for trial, (r,d) in enumerate(zip(ra, dec)):
        params['RAdeg'] = np.degrees(r)
        params['DEdeg'] = np.degrees(d)
        ev = localization.Event.from_params(params=params)
        ev.add_network(n)
        all_list.append(ev)
        if ev.detected:
            nf += 1
            if ev.localized >= 3:
                nl += 1
                ev.localize_all()

    # Plot the localization matrix for each point
    plt.figure(figsize=[20, 20])
    ax = plt.axes(projection=ccrs.Mollweide())
    data_crs = ccrs.PlateCarree()
    ax.coastlines()

    for i, ev in enumerate(all_list):
        if ev.detected and ev.localized >= 3:
            phi, theta = ev.localization['time'].make_ellipse()
            ax.plot(np.degrees((phi - np.pi) % (2 * np.pi) + np.pi),
                    np.degrees(theta), 'g', transform=data_crs)
        else:
            ax.plot(np.degrees(ev.ra - ev.gmst), np.degrees(ev.dec), 'rx',
                    markersize=6, markeredgewidth=1,
                    transform=data_crs)

    for ifo in ev.ifos:
        i = ev.__getattribute__(ifo)
        phi, theta = detectors.phitheta(i.location / np.linalg.norm(i.location))
        ax.plot(np.degrees(phi), np.degrees(theta), marker=path, markersize=25,
                markerfacecolor='w', markeredgecolor='k', markeredgewidth=4,
                transform=data_crs)

    plt.title(opts.net_state, fontsize=24)
    plt.savefig('%s_sky_ellipses.png' % opts.net_state)

    plt.figure(figsize=[20, 20])
    ax = plt.axes(projection=ccrs.Mollweide())
    data_crs = ccrs.PlateCarree()
    ax.coastlines()

    for i, ev in enumerate(all_list):
        if not ev.detected:
            ax.plot(np.degrees(ev.ra - ev.gmst), np.degrees(ev.dec), '#ff0325',
                    marker='x',
                    markersize=6, markeredgewidth=1, transform=data_crs)
        elif ev.localized < 3:
            ax.plot(np.degrees(ev.ra - ev.gmst), np.degrees(ev.dec), '#0346ff',
                    marker='x',
                    markersize=6, markeredgewidth=1, transform=data_crs)
        else:
            phi, theta = ev.localization['time'].make_ellipse()
            ax.plot(np.degrees((phi - np.pi) % (2 * np.pi) + np.pi),
                    np.degrees(theta), 'g', transform=data_crs)

    for ifo in ev.ifos:
        i = ev.__getattribute__(ifo)
        phi, theta = detectors.phitheta(i.location / np.linalg.norm(i.location))
        ax.plot(np.degrees(phi), np.degrees(theta), marker=path, markersize=25,
                markerfacecolor='w', markeredgecolor='k', markeredgewidth=4,
                transform=data_crs)

    plt.title(opts.net_state, fontsize=24)
    plt.savefig('%s_sky_ellipses_found.png' % opts.net_state)


if __name__ == "__main__":
    main()



