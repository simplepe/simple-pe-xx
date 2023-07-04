#! /usr/bin/env python

import os
from argparse import ArgumentParser
from pesummary.io import read
from pesummary.core.command_line import DictionaryAction
from .simple_pe_filter import (
    _load_psd_from_file, _estimate_data_length_from_template_parameters
)
from simple_pe.param_est import result
import numpy as np

__author__ = [
    "Charlie Hoy <charlie.hoy@ligo.org>",
    "Stephen Fairhurst <stephen.fairhurst@ligo.org>"
]


def command_line():
    """Define the command line arguments for `simple_pe_analysis`
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--seed",
        help="random seed to set for reproducibility",
        default=123456789,
        type=int
    )
    parser.add_argument(
        "--peak_parameters",
        help=(
            "JSON file containing peak parameters generated with "
            "the `simple_pe_analysis` executable"
        ),
    )
    parser.add_argument(
        "--peak_snrs",
        help=(
            "JSON file containing peak SNRs generated with the "
            "`simple_pe_analysis` executable"
        ),
    )
    parser.add_argument(
        "--asd",
        help=(
            "ASD files to use for the analysis. Must be provided as a space "
            "separated dictionary, e.g. H1:path/to/file L1:path/to/file"
        ),
        nargs="+",
        default={},
        action=DictionaryAction,
    )
    parser.add_argument(
        "--psd",
        help=(
            "PSD files to use for the analysis. Must be provided as a space "
            "separated dictionary, e.g. H1:path/to/file L1:path/to/file"
        ),
        nargs="+",
        default={},
        action=DictionaryAction,
    )
    parser.add_argument(
        "--approximant",
        help="Approximant to use for the analysis",
        default="IMRPhenomXPHM",
    )
    parser.add_argument(
        "--f_low",
        help="Low frequency cutoff to use for the analysis",
        default=15.,
        type=float,
    )
    parser.add_argument(
        "--delta_f",
        help="Difference in frequency samples to use for PSD generation",
        default=0.0625,
        type=float
    )
    parser.add_argument(
        "--f_high",
        help=(
            "High frequency cutoff to use for the analysis. This also sets "
            "the sample rate (defined as f_high * 2)"
        ),
        default=8192.,
        type=float,
    )
    parser.add_argument(
        "--minimum_data_length",
        help="Minimum data length to use for the analysis",
        default=16,
        type=int,
    )
    parser.add_argument(
        "--outdir",
        help="Directory to store the output",
        default="./",
    )
    parser.add_argument(
        "--metric_directions",
        help="Directions to calculate metric",
        nargs="+",
        default=['chirp_mass', 'symmetric_mass_ratio', 'chi_align', 'chi_p']
    )
    parser.add_argument(
        "--precession_directions",
        help="Directions for precession",
        nargs="+",
        default=['symmetric_mass_ratio', 'chi_align', 'chi_p']
    )
    parser.add_argument(
        "--multipole_directions",
        help="Directions for higher order multipoles",
        nargs="+",
        default=['chirp_mass', 'symmetric_mass_ratio', 'chi_align']
    )
    parser.add_argument(
        "--distance_directions",
        help="Directions for distance",
        nargs="+",
        default=['chirp_mass', 'symmetric_mass_ratio', 'chi_align']
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
    peak_parameters = read(opts.peak_parameters).samples_dict
    snrs = read(opts.peak_snrs).samples_dict
    data_len = _estimate_data_length_from_template_parameters(
        peak_parameters, opts.f_low, minimum_data_length=opts.minimum_data_length
    )
    psd = _load_psd_from_file(
        opts.psd, opts.asd, int(opts.f_high * 2 / (opts.delta_f * 2) + 1),
        data_len, opts.delta_f, opts.f_low,
    )
    if 'chi_p2' in opts.metric_directions and 'chi_p2' not in peak_parameters:
        peak_parameters['chi_p2'] = peak_parameters['chi_p']**2
    data_from_matched_filter = {
        "template_parameters": {
            k: peak_parameters[k][0] for k in opts.metric_directions
        },
        "snrs": snrs,
        "alpha_net": peak_parameters['net_alpha'][0],
        "distance_face_on": peak_parameters['distance_face_on'][0],
        "sigma": peak_parameters['sigma'][0]
    }
    pe_result = result.Result(
        f_low=opts.f_low, psd=psd["hm"], approximant=opts.approximant,
        data_from_matched_filter=data_from_matched_filter
    )
    _ = pe_result.generate_samples_from_aligned_spin_template_parameters(
        opts.metric_directions, opts.precession_directions, opts.multipole_directions,
        opts.distance_directions, interp_points=5
    )
    pe_result.samples_dict.write(
        outdir=opts.outdir, filename="posterior_samples.dat", overwrite=True,
        file_format="dat"
    )


if __name__ == "__main__":
    main()
