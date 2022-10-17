#! /usr/bin/env python

import os
from argparse import ArgumentParser
from pesummary.io import read
from pesummary.core.command_line import CheckFilesExistAction
from simple_pe.param_est.pe import SimplePESamples

__authors__ = [
    "Charlie Hoy <charlie.hoy@ligo.org>",
]


def command_line():
    """Define the command line arguments for `simple_pe_corner`
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--posterior",
        help=(
            "File containing posterior samples. Must be compatible with "
            "pesummary.io.read"
        ),
        required=True,
        action=CheckFilesExistAction
    )
    parser.add_argument(
       "--truth",
       help="File containing the injected values",
       action=CheckFilesExistAction,
       default=None
    )
    parser.add_argument(
        "--outdir",
        help="Directory to store the output",
        default="./",
    )
    parser.add_argument(
        "--parameters",
        help="Parameters to include in the corner plot",
        default=[
            "chirp_mass", "symmetric_mass_ratio", "chi_align", "theta_jn",
            "luminosity_distance", "chi_p"
        ],
        nargs="+",
    )
    return parser


def main(args=None):
    """
    """
    parser = command_line()
    opts, _ = parser.parse_known_args(args=args)
    samples = SimplePESamples(read(opts.posterior).samples_dict)
    samples.generate_all_posterior_samples()
    if opts.truth is not None:
        truth = SimplePESamples(read(opts.truth).samples_dict)
        truth.generate_all_posterior_samples()
        truth = [truth.get(param, [None])[0] for param in opts.parameters]
    else:
        truth = None
    fig = samples.plot(
        type="corner", quantiles=[0.05, 0.5, 0.95], show_titles=True,
        title_kwargs={"fontsize": 12}, parameters=opts.parameters,
        truths=truth
    )
    if not os.path.isdir(opts.outdir):
        os.makedirs(opts.outdir)
    fig.savefig(f"{opts.outdir}/corner.png")
    fig.close()


if __name__ == "__main__":
    main()
