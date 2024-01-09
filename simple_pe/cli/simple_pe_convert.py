from simple_pe.param_est import pe
from pesummary.io import read
from pesummary.core.command_line import CheckFilesExistAction
from argparse import ArgumentParser

__authors__ = [
    "Charlie Hoy <charlie.hoy@ligo.org>",
]


def command_line():
    """Define the command line arguments for `simple_pe_convert`
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
        "--outdir",
        help="Directory to store the output",
        default="./",
    )
    return parser


def main(args=None):
    """Main interface for `simple_pe_convert`
    """
    parser = command_line()
    opts, _ = parser.parse_known_args(args=args)
    samples = pe.SimplePESamples(read(opts.posterior).samples_dict)
    samples.generate_all_posterior_samples(
        function=pe._add_chi_align
    )
    samples.generate_all_posterior_samples(
        function=pe._component_spins_from_chi_align_chi_p
    )
    samples.write(
        file_format="dat", outdir=opts.outdir,
        filename="converted_posterior_samples.dat"
    )
