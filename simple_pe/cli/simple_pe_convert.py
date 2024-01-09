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
    parser.add_argument(
        "--chip_to_spin1x",
        help="Assign chi_p to spin_1x",
        default=False,
        action="store_true"
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
        function=pe._component_spins_from_chi_align_chi_p,
        chip_to_spin1x=opts.chip_to_spin1x
    )
    samples["network_precessing_snr"] = samples.pop("rho_p")
    samples["network_33_multipole_snr"] = samples.pop("rho_33")
    samples.write(
        file_format="dat", outdir=opts.outdir,
        filename="converted_posterior_samples.dat"
    )
