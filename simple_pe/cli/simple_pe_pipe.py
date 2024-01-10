#! /usr/bin/env python

from argparse import ArgumentParser
from pesummary.core.command_line import ConfigAction as _ConfigAction
import lalsimulation as ls
import pycondor
import os
from . import logger

__authors__ = [
    "Charlie Hoy <charlie.hoy@ligo.org>",
]


class ConfigAction(_ConfigAction):
    @staticmethod
    def dict_from_str(string, delimiter=":"):
        mydict = _ConfigAction.dict_from_str(string, delimiter=delimiter)
        for key, item in mydict.items():
            if isinstance(item, list):
                mydict[key] = item[0]
        return mydict


def command_line():
    """Define the command line arguments for `simple_pe_pipe`
    """
    from .simple_pe_analysis import command_line as _analysis_command_line
    from .simple_pe_filter import command_line as _filter_command_line
    from .simple_pe_datafind import command_line as _datafind_command_line
    parser = ArgumentParser(
        parents=[_analysis_command_line(), _filter_command_line(),
                 _datafind_command_line()],
        conflict_handler='resolve'
    )
    remove = ["--peak_parameters", "--peak_snrs"]
    for action in parser._actions[::-1]:
        if len(action.option_strings) and action.option_strings[0] in remove:
            parser._handle_conflict_resolve(
                None, [(action.option_strings[0], action)]
            )
    parser.add_argument(
        "config_file", nargs="?", action=ConfigAction,
        help="Configuration file containing command line arguments",
        default=None
    )
    parser.add_argument(
        "--sid",
        help=(
            "superevent ID for the event you wish to analyse. If "
            "provided, and --trigger_parameters is not provided, the "
            "trigger parameters are downloaded from the best matching "
            "search template on GraceDB."
        )
    )
    parser.add_argument(
        "--gid",
        help=(
            "GraceDB ID for the event you wish to analyse. If "
            "provided, and --trigger_parameters is not provided, the "
            "trigger parameters are downloaded from the best matching "
            "search template on GraceDB. There is no need to provide both "
            "--sid and --gid; --gid will be used if provided."
        )
    )
    parser.add_argument(
        "--use_bayestar_localization",
        action="store_true",
        default=False,
        help="use the bayestar localization. --sid/--gid must also be provided"
    )
    parser.add_argument(
        "--convert_samples",
        action="store_true",
        default=False,
        help="run the `simple_pe_convert` executable as part of the workflow"
    )
    parser.add_argument(
        "--generate_corner",
        action="store_true",
        default=False,
        help="run the `simple_pe_corner` executable as part of the workflow"
    )
    parser.add_argument(
        "--truth",
        help="File containing the injected values. Used only for plotting",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--accounting_group_user",
        help="Accounting group user to use for this workflow",
    )
    parser.add_argument(
        "--accounting_group",
        help="Accounting group to use for this workflow",
    )
    return parser


def _gid_from_sid(sid):
    from pesummary.gw.gracedb import get_gracedb_data
    try:
        return get_gracedb_data(sid, superevent=True, info="preferred_event")
    except AttributeError:
        return get_gracedb_data(sid, superevent=True,
                                info="preferred_event_data")


def get_trigger_parameters(sid, gid):
    """
    Obtain the trigger parameters either from the trigger_parameters file
    or by reading them from GraceDB using the sid/gid

    Parameters
    ----------
    sid: str
        GraceDB ID for the event you wish to analyse
    gid: str
        Superevent GraceDB ID for the event you wish to analyse
    """
    from pesummary.gw.gracedb import get_gracedb_data

    if sid is not None and gid is not None:
        raise ValueError(
            "SID and GID both specified. "
            "Please provide either an SID or a GID"
        )
    elif sid is None and gid is None:
        raise ValueError(
            "Neither SID and GID specified. "
            "Please provide either an SID or a GID"
        )
    elif sid is not None and gid is None:
        gid = _gid_from_sid(sid)

    logger.info("Grabbing search data from gracedb for trigger_parameters")
    logger.info(f"Using the gid: {gid}")
    data = get_gracedb_data(gid)
    template_data = data["extra_attributes"]["SingleInspiral"][0]
    trigger_params = {
                "mass_1": template_data["mass1"],
                "mass_2": template_data["mass2"],
                "spin_1z": template_data["spin1z"],
                "spin_2z": template_data["spin2z"],
                "time": data["gpstime"], "chi_p": 0.2, "tilt": 0.1,
                "coa_phase": 0.
    }
    logger.info("Using the following trigger_parameters:")
    for param, item in trigger_params.items():
        logger.info(f"{param} = {item}")

    return trigger_params


class Dag(object):
    """Base Dag object to handle the creation of the DAG

    Parameters
    ----------
    opts: argparse.Namespace
        Namespace containing the command line arguments
    """
    def __init__(self, opts):
        self.opts = opts
        string = "%s/{}" % opts.outdir
        self.error = string.format("error")
        self.log = string.format("log")
        self.output = string.format("output")
        self.submit = string.format("submit")
        self.dagman = pycondor.Dagman(name="simple_pe", submit=self.submit)
        self.submit_file = os.path.join(self.dagman.submit, '{}.dag'.format(
            self.dagman.name))

    @property
    def bash_file(self):
        return os.path.join(self.submit, "bash_simple_pe.sh")

    @property
    def output_directories(self):
        dirs = {
            "error": self.error, "log": self.log, "output": self.output,
            "submit": self.submit
        }
        return dirs

    def _rename_dag(self):
        """
        """
        new_name = self.dagman.submit_file.replace(".submit", ".dag")
        os.rename(self.dagman.submit_file, new_name)
        self.dagman.submit_file = new_name

    def build(self):
        """Build the pycondor dag
        """
        self.dagman.build()
        self._rename_dag()
        self.write_bash_script()
        print("Dag Generation complete.")
        print("To submit jobs to condor run:\n")
        print("$ condor_submit_dag {}\n".format(self.dagman.submit_file))
        print("To analyse jobs locally run:\n")
        print("$ bash {}\n".format(self.bash_file))

    def build_submit(self):
        """Submit the pycondor dag
        """
        self.dagman.build_submit()

    def write_bash_script(self):
        """Write a bash script containing all the command lines used
        """
        with open(self.bash_file, "w") as f:
            f.write("#!/usr/bin/env bash\n\n")
            independent_jobs = []
            p_dependent_jobs = []
            pc_dependent_jobs = []
            for node in self.dagman.nodes:
                if len(node.parents) and not len(node.children):
                    p_dependent_jobs.append(node)
                elif len(node.parents):
                    pc_dependent_jobs.append(node)
                else:
                    independent_jobs.append(node)
            for node in independent_jobs + pc_dependent_jobs + p_dependent_jobs:
                f.write("# {}\n".format(node.name))
                f.write(
                    "# PARENTS {}\n".format(
                        " ".join([job.name for job in node.parents])
                    )
                )
                f.write(
                    "# CHILDREN {}\n".format(
                        " ".join([job.name for job in node.children])
                    )
                )
                job_str = "{} {}\n\n".format(node.executable, node.args[0].arg)
                job_str = job_str.replace("$(Cluster)", "0")
                job_str = job_str.replace("$(Process)", "0")
                f.write(job_str)


class Node(object):
    """Base node object to handle condor job creation

    Parameters
    ----------
    opts: argparse.Namespace
        Namespace containing the command line arguments
    dag: Dag
        Dag object to control the generation of the DAG
    """
    def __init__(self, opts, dag):
        self.opts = opts
        self.dag = dag
        self.dagman = self.dag.dagman
        self._executable = self.get_executable("simple_pe_pipe")

    @staticmethod
    def get_executable(executable):
        """Find the path to the executable

        Parameters
        ----------
        executable: str
            the name of the executable you wish to find
        """
        from distutils.spawn import find_executable
        exe = find_executable(executable)
        if exe is None:
            return executable
        return exe

    @property
    def executable(self):
        return self._executable

    @property
    def request_memory(self):
        return "8 GB"

    @property
    def request_disk(self):
        return "1 GB"

    @property
    def request_cpus(self):
        return 1

    @property
    def getenv(self):
        return True

    @property
    def notification(self):
        return "never"

    @property
    def universe(self):
        return "vanilla"

    @property
    def extra_lines(self):
        return [
            "accounting_group = {}".format(self.opts.accounting_group),
            "accounting_group_user = {}".format(self.opts.accounting_group_user)
        ]

    def _format_arg_lists(self, string_args, dict_args, list_args):
        args = [
            [f"--{param}", str(getattr(self.opts, param))] for param in
            string_args if getattr(self.opts, param) is not None
        ]
        args += [
            [f"--{param}", " ".join([val for val in getattr(self.opts, param)])]
            for param in list_args if (getattr(self.opts, param) is not None)
            and len(getattr(self.opts, param))
        ]
        args += [
            [
                f"--{param}", " ".join([
                    f"{key}:{item}" for key, item in
                    getattr(self.opts, param).items()
                ])
            ] for param in dict_args if (getattr(self.opts, param) is not None)
            and len(getattr(self.opts, param))
        ]
        return args

    def add_parent(self, parent):
        """Add a parent to the node

        Parameters
        ----------
        parent:

        """
        self.job.add_parent(parent)

    def add_child(self, child):
        """Add a child to the node

        Parameters
        ----------
        child: varaha.condor.Node
            child node you wish to add to your job
        """
        self.job.add_child(child)

    def create_pycondor_job(self):
        self.job = pycondor.Job(
            name=self.job_name, executable=self.executable,
            submit=self.dag.submit, error=self.dag.error,
            log=self.dag.log, output=self.dag.output,
            request_memory=self.request_memory, request_disk=self.request_disk,
            request_cpus=self.request_cpus, getenv=self.getenv,
            universe=self.universe, extra_lines=self.extra_lines,
            dag=self.dagman, arguments=self.arguments,
            notification=self.notification
        )


class AnalysisNode(Node):
    """Node to handle the generation of the main analysis job

    Parameters
    ----------
    opts: argparse.Namespace
        Namespace containing the command line arguments
    dag: Dag
        Dag object to control the generation of the DAG
    """
    job_name = "analysis"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._executable = self.get_executable("simple_pe_analysis")
        if self.opts.use_bayestar_localization:
            self.opts.bayestar_localization = \
                self._get_bayestar_localization(self.opts.sid, self.opts.gid)
        self.create_pycondor_job()

    @property
    def arguments(self):
        string_args = [
            "approximant", "f_low", "delta_f", "f_high", "minimum_data_length",
            "seed", "bayestar_localization", "snr_threshold",
            "localization_method", "neffective"
        ]
        dict_args = ["asd", "psd"]
        list_args = ["metric_directions", "precession_directions"]
        args = self._format_arg_lists(string_args, dict_args, list_args)
        args += [["--outdir", f"{self.opts.outdir}/output"]]
        args += [
            ["--peak_parameters", f"{self.opts.outdir}/output/peak_parameters.json"],
            ["--peak_snrs", f"{self.opts.outdir}/output/peak_snrs.json"]
        ]
        return " ".join([item for sublist in args for item in sublist])

    def _get_bayestar_localization(self, sid, gid):
        if sid is None and gid is None:
            raise ValueError(
                "Unable to use --use_bayestar_localization when --sid/--gid "
                "is not provided"
            )
        if gid is None:
            gid = _gid_from_sid(sid)

        from ligo.gracedb.rest import GraceDb
        from ligo.gracedb.exceptions import HTTPError
        logger.info("Grabbing localization data from gracedb")
        loc_filename = f"{self.opts.outdir}/output/{gid}_bayestar.fits"
        client = GraceDb("https://gracedb.ligo.org/api/")
        with open(loc_filename, "wb") as f:
            options = ["bayestar.fits", "bayestar.fits.gz",
                       "bayestar.multiorder.fits,0",
                       "bayestar_pycbc_C01.fits.gz"]
            for opt in options:
                try:
                    r = client.files(gid, opt)
                except HTTPError:
                    continue
            try:
                f.write(r.read())
            except UnboundLocalError:
                raise ValueError(
                    "Unable to grab localization data from gracedb. "
                    "This could be because you do not have "
                    "authentication, or the file has a non-standard "
                    "name."
                )
            except Exception:
                raise
        return loc_filename


class FilterNode(Node):
    """Node to handle the generation of the main match filter job

    Parameters
    ----------
    opts: argparse.Namespace
        Namespace containing the command line arguments
    dag: Dag
        Dag object to control the generation of the DAG
    """
    job_name = "filter"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._executable = self.get_executable("simple_pe_filter")
        self.opts.trigger_parameters = self._prepare_trigger_parameters(
            self.opts.sid, self.opts.gid, self.opts.trigger_parameters,
        )
        if not len(self.opts.psd) and not len(self.opts.asd):
            self.opts.psd = self._get_search_psd(self.opts.sid, self.opts.gid)
        self.create_pycondor_job()

    @property
    def arguments(self):
        string_args = [
            "trigger_parameters", "approximant", "f_low", "f_high",
            "minimum_data_length", "seed",
        ]
        dict_args = ["asd", "psd"]
        if "strain_cache" in self.opts.strain:
            string_args += ["strain"]
        else:
            dict_args += ["strain", "channels"]
        list_args = ["metric_directions"]
        args = self._format_arg_lists(string_args, dict_args, list_args)
        args += [["--outdir", f"{self.opts.outdir}/output"]]
        return " ".join([item for sublist in args for item in sublist])

    def _prepare_trigger_parameters(self, sid, gid, trigger_parameters):
        """
        Obtain the trigger parameters either from the trigger_parameters file
        or by reading them from GraceDB using the sid/gid

        Parameters
        ----------
        sid: str
            GraceDB ID for the event you wish to analyse
        gid: str
            Superevent GraceDB ID for the event you wish to analyse
        trigger_parameters: str or dict
            Either a dict containing the trigger parameters or the name of a
            file which contains the parameters
        """
        import json
        os.makedirs(f"{self.opts.outdir}/output", exist_ok=True)
        if trigger_parameters is None and sid is None and gid is None:
            raise ValueError(
                "Please provide a file containing the trigger parameters "
                "or a superevent ID to download the trigger parameters "
                "from GraceDB"
            )

        if trigger_parameters is None:
            # read parameters from graceDB using sid/gid
            trigger_parameters = get_trigger_parameters(sid, gid)

        if isinstance(trigger_parameters, dict):
            filename = f"{self.opts.outdir}/output/trigger_parameters.json"
            with open(filename, "w") as f:
                _trigger_parameters = {key: float(item) for key, item in
                                       trigger_parameters.items()}
                json.dump(_trigger_parameters, f)
        else:
            filename = trigger_parameters

        return filename

    def _get_search_psd(self, sid, gid):
        if sid is None and gid is None:
            raise ValueError(
                "No PSD/ASD provided and unable to grab one from GraceDB "
                "because no SID/GID has been provided. "
                "Please either provide PSD/ASDs via "
                "the --psd/--asd flags or a SID/GID via the --sid/--gid flags"
            )
        elif gid is None:
            gid = _gid_from_sid(sid)
        logger.info("Grabbing PSD/ASD data from coinc.xml file")
        from ligo.gracedb.rest import GraceDb
        from gwpy.frequencyseries import FrequencySeries
        client = GraceDb("https://gracedb.ligo.org/api/")
        coinc_filename = f"{self.opts.outdir}/output/{gid}_coinc.xml"
        with open(coinc_filename, "wb") as f:
            r = client.files(gid, filename="coinc.xml")
            f.write(r.read())
        psd_dict = {}
        for ifo in ["H1", "L1", "V1"]:
            try:
                psd = FrequencySeries.read(coinc_filename, instrument=ifo)
                psd_filename = f"{self.opts.outdir}/output/{ifo}_psd.txt"
                psd.write(target=psd_filename, format="txt")
                psd_dict[ifo] = psd_filename
                logger.info(f"Found PSD for {ifo}")
            except ValueError:
                continue
        if not len(psd_dict):
            raise ValueError(f"Unable to extract PSD from {coinc_filename}")
        return psd_dict


class DataFindNode(Node):
    """Node to handle fetching the strain data to analyse

    Parameters
    ----------
    opts: argparse.Namespace
        Namespace containing the command line arguments
    dag: Dag
        Dag object to control the generation of the DAG
    """
    job_name = "datafind"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._executable = self.get_executable("simple_pe_datafind")
        if not self.opts.injection and not self.opts.trigger_time:
            logger.info("Setting analysis time from supplied SID/GID")
            trigger_params = get_trigger_parameters(self.opts.sid,
                                                    self.opts.gid)
            self.opts.trigger_time = trigger_params["time"]
        self.create_pycondor_job()

    @property
    def universe(self):
        return "local"

    @property
    def arguments(self):
        string_args = ["outdir", "trigger_time", "injection"]
        args = self._format_arg_lists(string_args, ["channels"], [])
        return " ".join([item for sublist in args for item in sublist])


class CornerNode(Node):
    """Node to handle the generation of a corner plot showing the posterior

    Parameters
    ----------
    opts: argparse.Namespace
        Namespace containing the command line arguments
    dag: Dag
        Dag object to control the generation of the DAG
    """
    job_name = "corner"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._executable = self.get_executable("simple_pe_corner")
        self.create_pycondor_job()

    @property
    def universe(self):
        return "local"

    @property
    def arguments(self):
        args = self._format_arg_lists(["truth"], [], [])
        args += [
            ["--outdir", f"{self.opts.outdir}/output"],
            ["--posterior", f"{self.opts.outdir}/output/posterior_samples.dat"]
        ]
        args += [
            [
                "--parameters", "chirp_mass", "symmetric_mass_ratio",
                "chi_align", "theta_jn", "luminosity_distance"
            ]
        ]
        sp = ls.SimInspiralGetSpinSupportFromApproximant(
            getattr(ls, self.opts.approximant)
        )
        if sp > 2:
            args[-1] += ["chi_p"]
        return " ".join([item for sublist in args for item in sublist])


class ConvertNode(Node):
    """Node to handle generating additional posteriors generated by simple_pe

    Parameters
    ----------
    opts: argparse.Namespace
        Namespace containing the command line arguments
    dag: Dag
        Dag object to control the generation of the DAG
    """
    job_name = "convert"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._executable = self.get_executable("simple_pe_convert")
        self.create_pycondor_job()

    @property
    def arguments(self):
        args = [
            ["--outdir", f"{self.opts.outdir}/output"],
            ["--posterior", f"{self.opts.outdir}/output/posterior_samples.dat"],
            ["--chip_to_spin1x"]
        ]
        return " ".join([item for sublist in args for item in sublist])


class PostProcessingNode(Node):
    """Node to handle the generation of PESummary pages to show the posterior

    Parameters
    ----------
    opts: argparse.Namespace
        Namespace containing the command line arguments
    dag: Dag
        Dag object to control the generation of the DAG
    """
    job_name = "postprocessing"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._executable = self.get_executable("summarypages")
        self.create_pycondor_job()

    @property
    def arguments(self):
        string_args = [
            "approximant", "f_low"
        ]
        dict_args = ["psd"]
        args = self._format_arg_lists(string_args, dict_args, [])
        args += [
            ["--webdir", f"{self.opts.outdir}/webpage"],
            ["--gw"], ["--no_ligo_skymap"], ["--disable_interactive"],
            ["--label", "simple_pe"],
            ["--add_to_corner", "theta_jn", "network_precessing_snr",
             "network_33_multipole_snr"],
        ]
        if self.opts.convert_samples:
            args += [["--samples", f"{self.opts.outdir}/output/converted_posterior_samples.dat"]]
        else:
            args += [["--samples", f"{self.opts.outdir}/output/posterior_samples.dat"]]
        if self.opts.config_file is not None:
            args += [["--config", f"{self.opts.config_file}"]]
        return " ".join([item for sublist in args for item in sublist])


def main(args=None):
    """Main interface for `simple_pe_pipe`
    """
    parser = command_line()
    opts, _ = parser.parse_known_args(args=args)
    logger.info(opts)
    MainDag = Dag(opts)
    DATAFIND = False
    if opts.strain is None:
        DataFindJob = DataFindNode(opts, MainDag)
        DATAFIND = True
        opts.strain = f"{opts.outdir}/output/strain_cache.json"
    FilterJob = FilterNode(opts, MainDag)
    AnalysisJob = AnalysisNode(opts, MainDag)
    PostProcessingJob = PostProcessingNode(opts, MainDag)
    if opts.generate_corner:
        CornerJob = CornerNode(opts, MainDag)
        AnalysisJob.add_child(CornerJob.job)
    if opts.convert_samples:
        ConvertJob = ConvertNode(opts, MainDag)
        AnalysisJob.add_child(ConvertJob.job)
        ConvertJob.add_child(PostProcessingJob.job)
    else:
        AnalysisJob.add_child(PostProcessingJob.job)
    FilterJob.add_child(AnalysisJob.job)
    if DATAFIND:
        DataFindJob.add_child(FilterJob.job)
    MainDag.build()


if __name__ == "__main__":
    main()
