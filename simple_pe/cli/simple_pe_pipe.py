#! /usr/bin/env python

from argparse import ArgumentParser
from pesummary.core.command_line import ConfigAction as _ConfigAction
import lalsimulation as ls
import pycondor
import os
import numpy as np
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
    parser = ArgumentParser(
        parents=[_analysis_command_line(), _filter_command_line()],
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
        help="Configuration file containing command line arguments"
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
        "--use_bayestar_localization",
        action="store_true",
        default=False,
        help="use the bayestar localization. --sid must also be provided"
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


class Dag(object):
    """Base Dag object to handle the creation of the DAG

    Parameters
    ----------
    opts: argparse.Namespace
        Namespace containing the command line arguments
    """
    def __init__(self, opts):
        self.opts = opts
        string = "%s/{}" % (opts.outdir)
        self.error = string.format("error")
        self.log = string.format("log")
        self.output = string.format("output")
        self.submit = string.format("submit")
        self.dagman = pycondor.Dagman(name="simple_pe", submit=self.submit)

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

    def build(self):
        """Build the pycondor dag
        """
        self.dagman.build()
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
        """Write a bash script containing all of the command lines used
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
        self.create_pycondor_job()

    @property
    def arguments(self):
        string_args = [
            "approximant", "f_low", "delta_f", "f_high", "minimum_data_length",
            "seed"
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
            self.opts.sid, self.opts.trigger_parameters,
            self.opts.use_bayestar_localization
        )
        self.opts.strain, self.opts.channels = self._prepare_strain(
            self.opts.strain, self.opts.trigger_time, self.opts.channels,
            self.opts.injection
        )
        self.create_pycondor_job()

    @property
    def arguments(self):
        string_args = [
            "trigger_parameters", "approximant", "f_low", "f_high",
            "minimum_data_length", "seed", "snr_threshold"
        ]
        dict_args = ["strain", "channels", "asd", "psd"]
        list_args = ["metric_directions"]
        args = self._format_arg_lists(string_args, dict_args, list_args)
        args += [["--outdir", f"{self.opts.outdir}/output"]]
        return " ".join([item for sublist in args for item in sublist])

    def _prepare_trigger_parameters(self, sid, trigger_parameters, use_bayestar_localization):
        """
        """
        os.makedirs(f"{self.opts.outdir}/output", exist_ok=True)
        if trigger_parameters is not None:
            filename = trigger_parameters
        if sid is not None:
            from pesummary.gw.gracedb import get_gracedb_data
            import json
            try:
                gid = get_gracedb_data(sid, superevent=True, info="preferred_event")
            except AttributeError:
                gid = get_gracedb_data(sid, superevent=True, info="preferred_event_data")
            if trigger_parameters is None:
                logger.info("Grabbing search data from gracedb for trigger_parameters")
                data = get_gracedb_data(gid)
                template_data = data["extra_attributes"]["SingleInspiral"][0]
                json_data = {
                    "mass_1": template_data["mass1"], "mass_2": template_data["mass2"],
                    "spin_1z": template_data["spin1z"], "spin_2z": template_data["spin2z"],
                    "time": data["gpstime"], "chi_p": 0.2, "tilt": 0.1,
                    "coa_phase": 0.
                }
                logger.info("Using the following trigger_parameters:")
                for param, item in json_data.items():
                    logger.info(f"{param} = {item}")
                filename = f"{self.opts.outdir}/output/trigger_parameters.json"
                logger.debug(f"Saving template_parameters to {filename}")
                with open(filename, "w") as f:
                    json.dump(json_data, f)
        if trigger_parameters is None and sid is None:
            raise ValueError(
                "Please provide a file containing the trigger parameters "
                "or a superevent ID to download the trigger parameters "
                "from GraceDB"
            )
        if not use_bayestar_localization:
            return filename
        elif sid is None:
            raise ValueError(
                "Unable to use --use_bayestar_localization when --sid "
                "is not provided"
            )
        from ligo.gracedb.rest import GraceDb
        from ligo.gracedb.exceptions import HTTPError
        from ligo.skymap.io.fits import read_sky_map
        from ligo.skymap.postprocess.util import posterior_max
        logger.info("Grabbing localization data from gracedb for trigger_parameters")
        out_filename = f"{self.opts.outdir}/output/{gid}_bayestar.fits"
        client = GraceDb("https://gracedb.ligo.org/api/")
        with open(out_filename, "wb") as f:
            try:
                r = client.files(gid, "bayestar.fits")
            except HTTPError:
                r = client.files(gid, "bayestar.multiorder.fits,0")
            f.write(r.read())
        skymap, _ = read_sky_map(out_filename)
        _max = posterior_max(skymap)
        with open(filename, "r") as f:
            template_parameters = json.load(f)
        template_parameters["ra"] = np.radians(_max.ra.value)
        template_parameters["dec"] = np.radians(_max.dec.value)
        template_parameters["psi"] = np.random.uniform(0, np.pi, size=1)[0]
        logger.info("Using the following localization parameters:")
        for key in ["ra", "dec", "psi"]:
            logger.info(f"{key} = {template_parameters[key]}")
        logger.debug(f"Saving template_parameters to {filename}")
        with open(filename, "w") as f:
            json.dump(template_parameters, f)
        return filename

    def _prepare_strain(self, strain, trigger_time, channels, injection):
        """Prepare strain data

        Parameters
        ----------
        strain: dict
            dictionary containing strain data. Key must be {ifo}:{channel} and
            value must be path to gwf file. If channel = 'gwosc' then value must
            be the name of the GW signal that you wish to analyse. When
            channel = 'gwosc', data is downloaded with
            `gwpy.timeseries.TimeSeries.fetch_open_data`
        trigger_time:
        channels:
        injection:
        """
        from gwpy.timeseries import TimeSeries
        from gwosc.datasets import event_gps
        _strain = {}
        _channels = {}
        for ifo, value in channels.items():
            if "gwosc" in value.lower():
                if trigger_time is None:
                    raise ValueError(
                        "Please provide the name of the event you wish to analyse via "
                        "the trigger_time argument"
                    )
                gps = event_gps(trigger_time)
                start, stop = int(gps) + 512, int(gps) - 512
                logger.info(
                    f"Fetching strain data with: "
                    f"TimeSeries.fetch_open_data({ifo}, {start}, {stop})"
                )
                open_data = TimeSeries.fetch_open_data(ifo, start, stop)
                _channel = open_data.name
                open_data.name = f"{ifo}:{_channel}"
                open_data.channel = f"{ifo}:{_channel}"
                os.makedirs(f"{self.opts.outdir}/output", exist_ok=True)
                filename = (
                    f"{self.opts.outdir}/output/{ifo}-{_channel}-{int(gps)}.gwf"
                )
                logger.debug(f"Saving strain data to {filename}")
                open_data.write(filename)
                _channels[ifo] = _channel
                _strain[ifo] = filename
            elif "inj" in value.lower():
                # make waveform with independent code: pycbc.waveform.get_td_waveform
                from pycbc.waveform import get_td_waveform, taper_timeseries
                from pycbc.detector import Detector
                import json
                if not os.path.isfile(injection):
                    raise FileNotFoundError(f"Unable to find file: {injection}")
                with open(injection, "r") as f:
                    injection_params = json.load(f)
                # convert to pycbc convention
                params_to_convert = [
                    "mass_1", "mass_2", "spin_1x", "spin_1y", "spin_1z",
                    "spin_2x", "spin_2y", "spin_2z"
                ]
                logger.info("Generating injection with parameters using pycbc:")
                for param, item in injection_params.items():
                    logger.info(f"{param} = {item}")
                for param in params_to_convert:
                    injection_params[param.replace("_", "")] = injection_params.pop(param)
                hp, hc = get_td_waveform(**injection_params)
                hp.start_time += injection_params["time"]
                hc.start_time += injection_params["time"]
                ra = injection_params["ra"]
                dec = injection_params["dec"]
                psi = injection_params["psi"]
                ht = Detector(ifo).project_wave(hp, hc, ra, dec, psi)
                ht = taper_timeseries(ht, tapermethod="TAPER_STARTEND")
                prepend = int(512 / ht.delta_t)
                ht.append_zeros(prepend)
                ht.prepend_zeros(prepend)
                strain = TimeSeries.from_pycbc(ht)
                strain = strain.crop(injection_params["time"] - 512, injection_params["time"] + 512)
                strain.name = f"{ifo}:HWINJ_INJECTED"
                strain.channel = f"{ifo}:HWINJ_INJECTED"
                os.makedirs(f"{self.opts.outdir}/output", exist_ok=True)
                filename = (
                    f"{self.opts.outdir}/output/{ifo}-INJECTION.gwf"
                )
                logger.debug("Saving injection to {filename}")
                strain.write(filename)
                _channels[ifo] = "HWINJ_INJECTED"
                _strain[ifo] = filename
            else:
                import ast
                if trigger_time is not None:
                    gps = float(trigger_time)
                    start, stop = int(gps) - 512, int(gps) + 512
                    logger.info(
                        f"Fetching strain data with: "
                        f"TimeSeries.get('{ifo}:{value}', start={start}, end={stop}, "
                        f"verbose=False, allow_tape=True,).astype(dtype=np.float64, "
                        f"subok=True, copy=False)"
                    )
                    data = TimeSeries.get(
                        f"{ifo}:{value}", start=start, end=stop, verbose=False, allow_tape=True,
                    ).astype(dtype=np.float64, subok=True, copy=False)
                    filename = (
                        f"{self.opts.outdir}/output/{ifo}-{value}-{int(gps)}.gwf"
                    )
                    logger.debug(f"Saving strain data to {filename}")
                    data.write(filename)
                    _channels[ifo] = value
                    _strain[ifo] = filename
                elif ifo not in strain.keys():
                    raise ValueError(f"Please provide a gwf file for {ifo}")
                elif os.path.isfile(strain[ifo]):
                    _channels[ifo] = value
                    _strain[ifo] = strain[ifo]
                else:
                    raise ValueError("Unable to grab strain data")
        return _strain, _channels


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
                "--parameters", "chirp_mass", "symmetric_mass_ratio", "chi_align",
                "theta_jn", "luminosity_distance"
            ]
        ]
        sp = ls.SimInspiralGetSpinSupportFromApproximant(
            getattr(ls, self.opts.approximant)
        )
        if sp > 2:
            args[-1] += ["chi_p"]
        return " ".join([item for sublist in args for item in sublist])


def main(args=None):
    """Main interface for `simple_pe_pipe`
    """
    parser = command_line()
    opts, _ = parser.parse_known_args(args=args)
    logger.info(opts)
    MainDag = Dag(opts)
    FilterJob = FilterNode(opts, MainDag)
    CornerJob = CornerNode(opts, MainDag)
    AnalysisJob = AnalysisNode(opts, MainDag)
    AnalysisJob.add_child(CornerJob.job)
    FilterJob.add_child(AnalysisJob.job)
    MainDag.build()


if __name__ == "__main__":
    main()
